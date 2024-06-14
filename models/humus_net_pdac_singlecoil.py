from models.humus_pdac_block_singlecoil import HUMUSBlock_pdac_singlecoil
from fastmri.models import NormUnet
import fastmri
import torch.nn as nn
import torch
from typing import List, Tuple, Optional
import torch.utils.checkpoint as checkpoint
from models.humus_pdac_block_singlecoil import PositionalEncoding, MaskPredictor

class HUMUSNet_pdac_singlecoil(nn.Module):

    def __init__(
        self,
        num_list,
        num_cascades: int = 8,
        sens_chans: int = 16,
        sens_pools: int = 4,
        mask_center: bool = True,
        num_adj_slices: int = 1,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
            num_adj_slices: Odd integer, number of adjacent slices used as input
                1: single slice
                n: return (n - 1) / 2 slices on both sides from the center slice
        """
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.use_checkpoint = kwargs['use_checkpoint']
        im_size = kwargs['img_size']
        self.net = VarNetBlock(
                    HUMUSBlock_pdac_singlecoil(in_chans=2*self.num_adj_slices, **kwargs), self.num_adj_slices, im_size)

        self.num_list = num_list

        embed_dim = 66
        self.mask_embedder = nn.Sequential(
                PositionalEncoding(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.SiLU(),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        self.mask_predictor = MaskPredictor()
        self.num_cascades = num_cascades
    

    def update_mask(self ,current_mask, pred_prob, add_list, iter):
        pred_prob = pred_prob[:, :, None, :, None]      # [N, 1, 1, 384, 1]
        pred_prob = pred_prob * (1 - current_mask)
        add_num = add_list[:, iter]

        add_mask = torch.zeros_like(current_mask).to(current_mask)
        for i in range(add_num.shape[0]):
            add_idx = torch.argsort(pred_prob[i].squeeze(), descending=True)[:add_num[i]]
            add_mask[i, :, :, add_idx, :] = 1

        current_mask = current_mask + add_mask

        return current_mask
    

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        center_slice = (self.num_adj_slices - 1) // 2
        
        kspace_pred = masked_kspace.clone()
        current_mask = mask.clone()
        current_prob = current_mask.clone().squeeze(4).squeeze(2).squeeze(1)

        init_num = current_mask.squeeze(4).squeeze(2).squeeze(1).sum(dim=-1)
        actual_num_list = torch.Tensor(self.num_list).to(init_num)
        actual_num_list = actual_num_list.unsqueeze(0).repeat_interleave(init_num.shape[0], dim=0)
        actual_num_list[:, 0] = init_num                                             # adjust the initial sampling number according to the actual subsampled k-sapce
        add_list = actual_num_list[:, 1:] - actual_num_list[:, :-1]

        kspace_out_set = []
        inter_mask_set = []
        inter_prob_set = []
        for i in range(self.num_cascades):
            
            # degradation severity embedding
            prob_fea = self.mask_embedder(current_prob.to(masked_kspace))

            # reconstruction
            if self.use_checkpoint:
                kspace_pred, restore_im = checkpoint.checkpoint(self.net, kspace_pred, masked_kspace, mask, i, current_mask, current_prob, add_list, prob_fea)
            else:
                kspace_pred, restore_im = self.net(kspace_pred, masked_kspace, mask, i, current_mask, current_prob, add_list, prob_fea)
            
            kspace_out_set.append(kspace_pred)

            # predicting a decomposed degradation
            pred_prob = self.mask_predictor(fastmri.fft2c(restore_im.squeeze(dim=1)).permute(0, 3, 1, 2).contiguous())
            current_mask = self.update_mask(current_mask, pred_prob, add_list, i)
            
            # perform degradation accoding to the current sampling mask
            kspace_pred = kspace_pred * current_mask
            current_prob = pred_prob.squeeze(1) * current_mask.squeeze(4).squeeze(2).squeeze(1)

            # select center slice if exists adjacent slices
            kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[center_slice]

            inter_mask_set.append(current_mask)
            inter_prob_set.append(current_prob)

        out = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

        inter_mask_set = torch.stack(inter_mask_set, dim=1)       # (N, iter, 1, 1, W, 1)
        inter_prob_set = torch.stack(inter_prob_set, dim=1)       # (N, iter, W)
        kspace_out_set = torch.stack(kspace_out_set, dim=1)       # (N, iter, sens, H, W, 2)
        
        return out, kspace_out_set, inter_mask_set, inter_prob_set
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint['hyper_parameters']
        model = HUMUSNet_pdac_singlecoil(
                 num_list=hparams['num_list'],
                 num_cascades=hparams['num_cascades'],
                 img_size=hparams['img_size'], 
                 patch_size=hparams['patch_size'], 
                 embed_dim=hparams['embed_dim'], 
                 depths=hparams['depths'], 
                 num_heads=hparams['num_heads'],
                 window_size=hparams['window_size'], 
                 mlp_ratio=hparams['mlp_ratio'], 
                 use_checkpoint=hparams['use_checkpoint'] if 'use_checkpoint' in hparams else True, 
                 resi_connection=hparams['resi_connection'],
                 bottleneck_depth=hparams['bottleneck_depth'],
                 bottleneck_heads=hparams['bottleneck_heads'],
                 conv_downsample_first=hparams['conv_downsample_first'],
                 num_adj_slices=hparams['num_adj_slices'] if 'num_adj_slices' in hparams else 1,
                 sens_chans=hparams['sens_chans'],
                 no_residual_learning=hparams['no_residual_learning'] if 'no_residual_learning' in hparams else False,
            )

        state_dict = checkpoint['state_dict']
        state_dict = {'.'.join(k.split('.')[1:]):v for k, v in state_dict.items() if 'model' in k} 
        model.load_state_dict(state_dict)
        return model
        
class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module, num_adj_slices=1, im_size=None):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.im_size = im_size
        

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        _, c, _, _, _ = sens_maps.shape
        return fastmri.fft2c(fastmri.complex_mul(x.repeat_interleave(c // self.num_adj_slices, dim=1), sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        b, c, h, w, _ = x.shape
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).view(b, self.num_adj_slices, c // self.num_adj_slices, h, w, 2).sum(
            dim=2, keepdim=False
        )
    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    
    @staticmethod
    def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape # 1, 3 * 2, h, w
        #x = x.view(b, 2, c // 2 * h * w)
        x = x.view(b, c, h * w) # 1, 6, h * w

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    @staticmethod
    def unnorm(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    
    def pad_width(self, x):
        pad_w = self.im_size[1] - x.shape[-1] 
        if pad_w > 0:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            pad_w_left = pad_w_right = 0 
        return torch.nn.functional.pad(x, (pad_w_left, pad_w_right, 0, 0), 'reflect'), pad_w_left, pad_w_right
    
    def unpad_width(self, x, pad_w_left, pad_w_right):
        if pad_w_left > 0:
            x = x[:, :, :, pad_w_left:]
        if pad_w_right > 0:
            x = x[:, :, :, :-pad_w_right]
        return x
        

    def apply_model(self, x, mask):
        x = self.complex_to_chan_dim(x)
        if self.im_size is not None:
            x, p_left, p_right = self.pad_width(x)
        x, mean, std = self.norm(x)

        x = self.model(x, mask)

        x = self.unnorm(x, mean, std)
        if self.im_size is not None:
            x = self.unpad_width(x, p_left, p_right)
        x = self.chan_complex_to_last_dim(x)
        return x

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        iter: torch.Tensor,
        current_mask: torch.Tensor,     # [N, 1, 1, 384, 1]
        current_prob: torch.Tensor,     # [N, 384]
        add_list,
        prob_fea
    ) -> torch.Tensor:       
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)

        current_mask = current_mask.to(current_kspace)

        restore_im = self.apply_model(fastmri.ifft2c(current_kspace), prob_fea)        # restor_im [N, 1, 384, 384, 2]   pred_prob [N, 1, 384]
        model_term = fastmri.fft2c(restore_im)
        soft_dc = torch.where(mask, model_term - ref_kspace, zero) * self.dc_weight
        pred_kspace = model_term - soft_dc

        # return pred_kspace, restore_im
        return pred_kspace, fastmri.ifft2c(pred_kspace)
    
    