from argparse import ArgumentParser

import fastmri
import torch
from fastmri.data import transforms
from models.humus_net_pdac_singlecoil import HUMUSNet_pdac_singlecoil
from pl_modules.mri_module import MriModule


class PDACModule_singlecoil(MriModule):

    def __init__(
        self,
        lr: float = 0.0001,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        max_epoch: int = 40,
        num_adj_slices: int = 1,
        mask_center: bool = False,
        logger_type='tb',
        **kwargs,
    ):
        """
        Pytorch Lightning module to train and evaluate PDAC. 
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        if 'num_log_images' in kwargs:
            num_log_images = kwargs['num_log_images']
            kwargs.pop('num_log_images', None)
        else:
            num_log_images = 16
            
        super().__init__(num_log_images)
        self.save_hyperparameters()
        
        self.logger_type = logger_type

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.model = HUMUSNet_pdac_singlecoil(**kwargs)
        self.loss = torch.nn.L1Loss()

    
    def cal_interprob_ref_full(self, inter_out_full, gt):
        error = torch.abs(inter_out_full - gt)                      # [N, iter, num_sense, H, W, 2] = [N, 8, 15, 384, 384 ,2]
        error = error.sum(dim=-1).sum(dim=2).sum(dim=2)             # [N, iter, W] = [N, 8, 384]
        eng = torch.abs(gt).sum(dim=-1).sum(dim=2).sum(dim=2)       # [N, iter, W] = [N, 8, 384]

        norm_error = error / eng
        norm_error = torch.sigmoid(norm_error) * 2.0 - 1.0
        norm_error = 1 - norm_error
        
        return norm_error
    
    def forward(self, masked_kspace, mask):
        return self.model(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target, _, _, max_value, _, kspace = batch

        output, inter_out, inter_mask, inter_prob = self(masked_kspace, mask)

        # calculate reference for probability according to the normailized column errors in k-space
        prob_ref_full = self.cal_interprob_ref_full(inter_out, kspace.unsqueeze(1))
        prob_ref = prob_ref_full * inter_mask.squeeze()

        # crop to gt size
        target, output = transforms.center_crop_to_smallest(target, output)

        # training loss
        loss_rec = self.loss(output.unsqueeze(1) / max_value[:, None, None, None], target.unsqueeze(1) / max_value[:, None, None, None])
        loss_prob = self.loss(inter_prob, prob_ref)
        loss = loss_rec + 0.01 * loss_prob

        self.log("train_rec_loss", loss_rec.item())
        self.log("train_prob_loss", loss_prob.item())

        return loss

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, fname, slice_num, max_value, _, kspace = batch
        output, _, _, _ = self.forward(
            masked_kspace, mask
        )

        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(output.unsqueeze(1) / max_value[:, None, None, None], target.unsqueeze(1) / max_value[:, None, None, None]),
        }

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, _, fname, slice_num, _, crop_size = batch
        output = self(masked_kspace, mask)

        # check for FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]
    
    def log_image(self, name, image):
        if self.logger is not None:
            # tensorboard logging (default)
            self.logger.experiment.add_image(name, image, global_step=self.current_epoch)
            
    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint['hyper_parameters']
        pdac_module_singlecoil = PDACModule_singlecoil(**hparams)
        pdac_module_singlecoil.model = HUMUSNet_pdac_singlecoil.load_from_checkpoint(checkpoint_path)

        return pdac_module_singlecoil

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--embed_dim", 
            default=72, 
            type=int, 
            help="Embedding dimension"
        )
        parser.add_argument(
            "--depths",
            nargs="+",
            default=[2, 2, 2],
            type=int,
            help="Number of STLs per RSTB. The length of this array determines the number of blocks in the downsampling direction. The last block is always bottleneck and does not downsample.",
        )
        parser.add_argument(
            "--num_heads",
            nargs="+",
            default=[3, 6, 12],
            type=int,
            help="Number of attention heads in each RSTB.",
        )
        parser.add_argument(
            "--mlp_ratio", 
            default=2., 
            type=float, 
            help="Ratio of mlp hidden dim to embedding dim. Default: 2"
        )
        parser.add_argument(
            "--window_size", 
            default=8, 
            type=int, 
            help="Window size. Default: 8"
        )
        parser.add_argument(
            "--patch_size", 
            default=1, 
            type=int, 
            help="Patch size. Default: 1"
        )
        parser.add_argument(
            "--resi_connection", 
            default='1conv', 
            type=str, 
            help="The convolutional block before residual connection. '1conv'/'3conv'"
        )
        parser.add_argument(
            "--bottleneck_depth", 
            default=2,
            type=int, 
            help="Number of STLs in bottleneck."
        )
        parser.add_argument(
            "--bottleneck_heads", 
            default=24, 
            type=int, 
            help="Number of attention heads in bottleneck."
        )
        parser.add_argument(
            '--conv_downsample_first', 
            default=False,   
            action='store_true',          
            help='If set, downsample image by 2x first via convolutions before passing it to MUST.',
        )
        parser.add_argument(
            '--use_checkpointing', 
            default=False,   
            action='store_true',          
            help='If set, checkpointing is used to save GPU memory.',
        )
        parser.add_argument(
            '--no_residual_learning',
            default=False,
            action='store_true',
            help='By default, residual image is denoised in MUST. Setting this flag will turn off the residual path.',
        )

        # training params (opt)
        parser.add_argument(
            "--lr", 
            default=0.0001, 
            type=float, 
            help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", 
            default=0.1, 
            type=float, 
            help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )
        parser.add_argument(
            "--uniform_train_resolution",
            nargs="+",
            default=None,
            type=int,
            help="If given, training slices will be center cropped / reflection padded to this size to make sure inputs are of the same size.",
        )

        # unrolling params
        parser.add_argument(
            "--num_cascades",
            default=8,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=16,
            type=int,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--no_center_masking",
            default=False,
            action='store_true',
            help="If set, kspace center is not masked when estimating sensitivity maps.",
        )

        return parser