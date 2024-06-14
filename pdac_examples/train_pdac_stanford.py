import os, sys
import pathlib
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type

from data.data_transforms import PDACDataTransform
from pl_modules.stanford_data_module import StanfordDataModule

import yaml
from pdac_examples.utils import load_args_from_config
import torch.distributed

from  pl_modules.pdac_module import PDACModule

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
import tensorboardX


def build_args():
    parser = ArgumentParser()
    # basic args
    # backend = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False, static_graph=True)
    backend = DDPStrategy(process_group_backend="gloo", find_unused_parameters=False, static_graph=True)
    num_gpus = 1
    batch_size = 1

    # client arguments
    parser.add_argument(
        '--config_file', 
        default='pdac_examples/config/stanford2d/pdac.yaml',   
        type=pathlib.Path,          
        help='If given, experiment configuration will be loaded from this yaml file.',
    )
    parser.add_argument(
        '--verbose', 
        default=False,   
        action='store_true',          
        help='If set, print all command line arguments at startup.',
    )
    parser.add_argument(
        '--logger_type', 
        default='tb',   
        type=str,          
        help='Set Pytorch Lightning training logger. Options "tb" - Tensorboard (default), "wandb" - Weights and Biases',
    )
    parser.add_argument(
        '--experiment_name', 
        default='pdac-stanford',   
        type=str,          
        help='Used with wandb logger to define the project name.',
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[8],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = StanfordDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        mask_type="random",  # random masks for knee data
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    # module config
    parser = PDACModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=8,  # number of unrolled iterations
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.0001,  # Adam learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator='gpu',  # what distributed version to use
        strategy=backend,
        seed=42,  # random seed
        deterministic=False,  # makes things slower, but deterministic
    )

    args = parser.parse_args()
    
    # Load args if config file is given
    if args.config_file is not None:
        args = load_args_from_config(args)
        

    args.learningrate_callback = LearningRateMonitor(logging_interval="epoch")
    args.checkpoint_callback_ssim = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        verbose=True,
        monitor="val_metrics/ssim",
        mode="max",
        filename='epoch{epoch}-ssim{val_metrics/ssim:.4f}',
        auto_insert_metric_name=False,
        save_last=False
    )

    args.checkpoint_callback_psnr = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        verbose=True,
        monitor="val_metrics/psnr",
        mode="max",
        filename='epoch{epoch}-psnr{val_metrics/psnr:.2f}',
        auto_insert_metric_name=False,
        save_last=True
    )

    return args


def cli_main(args):
    if args.verbose:
        print(args.__dict__)
        
    pl.seed_everything(args.seed)
    # ------------
    # model
    # ------------
    model = PDACModule(
        num_list=args.num_list,
        num_cascades=args.num_cascades,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        img_size=args.uniform_train_resolution,
        patch_size=args.patch_size,
        window_size=args.window_size,
        embed_dim=args.embed_dim, 
        depths=args.depths,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio, 
        bottleneck_depth=args.bottleneck_depth,
        bottleneck_heads=args.bottleneck_heads,
        resi_connection=args.resi_connection,
        conv_downsample_first=args.conv_downsample_first,
        num_adj_slices=args.num_adj_slices,
        mask_center=(not args.no_center_masking),
        use_checkpoint=args.use_checkpointing,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        max_epoch=args.max_epochs,
        logger_type=args.logger_type,
    )
    
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )

    # use random masks for train transform, fixed masks for val transform
    train_transform = PDACDataTransform(uniform_train_resolution=args.uniform_train_resolution, mask_func=mask, use_seed=False)
    val_transform = PDACDataTransform(uniform_train_resolution=args.uniform_train_resolution, mask_func=mask)
    test_transform = PDACDataTransform(uniform_train_resolution=args.uniform_train_resolution)
    
    # ptl data module - this handles data loaders
    data_module = StanfordDataModule(
        data_path=args.data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu", "gpu")),
        train_val_seed=args.train_val_seed,
        train_val_split=args.train_val_split,
        num_adj_slices=args.num_adj_slices,
    )

    # ------------
    # trainer
    # ------------
    # set up logger
    if args.logger_type == 'tb':
        logger = True
    elif args.logger_type == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.experiment_name)
    else:
        raise ValueError('Unknown logger type.')
    trainer = pl.Trainer.from_argparse_args(args, 
                                            precision=16,
                                            callbacks=[args.checkpoint_callback_ssim, 
                                                       args.checkpoint_callback_psnr,
                                                       args.learningrate_callback],
                                            logger=logger,
                                            resume_from_checkpoint=args.resume_from,
                                            limit_train_batches=5,
                                            limit_val_batches=5,
                                            )
    
    # Save all hyperparameters to .yaml file in the current log dir
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                save_all_hparams(trainer, args)
    else: 
         save_all_hparams(trainer, args)
            
    # ------------
    # run
    # ------------
    trainer.fit(model, datamodule=data_module)

def save_all_hparams(trainer, args):
    if not os.path.exists(trainer.logger.log_dir):
        os.makedirs(trainer.logger.log_dir, exist_ok=True)
    save_dict = args.__dict__
    save_dict.pop('checkpoint_callback')
    with open(trainer.logger.log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()