import os, sys
import pathlib
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )

import pytorch_lightning as pl
import torch
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from  pl_modules.pdac_module import PDACModule

from data.data_transforms import PDACDataTransform
from pl_modules.fastmri_data_module import FastMriDataModule

# Imports for logging and other utility
import yaml
from utils import load_args_from_config
from pytorch_lightning.strategies import DDPStrategy


def cli_main(args):
    pl.seed_everything(args.seed)

    args.num_list = [48, 192, 288, 320, 336, 352, 364, 376, 384]         # customize the sampling list here

    # ------------
    # model
    # ------------
    if args.challenge == 'multicoil':
        model = PDACModule.load_from_checkpoint(args.checkpoint_file)
        hparams = torch.load(args.checkpoint_file)['hyper_parameters']
        num_adj_slices = hparams['num_adj_slices']
        uniform_train_resolution = hparams['img_size']
    else:
        raise ValueError('Single-coil data not supported.')
    model.eval()
    
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
        
    # use fixed masks for val transform
    val_transform = PDACDataTransform(uniform_train_resolution=uniform_train_resolution, mask_func=mask)
    
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path='/home/Dataset/fastMRI/',
        challenge=args.challenge,
        train_transform=None,
        val_transform=val_transform,
        test_transform=None,
        test_split=None,
        test_path=None,
        sample_rate=None,
        volume_sample_rate=1.0,
        batch_size=1,
        num_workers=4,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu", 'gpu')),
        combine_train_val=False,
        train_scanners=args.train_scanners,
        val_scanners=args.val_scanners,
        combined_scanner_val=args.combined_scanner_val,
        num_adj_slices=num_adj_slices,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=False,
    #  limit_val_batches=3
     )
        
    # ------------
    # run
    # ------------
    trainer.validate(model, datamodule=data_module)


def build_args():
    parser = ArgumentParser()

    # basic args
    backend = DDPStrategy(process_group_backend="gloo", find_unused_parameters=False)
    batch_size = 1

    # client arguments
    parser.add_argument(
        '--checkpoint_file', 
        type=pathlib.Path,          
        help='Path to the checkpoint to load the model from.',
        default=''
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
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
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        challenge="multicoil",
        mask_type="random",  # random masks for knee data
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
        accelerations=[8], # default experimental setup: 8x acceleration
        center_fractions=[0.04]
    )
    
    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=1,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator='gpu',  # what distributed version to use
        strategy=backend,
        seed=42,  # random seed
        deterministic=False,  # makes things slower, but deterministic
    )

    args = parser.parse_args()

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()