import numpy as np
import contextlib
from typing import Optional, Sequence, Tuple, Union

import torch


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions: Sequence[float], num_list: Sequence[int]):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """

        self.center_fractions = center_fractions
        self.num_list = num_list

    def __call__(
        self, shape: Sequence[int], ind: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None, 
    ) -> torch.Tensor:
        raise NotImplementedError


def syn_maskset_from_mask(
    mask_type_str: str,
    center_fractions: Sequence[float],
    num_list: Sequence[int],
    ) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
    """
    if mask_type_str == "random":
        return SynMasksetFunc(center_fractions, num_list)
    else:
        raise Exception(f"{mask_type_str} not supported")


class SynMasksetFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __call__(
        self, init_mask: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Create the mask.

        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(init_mask.shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        init_mask = np.array(init_mask)
        shape = init_mask.shape[-2]
        self.num_list = np.array(self.num_list)
        num_init = init_mask.squeeze().sum()
        maskset = []

        pdf = np.ones_like(init_mask.squeeze())
        pdf[init_mask.squeeze()==1] = 0
        pdf /= np.sum(pdf)      # uniform pdf

        adding_list = (self.num_list - num_init)[1:]
        adding_list = adding_list.astype(np.int)

        draw_idx = np.random.choice(shape, (shape - num_init).astype(np.int), replace=False, p=pdf)

        maskset.append(init_mask.copy())
        for i in adding_list:
            init_mask[..., draw_idx[:i], :] = True
            maskset.append(init_mask.copy())
        
        maskset = torch.from_numpy(np.array(maskset).astype(np.float32))

        return maskset


