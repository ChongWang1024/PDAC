# [CVPR 2024 Highlight] Progressive Divide-and-Conquer via Subsampling Decomposition for Accelerated MRI
This is the official implementation of the CVPR 2024 highlight paper 
["Progressive Divide-and-Conquer via Subsampling Decomposition for Accelerated MRI"](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Progressive_Divide-and-Conquer_via_Subsampling_Decomposition_for_Accelerated_MRI_CVPR_2024_paper.pdf).

*Chong Wang, Lanqing Guo, Yufei Wang, Hao Cheng, Yi Yu, Bihan Wen*

*Nanyang Technological University*

>  ### Abstract
> *Deep unfolding networks (DUN) have emerged as a popular iterative framework for accelerated magnetic resonance imaging (MRI) reconstruction. However, conventional DUN aims to reconstruct all the missing information within the entire null space in each iteration. Thus it could be challenging when dealing with highly ill-posed degradation, often resulting in subpar reconstruction. In this work, we propose a Progressive Divide-And-Conquer (PDAC) strategy, aiming to break down the subsampling process in the actual severe degradation and thus perform reconstruction sequentially. Starting from decomposing the original maximum-a-posteriori problem of accelerated MRI, we present a rigorous derivation of the proposed PDAC framework, which could be further unfolded into an end-to-end trainable network. Each PDAC iteration specifically targets a distinct segment of moderate degradation, based on the decomposition. Furthermore, as part of the PDAC iteration, such decomposition is adaptively learned as an auxiliary task through a degradation predictor which provides an estimation of the decomposed sampling mask. Following this prediction, the sampling mask is further integrated via a severity conditioning module to ensure awareness of the degradation severity at each stage. Extensive experiments demonstrate that our proposed method achieves superior performance on the publicly available fastMRI and Stanford2D FSE datasets in both multi-coil and single-coil settings.*
<!-- ![Intro](/assets/intro.png) -->
<p align="center">
<img src="assets/intro.png" width="600px"/>
</p>

## Installation
The code is implemented with Python 3.8 and PyTorch 2.3.1. Create the conda environment by
```
git clone https://github.com/ChongWang1024/PDAC.git
cd PDAC
conda create -n pdac python==3.8
conda activate pdac
```
Install the dependencies (**Note:** `h5py` should be installed using `conda` to avoid memory leakage issue)
```
pip install fastmri==0.1.1
pip install pytorch-lightning==1.9.4
pip install numpy timm einops PyYAML tensorboardX
conda install h5py
```

## Datasets
### fastMRI knee dataset
The access for fastMRI dataset could be applied at https://fastmri.med.nyu.edu/. You need to download the fastMRI knee dataset with the following files:

- knee_singlecoil_train.tar.gz
- knee_singlecoil_val.tar.gz
- knee_multicoil_train.tar.gz
- knee_multicoil_val.tar.gz

After downloading these files, extract them into the same directory. Make sure that the directory contains exactly the following folders:

- singlecoil_train
- singlecoil_val
- multicoil_train
- multicoil_val

### Stanford 2D FSE datasets

Please follow [these instructions](data/stanford/README.md) to batch-download the Stanford datasets.
Alternatively, they can be downloaded from http://mridata.org volume-by-volume at the following links:

- [Stanford 2D FSE](http://mridata.org/list?project=Stanford%202D%20FSE)

After downloading the .h5 files the dataset has to be converted to a format compatible with fastMRI modules. To create the datasets used in the paper please follow the instructions [here](data/stanford/README.md).

## Training
To train a PDAC model from scratch, please follow the steps as follows

1. Customize the `.yaml` file under `./pdac_examples/config/`

```
data_path          # root directory containing fastMRI data
default_root_dir   # directory to save the experiments data
num_list           # degradation budget schedule for pdac iterations
```
2. Train the model
### fastMRI knee
```
# multi-coil data
python pdac_examples/train_pdac_fastmri.py --config_file pdac_examples/config/fastmri/multicoil/pdac.yaml
```
```
# single-coil data
python pdac_examples/train_pdac_fastmri.py --config_file pdac_examples/config/fastmri/singlecoil/pdac.yaml
```
### Standford 2D FSE
```
python pdac_examples/train_pdac_stanford.py --config_file pdac_examples/config/stanford2d/pdac.yaml
```

## Evaluation
You can directly evaluate the performance of the pre-trained model as follows
### Pretrained models
Please download the corresponding [pretrained models](https://drive.google.com/drive/folders/1CPq3B0ea6wZYuE7V1KedqQkpWnjjOm0A?usp=drive_link) and put them under the directory `./pretrained/`.
### fastMRI knee
```
# multi-coil data
python pdac_examples/eval_pdac_fastmri.py \
--challenge multicoil \
--accelerations 8 \
--center_fractions 0.04 \
--checkpoint_file ./pretrained/pdac_fastmri_multicoil_8x
--data_path DATA_DIR
```
```
# single-coil data
python pdac_examples/eval_pdac_fastmri.py \
--challenge singlecoil \
--accelerations 8 \
--center_fractions 0.04 \
--checkpoint_file ./pretrained/pdac_fastmri_singlecoil_8x
--data_path DATA_DIR
```

- `DATA_DIR`: root directory containing fastMRI data (with folders such as `multicoil_train` and `multicoil_val`)

### Standford 2D FSE
```
python pdac_examples/eval_pdac_stanford.py \
--challenge multicoil \
--accelerations 8 \
--center_fractions 0.04 \
--checkpoint_file ./pretrained/pdac_stanford_multicoil_8x
--data_path DATA_DIR
```
- In this case `DATA_DIR` should point directly to the folder containing the *converted* `.h5` files.

## Acknowledgements
Our implementation is based on [HUMUS-Net](https://github.com/z-fabian/HUMUS-Net) and [fastMRI](https://github.com/facebookresearch/fastMRI).


## Citation
If you find this repo helpful, please cite:
```
@inproceedings{wang2024progressive,
  title={Progressive Divide-and-Conquer via Subsampling Decomposition for Accelerated MRI},
  author={Wang, Chong and Guo, Lanqing and Wang, Yufei and Cheng, Hao and Yu, Yi and Wen, Bihan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25128--25137},
  year={2024}
}
```

