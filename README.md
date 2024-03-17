# RepAn: Enhanced Annealing through Re-parameterization

Anonymous code submission for our paper "RepAn: Enhanced Annealing through Re-parameterization".

This code is implemented with [Pytorch](https://github.com/pytorch/pytorch). We thank every reviewer for their hard work.


[**Abstract**](#abstract) | [**Requirements**](#requirements) | [**Data Preparation**](#data-preparation) | [**Training**](#training) | [**Pre-trained Weights**](#pre-trained-weights)

<p>
<img src="https://img.shields.io/badge/Python-%3E%3D3.7-blue">
<img src="https://img.shields.io/badge/PyTorch-1.9-informational">
</p>

---


<br>



## Abstract

The simulated annealing algorithm aims to improve model convergence through multiple restarts of training. However, existing annealing algorithms overlook the correlation between different cycles, neglecting the potential for incremental learning. We contend that a fixed network structure prevents the model from recognizing distinct features at different training stages. To this end, we propose RepAn, redesigning the irreversible re-parameterization (Rep) method and integrating it with annealing to enhance training. Specifically, the network goes through Rep, expansion, restoration, and backpropagation operations during training, and iterating through these processes in each annealing round. Such a method exhibits good generalization and is easy to apply, and we provide theoretical explanations for its effectiveness. Experiments demonstrate that our method improves baseline performance by $6.38\%$ on the CIFAR-100 dataset and $2.80\%$ on ImageNet, achieving state-of-the-art performance in the Rep field. The code is available in our supplementary material.


## Requirements

To run our code, `python>=3.7` and `pytorch>=1.9` are required. Other versions of `PyTorch` may also work well, but there are potential API differences that can cause warnings to be generated.

Other required packages are listed in the `requirements.txt` file. You can simply install these dependencies by:

```bash
pip install -r requirements.txt
```

Then, set the `$PYTHONPATH` environment variable before running this code:

```bash
export PYTHONPATH=$PYTHONPATH:/Path/to/This/Code
```

(Optional) Set the visible GPU devices:

```bash
export CUDA_VISIBLE_DEVICES=0
```


## Data Preparation

The dataloaders of our code read the dataset files from `$CODE_PATH/data/$DATASET_NAME` by default, and we use lowercase filenames and remove the hyphens. For example, files for CIFAR-10 should be placed (or auto downloaded) under `$CODE_PATH/data/cifar10` directory.

Our method can load pre-trained weights for faster training. To achieve this, place the weights file in the `$CODE_PATH/weights` folder, and modify the configuration files correspondingly.


## Training

Our code reads configuration files from the command line, and can be overriddden by manually adding or modifying. We list several examples for running our code as follows:

#### Baselines

**Normal training** 

```bash
python tools/train/CODE.py --cfg configs/CONFIG_FILE.yaml
```


**Knowledge Distillation (KD) training** 

```bash
python tools/kd/CODE.py --cfg configs/CONFIG_FILE.yaml
```


Examples:

```bash
python tools/train/train_repvgg_cifar.py --cfg configs/rep/rep_c100_normal.yaml
```

```bash
python tools/kd/kd_dbb_cifar.py --cfg configs/dbb/dbb_c10_normal.yaml
```

#### Our Approach

```bash
python tools/cycle/CODE.py --cfg configs/CONFIG_FILE.yaml
```


Examples:

```bash
python tools/cycle/cycle_repvgg_cifar.py --cfg configs/rep/rep_c100_cycle.yaml
```

```bash
python tools/cycle/cycle_dbb_cifar.py --cfg configs/ac/acb_c10_cycle.yaml
```

```bash
python tools/cycle/cycle_dbb_img.py --cfg configs/dbb_IMG_cycle.yaml
```

Other cases can be run by modifying the configuration files in the `configs` folder.
