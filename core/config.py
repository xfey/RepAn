"""Configuration file (powered by YACS)."""

import os
import sys
import argparse
from yacs.config import CfgNode


# Global config object
_C = CfgNode(new_allowed=True)
cfg = _C


# -------------------------------------------------------- #
# Data Loader options
# -------------------------------------------------------- #
_C.LOADER = CfgNode(new_allowed=True)

_C.LOADER.DATASET = "cifar10"

# stay empty to use "./data/$dataset" as default
_C.LOADER.DATAPATH = ""

_C.LOADER.SPLIT = [0.8, 0.2]

# whether using val dataset (imagenet only)
_C.LOADER.USE_VAL = False

_C.LOADER.NUM_CLASSES = 10

_C.LOADER.NUM_WORKERS = 8

_C.LOADER.PIN_MEMORY = True

# batch size of training and validation
# type: int or list(different during validation)
# _C.LOADER.BATCH_SIZE = [256, 128]
_C.LOADER.BATCH_SIZE = 256

# augment type using by ImageNet only
# chosen from ['default', 'auto_augment_tf']
_C.LOADER.TRANSFORM = "default"


# ------------------------------------------------------------------------------------ #
# Optimizer options in network
# ------------------------------------------------------------------------------------ #
_C.OPTIM = CfgNode(new_allowed=True)

# Base learning rate, init_lr = OPTIM.BASE_LR * NUM_GPUS
_C.OPTIM.BASE_LR = 0.1

_C.OPTIM.MIN_LR = 1.e-3


# Learning rate policy select from {'cos', 'exp', 'step'}
_C.OPTIM.LR_POLICY = "cos"
# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = [30, 60, 90]
# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1


# Momentum
_C.OPTIM.MOMENTUM = 0.9
# Momentum dampening
_C.OPTIM.DAMPENING = 0.0
# Nesterov momentum
_C.OPTIM.NESTEROV = False


_C.OPTIM.WEIGHT_DECAY = 5e-4

_C.OPTIM.GRAD_CLIP = 10.0

_C.OPTIM.MAX_EPOCH = 100
# Warm up epochs
_C.OPTIM.WARMUP_EPOCH = 0
# Start the warm up from init_lr * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1
# Ending epochs
_C.OPTIM.FINAL_EPOCH = 0



# ------------------------------------------------------------------------------------ #
# Options for model training
# ------------------------------------------------------------------------------------ #
_C.TRAIN = CfgNode(new_allowed=True)

_C.TRAIN.IM_SIZE = 32

_C.TRAIN.LABEL_SMOOTH = 0.1

_C.TRAIN.WEIGHTS = ""


# -------------------------------------------------------- #
# Model testing options
# -------------------------------------------------------- #
_C.TEST = CfgNode(new_allowed=True)

_C.TEST.IM_SIZE = 224

# using specific batchsize for testing
# using search.batch_size if this value keeps -1
_C.TEST.BATCH_SIZE = -1


# -------------------------------------------------------- #
# Misc options
# -------------------------------------------------------- #

_C.CUDNN_BENCH = True

_C.LOG_PERIOD = 10

_C.EVAL_PERIOD = 1

_C.SAVE_PERIOD = 5

_C.NUM_GPUS = 1

_C.OUT_DIR = "exp/"

_C.DETERMINSTIC = True

_C.RNG_SEED = 1



# -------------------------------------------------------- #

def dump_cfgfile(cfg_dest="config.yaml"):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, cfg_dest)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfgfile(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_configs(_path=None):
    """Load config from command line arguments and set any specified options.
       How to use: python xx.py --cfg path_to_your_config.cfg test1 0 test2 True
       opts will return a list with ['test1', '0', 'test2', 'True'], yacs will compile to corresponding values
    """
    if _path is None:
        parser = argparse.ArgumentParser(description="Config file options.")
        parser.add_argument("--cfg", required=True, type=str)
        parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(1)
        args = parser.parse_args()
        _C.merge_from_file(args.cfg)
        _C.merge_from_list(args.opts)
    else:
        _C.merge_from_file(_path)
