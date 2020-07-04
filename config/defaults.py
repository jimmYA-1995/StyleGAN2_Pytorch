# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()
_C.OUT_DIR = 'results'
_C.N_SAMPLE = 64
_C.GPUS = (0, 1, 2, 3, 4, 5, 6, 7)
_C.WORKERS = 4
_C.RESOLUTION = 256
_C.N_SAMPLE = 64


_C.DATASET = CN()
_C.DATASET.DATASET = 'MultiChannelDataset'
_C.DATASET.ROOTS = ['/root/notebooks/data/mpii']
_C.DATASET.SOURCE = ['images']
_C.DATASET.LOAD_IN_MEM = False
_C.DATASET.FLIP = True

_C.MODEL = CN()
_C.MODEL.N_MLP = 8
_C.MODEL.LATENT_SIZE = 512
_C.MODEL.CHANNEL_MULTIPLIER = 2
_C.MODEL.EXTRA_CHANNEL = 2
_C.MODEL.NV_WEIGHTS_PATH = ''

_C.TRAIN = CN()
_C.TRAIN.ITERATION = 80000
_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.LR = 0.002
_C.TRAIN.R1 = 10
_C.TRAIN.PATH_REGULARIZE = 2
_C.TRAIN.PATH_BATCH_SHRINK = 2
_C.TRAIN.G_REG_EVERY = 4
_C.TRAIN.D_REG_EVERY = 16
_C.TRAIN.STYLE_MIXING_PROB = 0.9
_C.TRAIN.CKPT = ''
_C.TRAIN.SAVE_CKPT_EVERY = 2500
_C.TRAIN.CKPT_MAX_KEEP = 10

_C.EVAL = CN()
_C.EVAL.METRICS = ""
_C.EVAL.FID = CN()
_C.EVAL.FID.EVERY = 0
_C.EVAL.FID.BATCH_SIZE = 32
_C.EVAL.FID.N_SAMPLE = 50000
_C.EVAL.FID.INCEPTION_CACHE = "inception_cache.pkl"


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
