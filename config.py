from yacs.config import CfgNode as CN


_VALID_TYPES = {tuple, list, str, int, float, bool}

_C = CN()
_C.description = ''
_C.outdir = 'results'
_C.n_sample = 64
_C.resolution = 256
_C.num_classes = 0

# ------ dataset ------
_C.DATASET = CN()
_C.DATASET.workers = 4
_C.DATASET.dataset = 'MultiChannelDataset'
_C.DATASET.roots = ['/root/notebooks/data/mpii']
_C.DATASET.source = ['images']
_C.DATASET.channels = [3]
_C.DATASET.mean = [0.5, 0.5, 0.5]
_C.DATASET.std = [0.5, 0.5, 0.5]
_C.DATASET.pin_memory = False
_C.DATASET.xflip = True
_C.DATASET.ADA = False
_C.DATASET.ADA_target = 0.6
_C.DATASET.ADA_p = 0
_C.DATASET.ADA_interval = 4
_C.DATASET.ADA_kimg = 500

# ------ Model ------
_C.MODEL = CN()
_C.MODEL.z_dim = 512
# _C.MODEL.CHANNEL_MULTIPLIER = 2
_C.MODEL.extra_channel = 0
_C.MODEL.use_style_encoder = False

_C.MODEL.G_MAP = CN()
_C.MODEL.G_MAP.embed_dim = 0
_C.MODEL.G_MAP.dlatent_dim = 512
_C.MODEL.G_MAP.num_layer = 8
_C.MODEL.G_MAP.num_channel = 512
_C.MODEL.G_MAP.lrmul = 0.01

_C.MODEL.STYLE_ENCODER = CN()
_C.MODEL.STYLE_ENCODER.nf_in = 3
_C.MODEL.STYLE_ENCODER.max_nf = 256
_C.MODEL.STYLE_ENCODER.dlatent_dim = 256

_C.MODEL.G_SYNTHESIS = CN()
_C.MODEL.G_SYNTHESIS.fmap_base = 16384
_C.MODEL.G_SYNTHESIS.fmap_decay = 1.0
_C.MODEL.G_SYNTHESIS.fmap_min = 1
_C.MODEL.G_SYNTHESIS.fmap_max = 512
_C.MODEL.G_SYNTHESIS.use_content_encoder = False

_C.MODEL.G_SYNTHESIS.content_encoder_kwargs = CN()
_C.MODEL.G_SYNTHESIS.content_encoder_kwargs.nf_in = 3
_C.MODEL.G_SYNTHESIS.content_encoder_kwargs.max_nf = 512

# ----- training ------
_C.TRAIN = CN()
_C.TRAIN.iteration = 80000
_C.TRAIN.batch_gpu = 16
_C.TRAIN.lrate = 0.002
_C.TRAIN.r1 = 10
_C.TRAIN.path_reg_gain = 2
_C.TRAIN.path_bs_shrink = 2
_C.TRAIN.Greg_every = 4
_C.TRAIN.Dreg_every = 16
_C.TRAIN.style_mixing_prob = 0.9
_C.TRAIN.ckpt = ''
_C.TRAIN.save_ckpt_every = 2500
_C.TRAIN.ckpt_max_keep = 10
_C.TRAIN.sample_every = 1000

_C.ADA = CN()
_C.ADA.xflip = 1
_C.ADA.rotate90 = 1
_C.ADA.xint = 1
_C.ADA.scale = 1
_C.ADA.rotate = 1
_C.ADA.aniso = 1
_C.ADA.xfrac = 1
_C.ADA.brightness = 1
_C.ADA.contrast = 1
_C.ADA.lumaflip = 1
_C.ADA.hue = 1
_C.ADA.saturation = 1

# ------ evaluation ------
_C.EVAL = CN()
_C.EVAL.metrics = ""
_C.EVAL.FID = CN()
_C.EVAL.FID.every = 0
_C.EVAL.FID.batch_size = 32
_C.EVAL.FID.n_sample = 50000
_C.EVAL.FID.inception_cache = "inception_cache.pkl"
_C.EVAL.FID.sample_dir = ""


def get_cfg_defaults():
    return _C.clone()


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict
