from yacs.config import CfgNode as CN


_VALID_TYPES = {tuple, list, str, int, float, bool}

# Default configuration. Overriden by experiments/<something>.yml with `CfgNode.merge_from_file`
_C = CN()
_C.name = ''
_C.description = ''
_C.outdir = 'results'
_C.n_sample = 64
_C.resolution = 256
_C.classes = []

# ------ dataset ------
_C.DATASET = CN()
_C.DATASET.workers = 4
_C.DATASET.dataset = 'MultiChannelDataset'
_C.DATASET.roots = ['/root/notebooks/data/mpii']
_C.DATASET.source = ['images']
_C.DATASET.channels = [3]
_C.DATASET.mean = [0.5, 0.5, 0.5]
_C.DATASET.std = [0.5, 0.5, 0.5]
_C.DATASET.kwargs = CN(new_allowed=True)
_C.DATASET.kwargs.pose_on = False
_C.DATASET.pin_memory = False
_C.DATASET.xflip = False
_C.DATASET.ADA = False
_C.DATASET.ADA_target = 0.6
_C.DATASET.ADA_p = 0.0
_C.DATASET.ADA_interval = 4
_C.DATASET.ADA_kimg = 500

# ------ Model ------
_C.MODEL = CN()
_C.MODEL.z_dim = 512
_C.MODEL.w_dim = 512

_C.MODEL.MAPPING = CN()
_C.MODEL.MAPPING.num_layers = 8
_C.MODEL.MAPPING.embed_dim = 512  # Force to zero if no label(len(classes) == 1)
_C.MODEL.MAPPING.layer_dim = 512
_C.MODEL.MAPPING.lrmul = 0.01

_C.MODEL.POSE_ENCODER = CN()
_C.MODEL.POSE_ENCODER

_C.MODEL.SYNTHESIS = CN()
_C.MODEL.SYNTHESIS.architecture = 'skip'  # TODO: add other architectures
_C.MODEL.SYNTHESIS.bottom_res = 4
_C.MODEL.SYNTHESIS.pose_on = False
_C.MODEL.SYNTHESIS.pose_encoder_kwargs = CN(new_allowed=True)
_C.MODEL.SYNTHESIS.pose_encoder_kwargs.name = 'DefaultPoseEncoder'
_C.MODEL.SYNTHESIS.channel_base = 32768
_C.MODEL.SYNTHESIS.channel_max = 512

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
_C.EVAL.FID.dataset = ""
_C.EVAL.FID.every = 0
_C.EVAL.FID.batch_size = 32
_C.EVAL.FID.n_sample = 50000
_C.EVAL.FID.inception_cache = "inception_cache.pkl"
_C.EVAL.FID.sample_dir = ""


def get_cfg_defaults():
    """ return local variable use pattern and link some config. together """

    cfg = _C.clone()
    cfg.DATASET.resolution = cfg.resolution
    cfg.DATASET.batch_size = cfg.TRAIN.batch_gpu

    return cfg


# legacy. It seems we can directly unpacking cfgNode to function
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


def override(cfg: CN, item: dict, copy: bool = False) -> CN:
    "only support 1 level override for simplicity"
    if copy:
        cfg = cfg.clone()

    cfg.defrost()
    for key, override_val in item.items():
        setattr(cfg, key, override_val)
    cfg.freeze()

    if copy:
        return cfg
    return None
