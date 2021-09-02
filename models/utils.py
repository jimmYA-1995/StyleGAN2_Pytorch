import torch.nn as nn


REGISTRY = {}


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def build_pose_encoder(bottom_res: int, out_channels: int, **pose_encoder_kwargs) -> nn.Module:

    pose_encoder_name = pose_encoder_kwargs.pop('name')
    return REGISTRY['pose'].get(pose_encoder_name)(bottom_res, out_channels, **pose_encoder_kwargs)


def register(key):
    def wrapper(cls):
        registry = REGISTRY.setdefault(key, {})
        registry[cls.__name__] = cls

        return cls
    return wrapper
