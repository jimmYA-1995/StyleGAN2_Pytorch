import random
import pickle
from time import time
from warnings import warn
from pathlib import Path
from typing import List
from collections import namedtuple
from collections.abc import Sequence

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from misc import cords_to_map, draw_pose_from_cords

ALLOW_EXTS = ['jpg', 'jpeg', 'png', 'JPEG']


def ImageFolderDataset(config, resolution, transform=None):
    def image_loader(path):
        img = Image.open(path)
        img = img.resize((resolution, resolution), Image.ANTIALIAS)
        if img.mode == 'L':
            img = img.convert('RGB')
        return img

    def check_valid(img):
        if img is not None:
            return True
        return False

    return ImageFolder(config.roots[0], transform=transform, loader=image_loader, is_valid_file=check_valid)


class DefaultDataset(data.Dataset):

    def __init__(self, cfg, split='train'):
        assert len(cfg.roots) == len(cfg.source) == 1
        assert split in ['train', 'val', 'test', 'all']
        self.cfg = cfg
        self.root = Path(cfg.roots[0]).expanduser()
        self.face_dir = None
        self.fileIDs = None
        self.idx = None
        self.resolution = cfg.resolution
        self.split = split
        self.xflip = cfg.xflip
        self._img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std, inplace=True),
        ])
        self._mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.fileIDs) * 2 if self.xflip else len(self.fileIDs)

    def maybe_xflip(self, img):
        """ xflip if xflip enabled and index > len(ds) / 2,
            no op. otherwise
        """
        assert isinstance(img, Image.Image) and self.idx is not None
        if not self.xflip or self.idx < len(self.fileIDs):
            return img

        return img.transpose(method=Image.FLIP_LEFT_RIGHT)

    def img_transform(self, img):
        img = self.maybe_xflip(img)
        return self._img_transform(img)

    def mask_transform(self, img):
        img = self.maybe_xflip(img)
        return self._mask_transform(img)

    @classmethod
    def worker_init_fn(cls, worker_id):
        """ For reproducibility & randomness in multi-worker mode """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if hasattr(dataset, 'rng'):
            dataset.rng = np.random.default_rng(worker_seed)


class DeepFashion(DefaultDataset):
    def __init__(self, cfg, split='train', pose_on=False, sample=False, num_items=None):
        super().__init__(cfg, split=split)
        self.classes = ["DF_real_face", "DF_real_human"]
        self.pose_on = pose_on
        self.sample = sample

        # File ID
        split_map = pickle.load(open(self.root / 'new_split.pkl', 'rb'))
        self.fileIDs = [ID for IDs in split_map.values() for ID in IDs] if split == 'all' else split_map[split]
        self.fileIDs.sort()
        if num_items:
            self.fileIDs = self.fileIDs[:min(len(self.fileIDs), num_items)]
        self.size = {"DF_real_face": self.__len__(), "DF_real_human": self.__len__()}
        self.labels = np.zeros((len(self),), dtype=int)

        src = cfg.source[0]
        self.face_dir = self.root / src / 'face'
        self.target_dir = self.root / f'r{self.resolution}' / 'images'
        assert self.face_dir.exists() and self.target_dir.exists()
        assert set(self.fileIDs) <= set(p.stem for p in self.face_dir.glob('*.png'))
        assert set(self.fileIDs) <= set(p.stem for p in self.target_dir.glob('*.png'))

        if pose_on:
            self.kp_dir = self.root / 'kp_heatmaps/keypoints'
            assert self.kp_dir.exists() and set(self.fileIDs) <= set(p.stem for p in self.kp_dir.glob('*.pkl'))

    def __getitem__(self, idx):
        self.idx = idx
        try:
            fileID = self.fileIDs[idx % len(self.fileIDs)] if self.xflip else self.fileIDs[idx]
        except IndexError as e:
            print(self.xflip, idx)
            raise RuntimeError(e)

        face = Image.open(self.face_dir / f'{fileID}.png')
        target = Image.open(self.target_dir / f'{fileID}.png')
        assert face.mode == 'RGB' and target.mode == 'RGB'
        face = self.img_transform(face)
        target = self.img_transform(target)

        if self.pose_on:
            kp = pickle.load(open(self.kp_dir / f'{fileID}.pkl', 'rb'))[0][:, (1, 0, 2)]  # [K, (y, x, score)]
            cords = np.where(kp[:, 2:3] > 0.1, kp[:, :2], -np.ones_like(kp[:, :2]))
            heatmaps = cords_to_map(cords, (256, 256), sigma=8)
            if self.xflip and idx < len(self.fileIDs):
                heatmaps = heatmaps[:, ::-1]
            heatmaps = torch.from_numpy(heatmaps.transpose(2, 0, 1).copy())
            if self.sample:
                colors, mask = draw_pose_from_cords(cords.astype(int), (256, 256))
                vis_kp = torch.from_numpy((colors.astype(np.float32) - 127.5 / 127.5).transpose(2, 0, 1).copy())
                return vis_kp, heatmaps
            return face, target, heatmaps

        return face, target


class ConditionalBatchSampler(data.sampler.BatchSampler):
    """ This sampler is for sampling several classes at same time

        Use case:
        1. len(class_indices) == 1 and no_repeat is True:
           It will iterate over a specific class from mutli-class dataset once.

        2. no_repeat is False and num_items is not provided:
           It will iterate depends on the largest class.
           other classes will iteratre again to keep the batch size the same.

        3. no_repeat is False and num_items is provided:
           It will iterate until num_items items.
    """
    def __init__(self,
                 dataset,
                 class_indices: List[int],
                 sample_per_class: int = 1,
                 num_items: int = None,
                 shuffle: bool = False,
                 no_repeat: bool = False,
                 drop_last: bool = False,
                 num_gpus: int = 1,
                 rank: int = 0):
        assert hasattr(dataset, 'classes')
        assert all(0 <= idx < len(dataset.classes) for idx in class_indices)
        if not hasattr(dataset, 'labels'):
            if len(dataset.classes) > 1:
                raise AttributeError("multiclass dataset must have attribute labels")
            labels = np.zeros((len(dataset),), dtype=int)  # all belongs class0
        else:
            assert isinstance(dataset.labels, (Sequence, np.ndarray))
            labels = np.array(dataset.labels)

        if num_items is not None:
            assert isinstance(num_items, int) and num_items > 0
            assert drop_last is False

        if no_repeat:
            assert len(class_indices) == 1, "no_repeat is used for iterate over a specific class once."

        if rank >= num_gpus or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_gpus - 1))

        self.num_gpus = num_gpus
        self.rank = rank
        self.num_classes = len(dataset.classes)
        self.class_indices = class_indices
        self.sample_per_class = sample_per_class
        self.num_items = num_items
        self.batch_size = len(class_indices) * sample_per_class
        self.shuffle = shuffle
        self.no_repeat = no_repeat
        self.drop_last = drop_last
        self.label_indices = []
        data_size = 0
        for c in class_indices:
            label_indices = np.where(labels == c)[0]
            if len(label_indices) == 0:
                raise RuntimeError(f"no data for class{c}")

            if len(label_indices) < sample_per_class:
                warn(f"total samples of class No.{c} is less than required.")

            if len(label_indices) % num_gpus != 0:
                """ keep each replica have same elements to
                    avoid process hangover during torch.distributed.broadcast
                """
                complement = num_gpus - len(label_indices) % num_gpus
                label_indices = np.resize(label_indices, len(label_indices) + complement)

            self.label_indices.append(label_indices[rank::num_gpus])
            data_size = max(data_size, len(self.label_indices[-1]))
        self.data_size = data_size if num_items is None else num_items

    def __iter__(self):
        count = 0
        used_label_indices_count = [0] * len(self.class_indices)
        if self.shuffle:
            for label_indices in self.label_indices:
                np.random.shuffle(label_indices)

        while count < self.data_size:
            indices = []
            for i in range(len(self.class_indices)):
                label_indices = self.label_indices[i].copy()
                cur_idx = used_label_indices_count[i]
                remain = self.sample_per_class

                while remain > 0:
                    if self.shuffle and remain < self.sample_per_class:
                        np.random.shuffle(label_indices)

                    end = min(len(label_indices), cur_idx + remain)
                    indices.extend(label_indices[cur_idx:end])
                    consumed = end - cur_idx
                    remain -= consumed
                    if end == len(label_indices) and self.no_repeat:
                        yield indices
                        return

                    cur_idx = (cur_idx + consumed) % len(label_indices)
                used_label_indices_count[i] = cur_idx

            count += len(indices)
            if count > self.data_size:
                if self.drop_last:
                    return
                indices = indices[:self.data_size - count]
            yield indices

    def __len__(self):
        return self.data_size // self.batch_size + int(self.data_size % self.batch_size != 0)


def get_dataset(cfg, split='train', **kwargs):
    """ Helper function."""
    Dataset = globals().get(cfg.dataset, None)
    if Dataset is None:
        raise ValueError(f"{cfg.dataset} is not defined")

    if cfg.kwargs is not None:
        return Dataset(cfg, split=split, **cfg.kwargs, **kwargs)
    return Dataset(cfg, split=split, **kwargs)


def get_dataloader(ds, batch_size, distributed=False, **override_kwargs):
    """ provide default logic for constructing dataloader
        and optionally override arguments
    """
    assert isinstance(ds, data.Dataset)
    assert isinstance(batch_size, int) and batch_size > 0

    loader_kwargs = {}
    loader_kwargs['batch_size'] = batch_size
    loader_kwargs['drop_last'] = True
    loader_kwargs['pin_memory'] = ds.cfg.pin_memory
    loader_kwargs['num_workers'] = ds.cfg.workers
    if loader_kwargs['num_workers'] > 0:
        loader_kwargs['worker_init_fn'] = ds.__class__.worker_init_fn

    if distributed:
        # https://discuss.pytorch.org/t/distributedsampler/90205/2?u=jimmya-1995
        assert ds.split in ['train', 'all'], "dist. sampler is only used in training"
        assert 'sampler' not in override_kwargs
        loader_kwargs['sampler'] = data.distributed.DistributedSampler(ds, shuffle=True)
    else:
        loader_kwargs['shuffle'] = (ds.split == 'train')

    loader_kwargs.update(override_kwargs)

    return data.DataLoader(ds, **loader_kwargs)
