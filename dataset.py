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


class ResamplingDatasetV2(data.Dataset):
    Gaussian = namedtuple('Gaussian', 'X_mean X_std cx_mean cx_std cy_mean cy_std')
    Gaussian.__qualname__ = "ResamplingDatasetV2.Gaussian"

    def __init__(self, cfg, split='train'):
        """  Resampling faces position & 2d orientation """
        assert len(cfg.roots) == len(cfg.source) == 1
        assert split in ['train', 'val', 'test', 'all']
        self.cfg = cfg
        self.root = Path(cfg.roots[0]).expanduser()
        self.face_dir = None
        self.fileIDs = None
        self.info = None  # face information
        self.idx = None
        self.resolution = cfg.resolution
        self.split = split
        self.xflip = cfg.xflip
        self._img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std, inplace=True),
        ])
        self._mask_transform = transforms.ToTensor()

        # statistics
        self.rng = None
        self.big = ResamplingDatasetV2.Gaussian(78.08, 11.18, 517.075, 26.655, 131.60, 24.41)
        self.small = ResamplingDatasetV2.Gaussian(44.56, 3.32, 517.075, 26.655, 100.36, 17.22)

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

    def resample_face_position(self, face, face_angle):
        if self.rng is None:
            if torch.utils.data.get_worker_info():
                raise RuntimeError("using worker_init_fn to set RNG when num_wokers > 0")
            # main process (when num_workers=0)
            self.rng = np.random.default_rng()

        # get quad coord. (in 1024x1024 context)
        dist = self.big if np.random.random() < 0.7 else self.small
        rho = self.rng.normal(loc=dist.X_mean, scale=dist.X_std, size=())
        cx = self.rng.normal(loc=dist.cx_mean, scale=dist.cx_std, size=())
        cy = self.rng.normal(loc=dist.cy_mean, scale=dist.cy_std, size=())
        c = np.hstack([cx, cy])
        x = np.hstack([rho * np.cos(face_angle), rho * np.sin(face_angle)])
        y = np.flipud(x) * [-1, 1]
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)

        # warp
        face_np = np.asarray(face)
        res = face_np.shape[0]
        src = np.array([[0, 0], [0, res], [res, res], [res, 0]], dtype=np.float32)

        longest_side = (quad.max(axis=0) - quad.min(axis=0)).max()
        ratio = longest_side / res
        q = (quad - quad.min(axis=0)) / ratio

        M = cv2.getPerspectiveTransform(src, q)
        shrink = int(res * ratio * res / 1024)
        offset_x, offset_y = (quad.min(axis=0) / 4).astype(int)
        mb = cv2.warpPerspective(face_np, M, (res, res), borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))

        fm = np.any(mb != 127, axis=-1).astype(np.uint8) * 255
        fm = Image.fromarray(fm, mode='L').resize((shrink, shrink), Image.LANCZOS)
        fake_mask = Image.new('1', (res, res))
        fake_mask.paste(fm, (offset_x, offset_y))

        mb = Image.fromarray(mb).resize((shrink, shrink), Image.LANCZOS)
        masked_body = Image.new('RGB', (res, res), color=(127, 127, 127))
        masked_body.paste(mb, (offset_x, offset_y))

        masked_body = self.img_transform(masked_body)
        fake_mask = self.mask_transform(fake_mask)

        return masked_body, fake_mask

    @classmethod
    def worker_init_fn(cls, worker_id):
        """ For reproducibility & randomness in multi-worker mode """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        if hasattr(dataset, 'rng'):
            dataset.rng = np.random.default_rng(worker_seed)


class DeepFashionDataset(ResamplingDatasetV2):
    def __init__(self, cfg, split='train', resampling=None):
        super(DeepFashionDataset, self).__init__(cfg, split=split)
        self.resampling = resampling

        split_map = pickle.load(open(self.root / 'split.pkl', 'rb'))
        if split == 'all':
            self.fileIDs = [ID for IDs in split_map.values() for ID in IDs]
        else:
            self.fileIDs = split_map[split]
        self.fileIDs.sort()

        src = cfg.source[0]
        self.face_dir = self.root / src / 'face'
        self.mask_dir = self.root / src / 'mask'
        self.target_dir = self.root / f'r{self.resolution}' / 'images'
        assert self.face_dir.exists() and self.mask_dir.exists() and self.target_dir.exists()
        assert set(self.fileIDs) <= set(p.stem for p in self.face_dir.glob('*.png'))
        assert set(self.fileIDs) <= set(p.stem for p in self.mask_dir.glob('*.png'))
        assert set(self.fileIDs) <= set(p.stem for p in self.target_dir.glob('*.png'))

        if resampling:
            assert resampling in ['gt', 'pred']
            # file_stem -> {gt, pred}
            self.info = pickle.load(open(self.root / src / 'real_face_phi.pkl', 'rb'))

    def __getitem__(self, idx):
        self.idx = idx
        fileID = self.fileIDs[idx % len(self.fileIDs)] if self.xflip else self.fileIDs[idx]

        _face = Image.open(self.face_dir / f'{fileID}.png')
        target = Image.open(self.target_dir / f'{fileID}.png')
        assert _face.mode == 'RGB' and target.mode == 'RGB'
        face = self.img_transform(_face)
        target = self.img_transform(target)

        if self.resampling:
            masked_body, fake_mask = self.resample_face_position(_face, self.info[fileID][self.resampling])
            return target, face, fake_mask, masked_body

        real_mask = Image.open(self.mask_dir / f'{fileID}.png')
        assert real_mask.mode == 'L'
        real_mask = self.mask_transform(real_mask)

        return target, face, real_mask


# class ResamplingDataset(data.Dataset):
#     def __init__(self, cfg, resolution):
#         assert (Path(cfg.roots[0]).parent / 'landmarks_statistics.pkl').exists()
#         assert (Path(cfg.roots[0]).parent / 'stylegan2-ada-outputs').exists()
#         trf = [
#             transforms.ToTensor(),
#             transforms.Normalize(cfg.mean[:3], cfg.std[:3], inplace=True),
#         ]
#         self.transform = transforms.Compose(trf)
#         self.tgt_size = resolution
#         statistics = pickle.load(open(Path(cfg.roots[0]).parent / 'landmarks_statistics.pkl', 'rb'))
#         self.paths = sorted(list((Path(cfg.roots[0]).parent / 'stylegan2-ada-outputs').glob('*.png')))

#         self.ori_size = statistics['resolution']
#         self.V = statistics['V']
#         self.mu = statistics['mu']
#         self.sigma = statistics['sigma']
#         self.ndim = statistics['V'].shape[1]

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         canvas = np.ones((self.ori_size, self.ori_size, 3), dtype=np.uint8) * 127  # gray

#         # resampling
#         z = np.random.randn(self.ndim, )
#         resample_latent = z @ self.V.T * self.sigma + self.mu
#         resample_latent = np.where(resample_latent > 0, resample_latent, 0).astype(int)
#         x1, y1, x2, y2 = resample_latent[4:8]

#         face_img = Image.open(self.paths[idx])
#         face_np = np.asarray(face_img.resize((x2 - x1, y2 - y1), Image.ANTIALIAS))
#         canvas[y1:y2, x1:x2] = face_np
#         rect = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], dtype=np.float32)
#         quad = resample_latent[8:].reshape(4, 2).astype(np.float32)
#         M = cv2.getPerspectiveTransform(rect, quad)
#         canvas = cv2.warpPerspective(canvas, M, (self.ori_size, self.ori_size), borderMode=cv2.BORDER_REPLICATE)
#         mask = (canvas != 127).all(axis=-1).astype(np.float32)

#         face_img = face_img.resize((self.tgt_size, self.tgt_size), Image.ANTIALIAS)
#         masked_body = Image.fromarray(canvas).resize((self.tgt_size, self.tgt_size), Image.ANTIALIAS)
#         mask = cv2.resize(mask[..., None], (self.tgt_size, self.tgt_size), interpolation=cv2.INTER_NEAREST)
#         if self.transform:
#             face_img = self.transform(face_img)
#             masked_body = self.transform(masked_body)
#             mask = transforms.ToTensor()(mask.copy())

#         return masked_body, face_img, mask


class FakeDeepFashionFace(ResamplingDatasetV2):
    def __init__(self, cfg, split='train'):
        """  Resampling position & angles for fake faces
        """
        assert split in ['train', 'val']
        super(FakeDeepFashionFace, self).__init__(cfg, split=split)

        self.face_dir = self.root / cfg.source[0] / 'fake_face'
        assert self.face_dir.exists()

        self.info = pickle.load(open(self.root / cfg.source[0] / 'fake_face_phi.pkl', 'rb'))  # file_stem: phi
        fileIDs = sorted(k for k in self.info.keys())
        self.fileIDs = fileIDs[:len(fileIDs) // 2] if split == 'train' else fileIDs[len(fileIDs) // 2:]
        assert set(self.fileIDs) <= set(p.stem for p in self.face_dir.glob('*.png'))

    def __getitem__(self, idx):
        self.idx = idx
        fileID = self.fileIDs[idx % len(self.fileIDs)] if self.xflip else self.fileIDs[idx]
        face = Image.open(self.face_dir / f"{fileID}.png")
        masked_body, mask = self.resample_face_position(face, self.info[fileID])

        face = self.img_transform(face)

        return masked_body, face, mask


class ConditionalBatchSampler(data.sampler.BatchSampler):
    def __init__(self,
                 dataset,
                 class_indices: List[int],
                 sample_per_class: int = 1,
                 shuffle: bool = False,
                 no_repeat: bool = False,
                 num_gpus: int = 1,
                 rank: int = 0):
        assert hasattr(dataset, 'num_classes')
        assert hasattr(dataset, 'labels') and isinstance(dataset.labels, (Sequence, np.ndarray))
        assert all(0 <= idx < dataset.num_classes for idx in class_indices)
        if no_repeat:
            assert len(class_indices) == 1, "no_repeat is used for iterate over a specific class once."

        if rank >= num_gpus or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_gpus - 1))

        self.num_gpus = num_gpus
        self.rank = rank
        self.data_size = len(dataset)
        self.num_classes = dataset.num_classes
        self.class_indices = class_indices
        self.sample_per_class = sample_per_class
        self.batch_size = len(class_indices) * sample_per_class
        self.no_repeat = no_repeat
        self.label_indices = []
        for c in self.class_indices:
            label_indices = np.where(np.array(dataset.labels) == c)[0]
            if len(label_indices) == 0:
                raise RuntimeError(f"no data for class{c}")

            if len(label_indices) < sample_per_class:
                warn(f"total samples of class No.{class_indices[i]} is less than required.")

            if len(label_indices) % num_gpus != 0:
                # keep each replica have same elements
                complement = num_gpus - len(label_indices) % num_gpus
                label_indices = np.concatenate([label_indices, label_indicies[0].repeat(complement)])

            self.label_indices.append(label_indices[rank::num_gpus])

    def __iter__(self):
        count = 0
        used_label_indices_count = [0] * len(self.class_indices)
        while count < self.data_size:
            indices = []
            for i in range(len(self.class_indices)):
                label_indices = self.label_indices[i].copy()
                cur_idx = used_label_indices_count[i]
                remain = self.sample_per_class

                while remain > 0:
                    if remain < self.sample_per_class:
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

            yield indices
            count += len(indices)

    def __len__(self):
        return self.data_size // self.batch_size


def get_dataset(cfg, split='train'):
    """ Helper function."""
    Dataset = globals().get(cfg.dataset, None)
    if Dataset is None:
        raise ValueError(f"{cfg.dataset} is not defined")

    if cfg.kwargs is not None:
        return Dataset(cfg, split=split, **cfg.kwargs)
    return Dataset(cfg, split=split)


def get_dataloader(ds, batch_size, distributed=False, **override_kwargs):
    """ provide default logic for constructing dataloader
        and optionally override arguments
    """
    assert isinstance(ds, data.Dataset)
    assert isinstance(batch_size, int) and batch_size > 0
    # dataset = get_dataset(cfg.DATASET, split=split)
    loader_kwargs = {}
    loader_kwargs['batch_size'] = batch_size
    loader_kwargs['drop_last'] = True
    loader_kwargs['pin_memory'] = ds.cfg.pin_memory
    loader_kwargs['num_workers'] = ds.cfg.workers
    if loader_kwargs['num_workers'] > 0:
        loader_kwargs['worker_init_fn'] = ds.__class__.worker_init_fn

    if distributed:
        # https://discuss.pytorch.org/t/distributedsampler/90205/2?u=jimmya-1995
        assert ds.split == 'train', "dist. sampler is only used in training"
        assert 'sampler' not in override_kwargs
        loader_kwargs['sampler'] = data.distributed.DistributedSampler(ds, shuffle=True)
    else:
        loader_kwargs['shuffle'] = (ds.split == 'train')

    loader_kwargs.update(override_kwargs)
    print(loader_kwargs)

    return data.DataLoader(ds, **loader_kwargs)
