import pickle
from time import time
from io import BytesIO
from pathlib import Path
from functools import partial
from collections import namedtuple

import lmdb
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder


ALLOW_EXTS = ['jpg', 'jpeg', 'png', 'JPEG']


class MultiResolutionDataset(data.Dataset):
    def __init__(self, config, resolution, transform=None):
        path = config.roots[0]
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        raise RuntimeError("Not implement conditional for the dataset.")
        return img, None


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


def load_images_and_concat(path, resolution, sources, channel_info=None, flip=False):
    # assert isinstance(paths, (tuple, list))
    src_path = str(path)
    try:
        imgs = []
        for i, src in enumerate(sources):
            path = src_path.replace(sources[0], src)
            if not Path(path).exists():
                raise RuntimeError(f'Path {path} does not exists. Please check all your sources \
                                   in dataset are match')
            img = Image.open(path).resize((resolution, resolution), Image.ANTIALIAS)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            img = np.array(img)
            if img.ndim == 2:
                img = img[..., None]
            if channel_info:
                assert img.shape[-1] == channel_info[i]
            imgs.append(img)
    except:
        raise RuntimeError(f'fail to load the image: {path}')

    cat_images = np.concatenate(imgs, axis=-1)

    return cat_images


class MultiChannelDataset(data.Dataset):
    """
    This dataset concatenate the source images with other
    information images like skeletons or masks.
    """
    def __init__(self, config, resolution, transform=None, **kwargs):
        from torchvision import get_image_backend
        assert get_image_backend() == 'PIL'
        assert len(config.source) == len(config.channels), \
            f"the numbers of sources and channels don't match."

        self.roots = config.roots
        self.transform = transform
        self.target_transform = None
        self.loader = partial(load_images_and_concat,
                              resolution=resolution,
                              sources=config.source,
                              channel_info=config.channels)
        self.load_in_mem = config.LOAD_IN_MEM
        self.flip = config.xflip

        sources = config.source
        self.img_paths = []
        for root in self.roots:
            root = Path(root)
            for src in sources:
                assert (root / src).is_dir(), f'source directory {src} is not in root path: {root}'

            self.img_paths.extend(list((root / sources[0]).glob('*.jpg')))

        self.length = len(self.img_paths)
        if self.flip:
            self.length *= 2

        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(self.length)):
                path = self.img_paths[index % (self.length // 2)]
                target = None  # unconditional for now
                flip = index >= (self.length // 2)
                concat_img = self.loader(path, flip=flip)
                if self.transform is not None:
                    concat_img = self.transform(concat_img)
                # if self.target_transform is not None:
                #     target = self.target_transform(target)
                self.data.append(concat_img)
                self.labels.append(target)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.load_in_mem:
            concat_img = self.data[index]
            target = self.labels[index]
        else:
            path = self.img_paths[index % (self.length // 2)]
            flip = index >= (self.length // 2)
            concat_img = self.loader(path, flip=flip)
            target = 0

            if self.transform is not None:
                concat_img = self.transform(concat_img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return concat_img, target

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.roots)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class GenericDataset(data.Dataset):
    def __init__(self, config, resolution, transform=None, split='train', **kwargs):
        self.paths = []
        for ext in ALLOW_EXTS:
            self.paths += sorted(list((Path(config.roots[0]) / split).glob(f"*.{ext}")))
        self.config = config
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raise NotImplementedError()


class DeepFashionDataset(data.Dataset):
    def __init__(self, config, resolution, transform=None, split='train'):
        assert len(config.roots) == len(config.source) == 1
        assert split in ['train', 'val', 'test', 'all']
        root = Path(config.roots[0]).expanduser()
        src = config.source[0]
        self.config = config
        self.resolution = resolution
        self.transform = transform
        self.mask_trf = transforms.ToTensor()

        split_file = Path(root) / 'split.pkl'
        assert split_file.exists()
        split_map = pickle.load(open(split_file, 'rb'))
        if split == 'all':
            self.fileID = [ID for IDs in split_map.values() for ID in IDs]
        else:
            self.fileID = split_map[split]
        self.fileID.sort()

        self.face_dir = Path(root) / src / 'face'
        self.mask_dir = Path(root) / src / 'mask'
        self.target_dir = Path(root) / f'r{resolution}' / 'images'
        assert self.face_dir.exists() and self.mask_dir.exists() and self.target_dir.exists()
        assert set(self.fileID) <= set(p.stem for p in self.face_dir.glob('*.png'))
        assert set(self.fileID) <= set(p.stem for p in self.mask_dir.glob('*.png'))
        assert set(self.fileID) <= set(p.stem for p in self.target_dir.glob('*.png'))

    def __len__(self):
        return len(self.fileID)

    def __getitem__(self, idx):
        filename = f'{self.fileID[idx]}.png'
        res = self.resolution
        face = Image.open(self.face_dir / filename).resize((res, res), Image.ANTIALIAS)
        mask = Image.open(self.mask_dir / filename).resize((res, res), Image.NEAREST)
        target = Image.open(self.target_dir / filename)
        assert face.mode == 'RGB' and mask.mode == 'L' and target.mode == 'RGB'

        if self.transform is not None:
            face = self.transform(face)
            target = self.transform(target)
        mask = self.mask_trf(mask)

        return target, face, mask


class ResamplingDataset(data.Dataset):
    def __init__(self, cfg, resolution):
        assert (Path(cfg.roots[0]).parent / 'landmarks_statistics.pkl').exists()
        assert (Path(cfg.roots[0]).parent / 'stylegan2-ada-outputs').exists()
        trf = [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean[:3], cfg.std[:3], inplace=True),
        ]
        self.transform = transforms.Compose(trf)
        self.tgt_size = resolution
        statistics = pickle.load(open(Path(cfg.roots[0]).parent / 'landmarks_statistics.pkl', 'rb'))
        self.paths = sorted(list((Path(cfg.roots[0]).parent / 'stylegan2-ada-outputs').glob('*.png')))

        self.ori_size = statistics['resolution']
        self.V = statistics['V']
        self.mu = statistics['mu']
        self.sigma = statistics['sigma']
        self.ndim = statistics['V'].shape[1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        canvas = np.ones((self.ori_size, self.ori_size, 3), dtype=np.uint8) * 127  # gray

        # resampling
        z = np.random.randn(self.ndim, )
        resample_latent = z @ self.V.T * self.sigma + self.mu
        resample_latent = np.where(resample_latent > 0, resample_latent, 0).astype(int)
        x1, y1, x2, y2 = resample_latent[4:8]

        face_img = Image.open(self.paths[idx])
        face_np = np.asarray(face_img.resize((x2 - x1, y2 - y1), Image.ANTIALIAS))
        canvas[y1:y2, x1:x2] = face_np
        rect = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], dtype=np.float32)
        quad = resample_latent[8:].reshape(4, 2).astype(np.float32)
        M = cv2.getPerspectiveTransform(rect, quad)
        canvas = cv2.warpPerspective(canvas, M, (self.ori_size, self.ori_size), borderMode=cv2.BORDER_REPLICATE)
        mask = (canvas != 127).all(axis=-1).astype(np.float32)

        face_img = face_img.resize((self.tgt_size, self.tgt_size), Image.ANTIALIAS)
        masked_body = Image.fromarray(canvas).resize((self.tgt_size, self.tgt_size), Image.ANTIALIAS)
        mask = cv2.resize(mask[..., None], (self.tgt_size, self.tgt_size), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            face_img = self.transform(face_img)
            masked_body = self.transform(masked_body)
            mask = transforms.ToTensor()(mask.copy())

        return masked_body, face_img, mask


class ResamplingDatasetV2(data.Dataset):
    Gaussian = namedtuple('Gaussian', 'X_mean X_std cx_mean cx_std cy_mean cy_std')
    Gaussian.__qualname__ = "ResamplingDatasetV2.Gaussian"

    def __init__(self, config, resolution, split='train', **kwargs):
        """  Resampling position & angles for fake faces
        """
        assert len(config.roots) == len(config.source) == 1
        assert split in ['train', 'val']
        self.resolution = resolution
        trf = [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True),
        ]
        self.transform = transforms.Compose(trf)
        self.mask_trf = transforms.ToTensor()

        # statistics
        self.big = ResamplingDatasetV2.Gaussian(78.08, 11.18, 517.075, 26.655, 131.60, 24.41)
        self.small = ResamplingDatasetV2.Gaussian(44.56, 3.32, 517.075, 26.655, 100.36, 17.22)
        self.rng = np.random.default_rng()

        root = Path(config.roots[0]).expanduser() / config.source[0]
        self.fake_dir = root / 'fake_face'
        info_file = root / 'fake_face_phi.pkl'
        assert self.fake_dir.exists() and info_file.exists()

        info = pickle.load(open(info_file, 'rb'))  # file_stem, phi
        total = len(info)
        self.info = info[:total // 2] if split == 'train' else info[total // 2:]

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        res = self.resolution
        fake_face = Image.open(self.fake_dir / f"{self.info[idx][0]}.png")
        phi = self.info[idx][1]
        dist = self.big if np.random.random() < 0.7 else self.small
        rho = self.rng.normal(loc=dist.X_mean, scale=dist.X_std, size=())
        cx = self.rng.normal(loc=dist.cx_mean, scale=dist.cx_std, size=())
        cy = self.rng.normal(loc=dist.cy_mean, scale=dist.cy_std, size=())
        c = np.hstack([cx, cy])
        x = np.hstack([rho * np.cos(phi), rho * np.sin(phi)])
        y = np.flipud(x) * [-1, 1]
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)

        face_np = np.asarray(fake_face.resize((1024, 1024), Image.ANTIALIAS))
        src = np.array([[0, 0], [0, 1024], [1024, 1024], [1024, 0]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, quad)
        masked_body = cv2.warpPerspective(face_np, M, (1024, 1024), borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        mask = Image.fromarray(np.any(masked_body != 127, axis=-1).astype(np.uint8) * 255, mode='L').resize((res, res), Image.NEAREST)
        masked_body = Image.fromarray(masked_body).resize((res, res), Image.ANTIALIAS)

        if self.transform is not None:
            fake_face = self.transform(fake_face)
            masked_body = self.transform(masked_body)
        mask = self.mask_trf(mask)

        return masked_body, fake_face, mask


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    return data.SequentialSampler(dataset)


def get_dataset(config, resolution, split='train'):
    trf = [
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std, inplace=True),
    ]
    transform = transforms.Compose(trf)
    Dataset = globals().get(config.dataset)
    dataset = Dataset(config, resolution, transform=transform, split=split)
    return dataset


def get_dataloader(config, batch_size, n_workers=None, split='train', distributed=False):
    dataset = get_dataset(config.DATASET, config.resolution, split=split)
    if n_workers is None:
        n_workers = config.DATASET.workers

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=config.DATASET.pin_memory,
        sampler=data_sampler(dataset, shuffle=(split == 'train'), distributed=distributed),
        drop_last=True,
    )
    return loader


def get_dataloader_for_each_class(config, batch_size, distributed=False):
    dataset = get_dataset(config.DATASET, config.resolution)
    data_root = Path(config.DATASET.roots[0])
    dataloaders = []
    indices = list(range(len(dataset)))
    last_idx, cur_idx = 0, 0
    for i, (label_class, idx) in enumerate(dataset.class_to_idx.items(), 1):
        for ext in ALLOW_EXTS:
            cur_idx += len(list((data_root / label_class).glob(f'*.{ext}')))

        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.DATASET.workers,
            sampler=data.SubsetRandomSampler(indices[last_idx:cur_idx]),
            drop_last=True,
        )
        dataloaders.append(loader)
        last_idx = cur_idx
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    return dataloaders, idx_to_class
