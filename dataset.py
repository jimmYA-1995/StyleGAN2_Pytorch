from io import BytesIO
from pathlib import Path
from functools import partial
import pickle
from tqdm import tqdm
import lmdb
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder


ALLOW_EXTS = ['jpg', 'jpeg', 'png', 'JPEG']


class MultiResolutionDataset(data.Dataset):
    def __init__(self, config, resolution, transform=None):
        path = config.ROOTS[0]
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

    return ImageFolder(config.ROOTS[0], transform=transform, loader=image_loader, is_valid_file=check_valid)


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
        assert len(config.SOURCE) == len(config.CHANNELS), \
            f"the numbers of sources and channels don't match."

        self.roots = config.ROOTS
        self.transform = transform
        self.target_transform = None
        self.loader = partial(load_images_and_concat,
                              resolution=resolution,
                              sources=config.SOURCE,
                              channel_info=config.CHANNELS)
        self.load_in_mem = config.LOAD_IN_MEM
        self.flip = config.FLIP

        sources = config.SOURCE
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
            self.paths += list((Path(config.ROOTS[0]) / split).glob(f"*.{ext}"))
        self.config = config
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raise NotImplementedError()


class DeepFashionDataset(GenericDataset):
    def __init__(self, config, resolution, transform=None, split='train', **kwargs):
        super(DeepFashionDataset, self).__init__(
            config, resolution, transform=transform, split=split, **kwargs)

    def __getitem__(self, idx):
        cfg = self.config
        load_size = self.resolution + 30
        h1 = w1 = (load_size - self.resolution) // 2
        h2 = w2 = (load_size + self.resolution) // 2

        img = Image.open(self.paths[idx])
        assert (img.mode == 'RGBA' and cfg.CHANNELS[0] == 4) ^ (img.mode == 'RGB' and cfg.CHANNELS[0] == 3), \
            "image channel is not consistent, please check your config"

        img_np = np.asarray(img.convert('RGB').resize((load_size * 2, load_size), Image.ANTIALIAS))
        mask = None
        if img.mode == 'RGBA':
            mask = img.split()[-1].resize((load_size * 2, load_size), Image.NEAREST)
            mask_np = np.asarray(mask)[..., None]
            img_np = np.concatenate([img_np, mask_np], axis=-1)

        imgA = img_np[h1:h2, w1:w2, :]
        imgB = img_np[h1:h2, load_size + w1:load_size + w2, :]

        if self.transform is not None:
            imgA = self.transform(imgA.copy())
            imgB = self.transform(imgB.copy())

        if mask is not None:
            imgA = imgA[:3, :, :]
            imgB, mask = torch.split(imgB, 3, dim=0)
            return imgB, imgA, mask
        return imgB, imgA


class ResamplingDataset(data.Dataset):
    def __init__(self, cfg, resolution):
        trf = [
            transforms.ToTensor(),
            transforms.Normalize(cfg.MEAN[:3], cfg.STD[:3], inplace=True),
        ]
        self.transform = transforms.Compose(trf)
        self.tgt_size = resolution
        statistics = pickle.load(
            open(Path('~/data/deepfashion256_pix2pix/landmarks_statistics.pkl').expanduser(), 'rb'))
        self.paths = list(Path('~/data/stylgan2-ada-outputs/').expanduser().glob('*.png'))
        self.ori_size = statistics['resolution']
        self.V = statistics['V']
        self.mu = statistics['mu']
        self.sigma = statistics['sigma']
        self.ndim = statistics['V'].shape[1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        canvas = np.ones((self.ori_size, self.ori_size, 3), dtype=np.uint8) * 127  # gray
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


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    return data.SequentialSampler(dataset)


def get_dataset(config, resolution, split='train'):
    trf = [
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD, inplace=True),
    ]
    transform = transforms.Compose(trf)
    Dataset = globals().get(config.DATASET)
    dataset = Dataset(config, resolution, transform=transform, split=split)
    return dataset


def get_dataloader(config, batch_size, n_workers=None, split='train', distributed=False):
    dataset = get_dataset(config.DATASET, config.RESOLUTION, split=split)
    if n_workers is None:
        n_workers = config.DATASET.WORKERS

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=data_sampler(dataset, shuffle=True, distributed=distributed),
        drop_last=True,
    )
    return loader


def get_dataloader_for_each_class(config, batch_size, distributed=False):
    dataset = get_dataset(config.DATASET, config.RESOLUTION)
    data_root = Path(config.DATASET.ROOTS[0])
    dataloaders = []
    indices = list(range(len(dataset)))
    last_idx, cur_idx = 0, 0
    for i, (label_class, idx) in enumerate(dataset.class_to_idx.items(), 1):
        for ext in ALLOW_EXTS:
            cur_idx += len(list((data_root / label_class).glob(f'*.{ext}')))

        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.DATASET.WORKERS,
            sampler=data.SubsetRandomSampler(indices[last_idx:cur_idx]),
            drop_last=True,
        )
        dataloaders.append(loader)
        last_idx = cur_idx
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    return dataloaders, idx_to_class
