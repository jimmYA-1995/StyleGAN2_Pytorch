import sys
from io import BytesIO
from pathlib import Path
from functools import partial
from tqdm import tqdm
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
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

        return img


def ImageFolderDataset(path, transform=None, resolution=None):
    def image_loader(path):
        try:
            img = Image.open(path)
            if resolution:
                img = img.resize((resolution, resolution), Image.ANTIALIAS)
            if img.mode == 'L':
                img = img.convert('RGB')
            return img
        except:
            print(f'fail to load the image: {path}')
            return None
        
    check_valid = lambda img: True if img is not None else False
    return ImageFolder(path, transform=transform, loader=image_loader, is_valid_file=check_valid)


def load_images_and_concat(path, resolution, sources, channel_info=None, flip=False):
    from torchvision import get_image_backend
    assert get_image_backend() == 'PIL'
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


class MultiChannelDataset(Dataset):
    """ This dataset concatenate the source images with other 
        information images like skeletons or masks.
    
    """
    def __init__(self, config, transform=None, **kwargs):
        assert len(config.DATASET.SOURCE) == len(config.DATASET.CHANNELS), \
                f"the length of sources and channels don't match."
        
        self.roots = config.DATASET.ROOTS
        self.transform = transform
        self.target_transform = None ##
        self.loader = partial(load_images_and_concat,
                              resolution=config.RESOLUTION,
                              sources=config.DATASET.SOURCE,
                              channel_info=config.DATASET.CHANNELS)
        self.load_in_mem = config.DATASET.LOAD_IN_MEM
        self.flip = config.DATASET.FLIP
      
        sources = config.DATASET.SOURCE
        self.img_paths = []
        for root in self.roots:
            root = Path(root)
            for src in sources:
                assert (root / src).is_dir(), f'source directory {src} is not in root path: {root}'
        
            self.img_paths.extend(list((root / sources[0]).glob('**/*.jpg')))
        
        
        self.length = len(self.img_paths)
        if self.flip:
            self.length *= 2
        
        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(self.length)):
                path = self.img_paths[index % (self.length // 2)]
                target = 0 # unconditional for now
                flip = index >= (self.length // 2)
                concat_img = self.loader(path, flip=flip)
                if self.transform is not None:
                    concat_img = self.transform(concat_img)
                #if self.target_transform is not None:
                #    target = self.target_transform(target)
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
                
            #if self.target_transform is not None:
            #    target = self.target_transform(target) 
            
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
