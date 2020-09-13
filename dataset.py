import sys
from io import BytesIO
from pathlib import Path
from functools import partial
from tqdm import tqdm
import lmdb
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder


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
    path = config.ROOTS[0]
    def image_loader(path):
        try:
            img = Image.open(path)
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
    """ This dataset concatenate the source images with other 
        information images like skeletons or masks.
    
    """
    def __init__(self, config, resolution, transform=None, **kwargs):
        from torchvision import get_image_backend
        assert get_image_backend() == 'PIL'
        assert len(config.SOURCE) == len(config.CHANNELS), \
                f"the numbers of sources and channels don't match."
        
        self.roots = config.ROOTS
        self.transform = transform
        self.target_transform = None ##
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
        
            self.img_paths.extend(list((root / sources[0]).glob('**/*.jpg')))
        
        
        self.length = len(self.img_paths)
        if self.flip:
            self.length *= 2
        
        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(self.length)):
                path = self.img_paths[index % (self.length // 2)]
                target = None # unconditional for now
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
            target = None
            
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
    
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def get_dataset(config, resolution):
    # config.DATASET
    trf = [
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD, inplace=True),
    ]
    transform = transforms.Compose(trf)
    Dataset = globals().get(config.DATASET)
    dataset = Dataset(config, resolution, transform=transform)
    return dataset   
    
def get_dataloader(config, batch_size, distributed=False):
    dataset = get_dataset(config.DATASET, config.RESOLUTION)
    
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.DATASET.WORKERS,
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
        cur_idx += len(list((data_root / label_class).glob('*.jpg')))
        
        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.DATASET.WORKERS,
            sampler=data.SubsetRandomSampler(indices[last_idx:cur_idx]),
            drop_last=True,
        )
        dataloaders.append(loader)
        laslt_idx = cur_idx
        idx_to_class = {v: k for k,v in dataset.class_to_idx.items()}
        
    return dataloaders, idx_to_class


if __name__ == "__main__":
    # test code
    pass