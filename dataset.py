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
            return img
        except:
            print(f'fail to load the image: {path}')
            return None
        
    check_valid = lambda img: True if img is not None else False
    return ImageFolder(path, transform=transform, loader=image_loader, is_valid_file=check_valid)


def load_images_and_concat(path, resolution, sources): # is functools.partial needs partial arg a keyword arg?
    from torchvision import get_image_backend
    assert get_image_backend() == 'PIL'
    # assert isinstance(paths, (tuple, list))
    src_path = str(path)
    try:
        imgs = []
        for src in sources:
            path = src_path.replace(sources[-1], src)
            img = Image.open(path).resize((resolution, resolution), Image.ANTIALIAS)
            img = np.array(img)
            if img.ndim == 2:
                img = img[..., None]
                # some grayscale image(1 channel)
                if src == 'images':
                    img = np.repeat(img, 3, axis=-1) 
            imgs.append(img)
    except:
        raise runtimeError(f'fail to load the image: {path}')
    
    cat_images = np.concatenate(imgs, axis=-1)
    return cat_images
    

class MultiChannelDataset(Dataset):
    def __init__(self, root, resolution, sources=['images'], suffix=None, transform=None, target_transform=None,
                 loader=load_images_and_concat, load_in_mem=False, **kwargs):
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = partial(loader, resolution=resolution, sources=sources)
        self.load_in_mem = load_in_mem
        
        root = Path(root)
        for src in sources:
            assert (root / src).is_dir(), f'source directory {src} is not in root path'
        
        def path_exists(path):
            return Path(str(path).replace('images', 'skeleton')).is_file()
        
        if suffix:
            self.img_paths = list((root / sources[-1] / suffix).glob('*.jpg'))
        else:
            self.img_paths = list((root / sources[-1]).glob('*.jpg'))
            
#         self.img_paths = []
#         for img_path in tqdm(img_paths):
#             path_pair =[]
#             for src in sources[:-1]:
#                 p = str(img_path).replace(sources[-1], src)
#                 if Path(p).is_file():
#                     path_pair.append(p)
#                 else:
#                     break
#             else:
#                 path_pair.append(0) # label
#                 assert len(path_pair) == len(sources)+1, f"wrong in filter existing paths {path_pair}"
#                 self.img_paths.append(path_pair)
        
        self.length = len(self.img_paths)
        
        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.img_paths))):
                paths = self.img_paths[index][:-1]
                target = self.img_paths[index][-1]
                data = self.loader(paths)
                if self.transform is not None:
                    data = self.transform(data)
                # if self.target_transform is not None:
                #     target = self.target_transform(target)
                self.data.append(data)
                # self.labels.append(target)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.load_in_mem:
            img = self.data[index]
            target = 0 # self.labels[index]
        else:
            paths = self.img_paths[index]
            imgs = self.loader(paths)
            
            if self.transform is not None:
                try:
                    img = self.transform(imgs)
                except RuntimeError as e:
                    print(" *** exception ***", imgs.shape, paths)
                    raise RuntimeError(e)
                    
            # target = self.img_paths[index][-1]

            # if self.target_transform is not None:
            #     target = self.target_transform(target)
            
        return img, 0

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
