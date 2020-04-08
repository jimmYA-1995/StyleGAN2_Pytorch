from io import BytesIO

import lmdb
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
# class ImageFolderDataset(ImageFolder):
#     def __init__(self, path, transform=None, resolution=None):
#         def image_loader(path):
#             try:
#                 img = Image.open(path)
#                 if not resolution:
#                     img = img.resize((resolution, resolution), Image.ANTIALIAS)
#                 return img
#             except:
#                 print(f'fail to load the image: {path}')
#                 return None
#         check_valid = lambda img: True if img is not None else False
#         super().__init__(path, transform=transform, loader=image_loader, is_valid_file=check_valid)
#     def __getitem__(self, index):
        