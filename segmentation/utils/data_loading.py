import logging
import random
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def random_augment(img, mask):
    """img [1,H,W] float in [0,1], mask [H,W] int64. Classic U-Net recipe:
    affine (rotation+scale+shift), elastic deformation, gamma/brightness/noise."""
    h, w = mask.shape

    if random.random() < 0.7:   # affine: rotate +-20deg, scale 0.9-1.1, shift +-16px
        ang = np.deg2rad(random.uniform(-20, 20))
        sc = random.uniform(0.9, 1.1)
        c = np.array([[np.cos(ang) * sc, np.sin(ang) * sc],
                      [-np.sin(ang) * sc, np.cos(ang) * sc]])
        offset = (np.array([h, w]) / 2) - c @ (np.array([h, w]) / 2) + \
            np.array([random.uniform(-16, 16), random.uniform(-16, 16)])
        img = ndi.affine_transform(img[0], c, offset=offset, order=1, mode='nearest')[None]
        mask = ndi.affine_transform(mask, c, offset=offset, order=0, mode='nearest')

    if random.random() < 0.4:   # elastic deformation (original U-Net paper's key trick)
        dy, dx = (ndi.gaussian_filter(np.random.uniform(-1, 1, (h, w)), 6) * 5 for _ in range(2))
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        idx = [yy + dy, xx + dx]
        img = ndi.map_coordinates(img[0], idx, order=1, mode='nearest')[None]
        mask = ndi.map_coordinates(mask, idx, order=0, mode='nearest')

    if random.random() < 0.6:   # gamma
        img = np.clip(img, 1e-4, 1) ** random.uniform(0.8, 1.25)
    if random.random() < 0.5:   # brightness / contrast
        img = img * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)
    if random.random() < 0.4:   # gaussian noise
        img = img + np.random.normal(0, random.uniform(0.005, 0.02), img.shape)

    return np.clip(img, 0, 1), mask


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '',
                 augment: bool = False, target_crop: int = 0):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment
        self.target_crop = target_crop

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.target_crop:
            # target-centered crop around the gland centroid (mask follows identically);
            # training adds random +-16px jitter, validation crops deterministically
            m = np.asarray(mask) > 0
            ys, xs = np.nonzero(m)
            cy, cx = (ys.mean(), xs.mean()) if len(ys) else (img.size[1] / 2, img.size[0] / 2)
            r = self.target_crop / 2
            j = 16 if self.augment else 0
            y0 = int(np.clip(cy - r + random.uniform(-j, j), 0, img.size[1] - self.target_crop))
            x0 = int(np.clip(cx - r + random.uniform(-j, j), 0, img.size[0] - self.target_crop))
            box = (x0, y0, x0 + self.target_crop, y0 + self.target_crop)
            img, mask = img.crop(box), mask.crop(box)

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, augment=False, target_crop=0):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', augment=augment,
                         target_crop=target_crop)
