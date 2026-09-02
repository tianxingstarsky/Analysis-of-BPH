# -*- coding: utf-8 -*-
"""Case-wise validation evaluation for prostate U-Net checkpoints.
usage: python eval_val.py [--se] [--crop 176] ckpt1 [ckpt2 ...]"""
import collections
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_img
from unet import UNet

args = [a for a in sys.argv[1:]]
se = '--se' in args
classes = 2
if '--classes' in args:
    i = args.index('--classes')
    classes = int(args[i + 1])
    del args[i:i + 2]
args = [a for a in args if not a.startswith('--')]
crop = 0
if '--crop' in args:
    i = args.index('--crop')
    crop = int(args[i + 1])
    del args[i:i + 2]
ckpts = [a for a in args if a.endswith('.pth')]

device = torch.device('cuda')
net = UNet(n_channels=1, n_classes=classes, bilinear=False, se=se).to(device)


def dice(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    return 2 * inter / (inter + np.logical_or(a > 0, b > 0).sum())


def center_crop(img, m):
    ys, xs = np.nonzero(m)
    cy, cx = (ys.mean(), xs.mean()) if len(ys) else (img.shape[0] / 2, img.shape[1] / 2)
    r = crop / 2
    y0 = int(np.clip(cy - r, 0, img.shape[0] - crop))
    x0 = int(np.clip(cx - r, 0, img.shape[1] - crop))
    return img[y0:y0 + crop, x0:x0 + crop], m[y0:y0 + crop, x0:x0 + crop]


val_imgs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'prostate_slices', 'val', 'imgs', '*.png')))
data = {}
for f in val_imgs:
    name = os.path.basename(f)[:-4]
    im = np.asarray(Image.open(f))
    m = np.asarray(Image.open(f.replace('imgs', 'masks').replace('.png', '_mask.png'))) > 0
    if crop:
        im, m = center_crop(im, m)
    data[f] = (name, im, m)

for ck in ckpts:
    sd = torch.load(ck, map_location=device)
    sd.pop('mask_values', None)
    net.load_state_dict(sd)
    net.eval()
    ds = []
    for f in val_imgs:
        name, im, m = data[f]
        pred = predict_img(net, Image.fromarray(im), device, 1.0)
        ds.append(dice(m, pred))
    per = collections.defaultdict(list)
    for f, d in zip(val_imgs, ds):
        per[os.path.basename(f).split('_')[1]].append(d)
    percase = ' '.join(f'p{k}:{np.mean(v):.3f}' for k, v in sorted(per.items()))
    print(f'{os.path.basename(ck)}: mean {np.mean(ds):.4f} (min {np.min(ds):.3f} max {np.max(ds):.3f}) | {percase}')
