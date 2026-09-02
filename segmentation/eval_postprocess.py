# -*- coding: utf-8 -*-
"""Traditional post-processing calibration on the best U-Net (crop-trained):
predict on GT-centroid 176 crop, paste back to full frame, then compare
raw vs 2D morphology (largest CC + closing + hole fill) vs per-case 3D largest CC."""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet import UNet
from predict import predict_img

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join('..', '..', 'Pytorch-UNet', 'checkpoints', 'prostate_unet_best.pth')
CROP = 176

device = torch.device('cuda')
net = UNet(n_channels=1, n_classes=1, bilinear=False, se=False).to(device)
sd = torch.load(CKPT, map_location=device)
sd.pop('mask_values', None)
net.load_state_dict(sd)
net.eval()


def dice(a, b):
    i = np.logical_and(a > 0, b > 0).sum()
    return 2 * i / (i + np.logical_or(a > 0, b > 0).sum())


def morph2d(m):
    m = m > 0
    lab, n = ndi.label(m)
    if n > 1:  # keep largest connected component
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (np.argmax(sizes) + 1)
    m = ndi.binary_closing(m, structure=np.ones((5, 5)))
    m = ndi.binary_fill_holes(m)
    return m


cases = {}
for f in sorted(glob.glob(os.path.join('..', 'data', 'prostate_slices', 'val', 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), f, f.replace('imgs', 'masks').replace('.png', '_mask.png')))

scores = {'raw': [], 'morph2d': [], 'morph3d': []}
for cid, slices in sorted(cases.items()):
    slices.sort()
    gts, raws, m2ds = [], [], []
    for k, f, gp in slices:
        img = np.asarray(Image.open(f))
        gt = np.asarray(Image.open(gp)) > 0
        ys, xs = np.nonzero(gt)
        cy, cx = int(ys.mean()), int(xs.mean())
        y0 = int(np.clip(cy - CROP // 2, 0, img.shape[0] - CROP))
        x0 = int(np.clip(cx - CROP // 2, 0, img.shape[1] - CROP))
        pred_crop = predict_img(net, Image.fromarray(img[y0:y0 + CROP, x0:x0 + CROP]), device, 1.0) > 0
        raw = np.zeros(img.shape, bool)
        raw[y0:y0 + CROP, x0:x0 + CROP] = pred_crop
        gts.append(gt)
        raws.append(raw)
        m2ds.append(morph2d(raw))
    raw3 = np.stack(raws)
    lab, n = ndi.label(raw3)
    if n > 1:
        sizes = ndi.sum(raw3, lab, range(1, n + 1))
        raw3 = lab == (np.argmax(sizes) + 1)
    m3 = np.stack(m2ds)
    lab, n = ndi.label(m3)
    if n > 1:
        sizes = ndi.sum(m3, lab, range(1, n + 1))
        m3 = lab == (np.argmax(sizes) + 1)

    for key, pred_vol in (('raw', raw3), ('morph2d', np.stack(m2ds)), ('morph3d', m3)):
        d = dice(pred_vol, np.stack(gts))
        scores[key].append(d)
    print(f'{cid}: raw {scores["raw"][-1]:.4f} | morph2D {scores["morph2d"][-1]:.4f} | morph3D {scores["morph3d"][-1]:.4f}')

print()
for key, v in scores.items():
    print(f'{key:>8}: mean 3D Dice {np.mean(v):.4f} (min {np.min(v):.4f})')
