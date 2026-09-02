# -*- coding: utf-8 -*-
"""Overlays for the background-supervised model (bgsup_best.pth), same 4 val cases."""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from unet import UNet

CROP = 176
DATA = os.path.join(ROOT, 'data', 'prostate_slices', 'val')
CKPT = os.path.join(HERE, 'checkpoints', 'bgsup_best.pth')
FIGDIR = os.path.join(ROOT, 'figures')
TARGETS = {'prostate_06': 'p06', 'prostate_18': 'p18', 'prostate_24': 'p24', 'prostate_40': 'p40'}


def overlay(gray, gt, pred, alpha=0.40):
    rgb = np.stack([gray] * 3, -1).astype(np.float32)
    g, p = gt > 0, pred > 0
    rgb[g] = rgb[g] * (1 - alpha) + np.array([0., 180., 0.]) * alpha
    rgb[p] = rgb[p] * (1 - alpha) + np.array([225., 0., 0.]) * alpha
    for m, c in ((g, [0., 255., 0.]), (p, [255., 70., 0.])):
        rgb[(m ^ np.roll(m, 1, 0)) | (m ^ np.roll(m, 1, 1))] = c
    return np.clip(rgb, 0, 255).astype(np.uint8)


def dice(a, b):
    i = np.logical_and(a > 0, b > 0).sum()
    return 2 * i / (i + np.logical_or(a > 0, b > 0).sum())


model = UNet(n_channels=1, n_classes=2, bilinear=False, se=False)
device = torch.device('cuda')
model = model.to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd)
model.eval()

for f in sorted(glob.glob(os.path.join(DATA, 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    if cid not in TARGETS or k != f'{len(glob.glob(os.path.join(DATA, "imgs", cid + "_t2_*.png"))) // 2:03d}':
        continue
    img = np.asarray(Image.open(f))
    gt = np.asarray(Image.open(f.replace('imgs', 'masks').replace('.png', '_mask.png'))) > 0
    ys, xs = np.nonzero(gt)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - CROP // 2, 0, 320 - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, 320 - CROP))
    with torch.no_grad():
        out = model(torch.from_numpy(img[y0:y0 + CROP, x0:x0 + CROP] / 255.0).float()[None, None].to(device))
    pred = (torch.sigmoid(out[0, 0]) > 0.5).cpu().numpy()
    d = dice(gt[y0:y0 + CROP, x0:x0 + CROP], pred)
    Image.fromarray(overlay(img[y0:y0 + CROP, x0:x0 + CROP], gt[y0:y0 + CROP, x0:x0 + CROP], pred)).save(
        os.path.join(FIGDIR, f'overlay_bgsup_{TARGETS[cid]}.png'))
    print(f'bgsup {TARGETS[cid]}: Dice={d:.3f}')
