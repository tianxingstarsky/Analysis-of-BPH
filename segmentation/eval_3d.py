# -*- coding: utf-8 -*-
"""Per-case 3D volume Dice (MSD/paper standard).
Crop-trained models: predict on the deterministic GT-centroid crop, paste back
into the full frame, stack slices per case, compute volume-overlap Dice.
usage: python eval_3d.py [--no-se] [--crop 176] checkpoint.pth"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_img
from unet import UNet

argv = sys.argv[1:]
se = '--no-se' not in argv
crop = 0
classes = 1
if '--crop' in argv:
    i = argv.index('--crop')
    crop = int(argv[i + 1])
    del argv[i:i + 2]
if '--classes' in argv:
    i = argv.index('--classes')
    classes = int(argv[i + 1])
    del argv[i:i + 2]
ck = [a for a in argv if a.endswith('.pth')][0]

device = torch.device('cuda')
net = UNet(n_channels=1, n_classes=classes, bilinear=False, se=se).to(device)
sd = torch.load(ck, map_location=device)
sd.pop('mask_values', None)
net.load_state_dict(sd)
net.eval()

cases = {}
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'prostate_slices', 'val', 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), f))

print(f'{"case":<14}{"3D Dice":>9}')
scores = []
for cid, slices in sorted(cases.items()):
    inter = union = 0
    for k, f in slices:
        name = os.path.basename(f)[:-4]
        img = np.asarray(Image.open(f))
        gt = np.asarray(Image.open(f.replace('imgs', 'masks').replace('.png', '_mask.png')))
        frame_pred = np.zeros_like(img)
        if crop:
            ys, xs = np.nonzero(gt)
            cy, cx = (ys.mean(), xs.mean()) if len(ys) else (160, 160)
            r = crop / 2
            y0 = int(np.clip(cy - r, 0, img.shape[0] - crop))
            x0 = int(np.clip(cx - r, 0, img.shape[1] - crop))
            pred_crop = predict_img(net, Image.fromarray(img[y0:y0 + crop, x0:x0 + crop]), device, 1.0)
            frame_pred[y0:y0 + crop, x0:x0 + crop] = pred_crop
        else:
            frame_pred = predict_img(net, Image.fromarray(img), device, 1.0)
        inter += np.logical_and(frame_pred > 0, gt > 0).sum()
        union += np.logical_or(frame_pred > 0, gt > 0).sum()
    d = 2 * inter / (inter + union)
    scores.append(d)
    print(f'{cid:<14}{d:>9.4f}')
print(f'\nper-case 3D Dice: mean {np.mean(scores):.4f} (min {np.min(scores):.4f}, max {np.max(scores):.4f}, n={len(scores)})')
