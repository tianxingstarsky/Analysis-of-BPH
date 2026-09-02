# -*- coding: utf-8 -*-
"""Per-case 3D volume Dice (MSD/paper standard).
Crop-trained models: predict on the deterministic GT-centroid crop, paste back
into the full frame, stack slices per case, compute volume-overlap Dice.
usage: python eval_3d.py [--no-se] [--se] [--dual-sigmoid] [--crop 176]
                         [--classes N] checkpoint.pth

- --dual-sigmoid : background-supervised model (2-ch dual-sigmoid head,
  prediction = sigmoid(ch0) > 0.5); implies 176 GT-centroid crop protocol
- --crop 176     : GT-centroid crop inference (crop-trained models)
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from predict import predict_img
from unet import UNet

argv = sys.argv[1:]
se = '--no-se' not in argv
dual = '--dual-sigmoid' in argv
if dual:
    argv.remove('--dual-sigmoid')
crop = 0
if '--crop' in argv:
    i = argv.index('--crop')
    crop = int(argv[i + 1])
    del argv[i:i + 2]
classes = 2 if dual else 1
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

if dual and not crop:
    crop = 176  # bg-supervised model is crop-trained


def predict_frame(img, gt):
    """full-frame binary prediction for one slice"""
    if crop:
        ys, xs = np.nonzero(gt)
        cy, cx = (ys.mean(), xs.mean()) if len(ys) else (img.shape[0] / 2, img.shape[1] / 2)
        y0 = int(np.clip(cy - crop / 2, 0, img.shape[0] - crop))
        x0 = int(np.clip(cx - crop / 2, 0, img.shape[1] - crop))
        crop_i = img[y0:y0 + crop, x0:x0 + crop]
        if dual:
            x = torch.from_numpy(crop_i / 255.0).float()[None, None].to(device)
            with torch.no_grad():
                pred = (torch.sigmoid(net(x)[0, 0]) > 0.5).cpu().numpy()
        else:
            pred = predict_img(net, Image.fromarray(crop_i), device, 1.0)
        frame = np.zeros(img.shape, bool)
        frame[y0:y0 + crop, x0:x0 + crop] = pred > 0
        return frame
    return predict_img(net, Image.fromarray(img), device, 1.0) > 0


cases = {}
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'prostate_slices', 'val', 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), f))

print(f'{"case":<14}{"3D Dice":>9}')
scores = []
for cid, slices in sorted(cases.items()):
    inter = union = 0
    for k, f in slices:
        img = np.asarray(Image.open(f))
        gt = np.asarray(Image.open(f.replace('imgs', 'masks').replace('.png', '_mask.png'))) > 0
        pred = predict_frame(img, gt)
        inter += np.logical_and(pred, gt).sum()
        union += np.logical_or(pred, gt).sum()
    d = 2 * inter / (inter + union)
    scores.append(d)
    print(f'{cid:<14}{d:>9.4f}')
print(f'\nper-case 3D Dice: mean {np.mean(scores):.4f} (min {np.min(scores):.4f}, max {np.max(scores):.4f}, n={len(scores)})')
