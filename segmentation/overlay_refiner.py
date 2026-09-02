# -*- coding: utf-8 -*-
"""Visualize sub-model refinement effect: coarse vs refined overlays.
For each val case, show the slice where refinement gained the most.
Row 1 = coarse (raw U-Net), Row 2 = refined (sub-model). Green=GT, red=pred."""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from unet import UNet
from predict import predict_img

CROP = 176
DATA = os.path.join(ROOT, 'data', 'prostate_slices')
MAIN_CKPT = os.path.join(os.path.dirname(ROOT), 'Pytorch-UNet', 'checkpoints', 'prostate_unet_best.pth')
REF_CKPT = os.path.join(HERE, 'checkpoints', 'refiner_best.pth')
device = torch.device('cuda')


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


main = UNet(n_channels=1, n_classes=1, bilinear=False, se=False).to(device)
sd = torch.load(MAIN_CKPT, map_location=device)
sd.pop('mask_values', None)
main.load_state_dict(sd)
main.eval()

refiner = UNet(n_channels=2, n_classes=1, bilinear=False, se=False).to(device)
sd = torch.load(REF_CKPT, map_location=device)
refiner.load_state_dict(sd)
refiner.eval()

cases = {}
for f in sorted(glob.glob(os.path.join(DATA, 'val', 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), f, f.replace('imgs', 'masks').replace('.png', '_mask.png')))

S = 176
try:
    f_big = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    f_sm = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 13)
except OSError:
    f_big = f_sm = ImageFont.load_default()

selected = []  # (case_label, coarse_overlay, refined_overlay, d_coarse, d_refined)
for cid, slices in sorted(cases.items()):
    slices.sort()
    best = None
    for k, f, gp in slices:
        img = np.asarray(Image.open(f))
        gt = np.asarray(Image.open(gp)) > 0
        ys, xs = np.nonzero(gt)
        cy, cx = int(ys.mean()), int(xs.mean())
        y0 = int(np.clip(cy - CROP // 2, 0, 320 - CROP))
        x0 = int(np.clip(cx - CROP // 2, 0, 320 - CROP))
        crop_i = img[y0:y0 + CROP, x0:x0 + CROP]
        crop_g = gt[y0:y0 + CROP, x0:x0 + CROP]
        coarse = predict_img(main, Image.fromarray(crop_i), device, 1.0) > 0
        x = torch.from_numpy(np.stack([crop_i / 255.0, coarse.astype(np.float32)])[None]).float().to(device)
        with torch.no_grad():
            refined = (torch.sigmoid(refiner(x))[0, 0] > 0.5).cpu().numpy()
        dc, dr = dice(crop_g, coarse), dice(crop_g, refined)
        if best is None or (dr - dc) > (best[4] - best[3]):
            best = (cid.replace('prostate', 'p'), overlay(crop_i, crop_g, coarse),
                    overlay(crop_i, crop_g, refined), dc, dr)
    selected.append(best)

# compose: col per case; row1 coarse, row2 refined; dice labels under each panel
W = 4 * (S + 6) + 6
H = 30 + 2 * (S + 40)
canvas = Image.new('RGB', (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
draw.text((W // 2 - 90, 6), '上行：粗分割    下行：子模型精修', fill=(30, 30, 30), font=f_big)
for c, (lab, oc, orr, dc, dr) in enumerate(selected):
    x = 6 + c * (S + 6)
    y = 30
    canvas.paste(Image.fromarray(oc), (x, y))
    canvas.paste(Image.fromarray(orr), (x, y + S + 26))
    draw.text((x + 4, y + S + 2), f'粗 Dice {dc:.3f}', fill=(180, 30, 30), font=f_sm)
    draw.text((x + 4, y + 2 * S + 28), f'精 Dice {dr:.3f}', fill=(20, 130, 20), font=f_sm)
    draw.text((x + 4, y + 2 * S + 44), lab, fill=(60, 60, 60), font=f_sm)
canvas.save(os.path.join(ROOT, 'figures', 'fig_refiner_effect.png'))
for lab, oc, orr, dc, dr in selected:
    print(f'{lab}: coarse {dc:.3f} -> refined {dr:.3f} ({(dr-dc)*1000:+.0f} mille)')
print('saved figures/fig_refiner_effect.png')
