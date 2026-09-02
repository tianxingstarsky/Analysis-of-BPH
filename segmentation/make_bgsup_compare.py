# -*- coding: utf-8 -*-
"""Side-by-side figure: fg-only U-Net vs background-supervised U-Net on the same
middle slices. Row 1 = 无背景监督, Row 2 = 显式背景监督. Green=GT, red=pred."""
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

CROP = 176
DATA = os.path.join(ROOT, 'data', 'prostate_slices')
FG_CKPT = os.path.join(os.path.dirname(ROOT), 'Pytorch-UNet', 'checkpoints', 'prostate_unet_best.pth')
BG_CKPT = os.path.join(HERE, 'checkpoints', 'bgsup_best.pth')
OUT = os.path.join(ROOT, 'figures', 'fig_bgsup_compare.png')
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


def build(ckpt, dual):
    net = UNet(n_channels=1, n_classes=2 if dual else 1, bilinear=False, se=False).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd.pop('mask_values', None)
    net.load_state_dict(sd)
    net.eval()
    return net


def predict(net, img, dual):
    x = torch.from_numpy(img / 255.0).float()[None, None].to(device)
    with torch.no_grad():
        out = net(x)
    if dual:
        return (torch.sigmoid(out[0, 0]) > 0.5).cpu().numpy()
    return (out[0, 0] > 0).cpu().numpy()


fg_net = build(FG_CKPT, dual=False)
bg_net = build(BG_CKPT, dual=True)

cases = {}
for f in sorted(glob.glob(os.path.join(DATA, 'val', 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), f, f.replace('imgs', 'masks').replace('.png', '_mask.png')))

S = 176
try:
    f_big = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    f_sm = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 13)
except OSError:
    f_big = f_sm = ImageFont.load_default()

W = 4 * (S + 6) + 6
H = 30 + 2 * (S + 40)
canvas = Image.new('RGB', (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
draw.text((W // 2 - 140, 6), '上行：无背景监督    下行：显式背景监督（同切片）', fill=(30, 30, 30), font=f_big)

for c, cid in enumerate(['prostate_06', 'prostate_18', 'prostate_24', 'prostate_40']):
    slices = sorted(cases[cid])
    k, f, gp = slices[len(slices) // 2]  # middle slice
    img = np.asarray(Image.open(f))
    gt = np.asarray(Image.open(gp)) > 0
    p_fg = predict(fg_net, img, dual=False)
    p_bg = predict(bg_net, img, dual=True)
    x = 6 + c * (S + 6)
    y = 30
    canvas.paste(Image.fromarray(overlay(img, gt, p_fg)), (x, y))
    canvas.paste(Image.fromarray(overlay(img, gt, p_bg)), (x, y + S + 26))
    draw.text((x + 4, y + S + 2), f'无背景监督 Dice {dice(gt, p_fg):.3f}', fill=(180, 30, 30), font=f_sm)
    draw.text((x + 4, y + 2 * S + 28), f'背景监督 Dice {dice(gt, p_bg):.3f}', fill=(20, 130, 20), font=f_sm)
    draw.text((x + 4, y + 2 * S + 44), cid.replace('prostate', 'p'), fill=(60, 60, 60), font=f_sm)

canvas.save(OUT)
print('saved', OUT)
