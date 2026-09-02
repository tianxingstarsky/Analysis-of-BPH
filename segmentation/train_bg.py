# -*- coding: utf-8 -*-
"""Explicit background supervision experiment (user idea: 模型需要学习分割背景).
Dual-sigmoid head: ch0 supervised by foreground GT, ch1 supervised by inverted GT.
Avoids classes=2 softmax competition collapse while explicitly supervising background.
Eval: per-case 3D Dice (crop protocol), compared against fg-only baseline 0.8725.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from unet import UNet

CROP = 176
DATA = os.path.join(ROOT, 'data', 'prostate_slices')
device = torch.device('cuda')


def dice(a, b):
    i = np.logical_and(a > 0, b > 0).sum()
    return 2 * i / (i + np.logical_or(a > 0, b > 0).sum())


def dice_loss(p, g, eps=1.0):
    inter = (p * g).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def crop_box(gt):
    ys, xs = np.nonzero(gt)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - CROP // 2, 0, 320 - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, 320 - CROP))
    return y0, x0


def load_sample(f, jitter=False):
    img = np.asarray(Image.open(f))
    gt = np.asarray(Image.open(f.replace('imgs', 'masks').replace('.png', '_mask.png'))) > 0
    y0, x0 = crop_box(gt)
    if jitter:
        y0 = int(np.clip(y0 + np.random.uniform(-16, 16), 0, 320 - CROP))
        x0 = int(np.clip(x0 + np.random.uniform(-16, 16), 0, 320 - CROP))
    return img[y0:y0 + CROP, x0:x0 + CROP], gt[y0:y0 + CROP, x0:x0 + CROP], (y0, x0), img.shape


train_files = sorted(glob.glob(os.path.join(DATA, 'train', 'imgs', '*.png')))
val_files = sorted(glob.glob(os.path.join(DATA, 'val', 'imgs', '*.png')))

model = UNet(n_channels=1, n_classes=2, bilinear=False, se=False).to(device)
optim = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-8)
EPOCHS, BATCH = 60, 4

best_3d = -1
for epoch in range(1, EPOCHS + 1):
    model.train()
    np.random.shuffle(train_files)
    tot = 0
    for s in range(0, len(train_files), BATCH):
        batch = train_files[s:s + BATCH]
        imgs, fgs, bgs = [], [], []
        for f in batch:
            c, g, _, _ = load_sample(f, jitter=True)
            imgs.append(c / 255.0)
            fgs.append(g.astype(np.float32))
            bgs.append(1 - g.astype(np.float32))
        x = torch.from_numpy(np.stack(imgs)).float().unsqueeze(1).to(device)
        fg = torch.from_numpy(np.stack(fgs)).float().unsqueeze(1).to(device)
        bg = torch.from_numpy(np.stack(bgs)).float().unsqueeze(1).to(device)
        out = model(x)                                   # [B,2,H,W] dual-sigmoid head
        fg_p, bg_p = torch.sigmoid(out[:, 0:1]), torch.sigmoid(out[:, 1:2])
        loss = (F.binary_cross_entropy_with_logits(out[:, 0], fg.squeeze(1))
                + F.binary_cross_entropy_with_logits(out[:, 1], bg.squeeze(1))
                + dice_loss(fg_p, fg) + dice_loss(bg_p, bg))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        tot += float(loss)

    # ---- validation: per-case 3D dice (fg channel, deterministic crop) ----
    model.eval()
    cases = {}
    for f in val_files:
        c, g, (y0, x0), shape = load_sample(f)
        with torch.no_grad():
            out = model(torch.from_numpy(c / 255.0).float()[None, None].to(device))
        pred = (torch.sigmoid(out[0, 0]) > 0.5).cpu().numpy()
        name = os.path.basename(f)[:-4]
        cid = name.rsplit('_t2_', 1)[0]
        full = np.zeros(shape, bool)
        full[y0:y0 + CROP, x0:x0 + CROP] = pred
        full_gt = np.zeros(shape, bool)
        full_gt[y0:y0 + CROP, x0:x0 + CROP] = g
        cases.setdefault(cid, []).append((dice(full_gt, full), full, full_gt))
    d3 = []
    for cid, items in cases.items():
        pred_v = np.stack([it[1] for it in items])
        gt_v = np.stack([it[2] for it in items])
        d3.append(dice(pred_v, gt_v))
    m = float(np.mean(d3))
    print(f'epoch {epoch}/{EPOCHS}: loss {tot/max(1,len(range(0,len(train_files),BATCH))):.4f} | val 3D dice {m:.4f}', flush=True)
    if m > best_3d:
        best_3d = m
        torch.save(model.state_dict(), os.path.join(HERE, 'checkpoints', 'bgsup_best.pth'))
        with open(os.path.join(HERE, 'checkpoints', 'bgsup_best_3d.txt'), 'w') as fh:
            fh.write(str(m))

print('best val 3D dice:', best_3d, '(fg-only baseline: 0.8725)')
