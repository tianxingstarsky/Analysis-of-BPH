# -*- coding: utf-8 -*-
"""Sub-model calibration experiment:
1) run frozen best U-Net on all slices -> coarse masks (GT-centroid crop protocol)
2) train a small refinement U-Net: input [image, coarse mask] -> refined mask
3) evaluate refiner on validation cases (3D Dice, crop-paste protocol)
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from unet import UNet
from predict import predict_img

CROP = 176
DATA = os.path.join(ROOT, 'data', 'prostate_slices')
MAIN_CKPT = os.path.join(os.path.dirname(ROOT), 'Pytorch-UNet', 'checkpoints', 'prostate_unet_best.pth')
REF_CKPT = os.path.join(HERE, 'checkpoints', 'refiner_pos_best.pth')
device = torch.device('cuda')


def center_crop(img, gt):
    ys, xs = np.nonzero(gt)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - CROP // 2, 0, img.shape[0] - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, img.shape[1] - CROP))
    return y0, x0, img[y0:y0 + CROP, x0:x0 + CROP], gt[y0:y0 + CROP, x0:x0 + CROP]


def dice(a, b):
    i = np.logical_and(a > 0, b > 0).sum()
    return 2 * i / (i + np.logical_or(a > 0, b > 0).sum())


def dice_loss(p, g, eps=1.0):
    inter = (p * g).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


# ---------- main model -> coarse masks ----------
main = UNet(n_channels=1, n_classes=1, bilinear=False, se=False).to(device)
sd = torch.load(MAIN_CKPT, map_location=device)
sd.pop('mask_values', None)
main.load_state_dict(sd)
main.eval()

for split in ('train', 'val'):
    for f in sorted(glob.glob(os.path.join(DATA, split, 'imgs', '*.png'))):
        name = os.path.basename(f)[:-4]
        gp = f.replace('imgs', 'masks').replace('.png', '_mask.png')
        out = os.path.join(DATA, split, 'coarse', name + '.png')
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        img = np.asarray(Image.open(f))
        gt = np.asarray(Image.open(gp)) > 0
        y0, x0, crop_i, crop_g = center_crop(img, gt)
        pred = predict_img(main, Image.fromarray(crop_i), device, 1.0) > 0
        full = np.zeros(img.shape, np.uint8)
        full[y0:y0 + CROP, x0:x0 + CROP] = pred.astype(np.uint8) * 255
        Image.fromarray(full).save(out)
    print(f'coarse masks [{split}] done', flush=True)

# ---------- refiner training data ----------
def load_triplet(split, f, pos_channels=False):
    name = os.path.basename(f)[:-4]
    img = np.asarray(Image.open(f))
    coarse = np.asarray(Image.open(os.path.join(DATA, split, 'coarse', name + '.png'))) > 0
    gt = np.asarray(Image.open(f.replace('imgs', 'masks').replace('.png', '_mask.png'))) > 0
    chans = [img / 255.0, coarse.astype(np.float32)]
    if pos_channels:
        # target-centered relative position encoding (CoordConv/SDF style)
        sdf = ndi.distance_transform_edt(~coarse) - ndi.distance_transform_edt(coarse)
        ys, xs = np.nonzero(coarse)
        cy, cx = (ys.mean(), xs.mean()) if len(ys) else (img.shape[0] / 2, img.shape[1] / 2)
        yy, xx = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
        rel_y = np.clip((yy - cy) / 176.0, -1, 1)
        rel_x = np.clip((xx - cx) / 176.0, -1, 1)
        chans += [np.clip(sdf / 16.0, -1, 1), rel_y, rel_x]
    two = np.stack(chans)  # [C,H,W]
    return two, gt.astype(np.float32)


train_files = sorted(glob.glob(os.path.join(DATA, 'train', 'imgs', '*.png')))
val_files = sorted(glob.glob(os.path.join(DATA, 'val', 'imgs', '*.png')))

# ---------- refiner model: small U-Net, 2-ch input ----------
POS = True
IN_CH = 5 if POS else 2
refiner = UNet(n_channels=IN_CH, n_classes=1, bilinear=False, se=False).to(device)
optim = torch.optim.AdamW(refiner.parameters(), lr=2e-4, weight_decay=1e-4)
EPOCHS = 25
BATCH = 8

best_val = -1
for epoch in range(1, EPOCHS + 1):
    random_idx = np.random.permutation(len(train_files))
    refiner.train()
    tot = 0
    for s in range(0, len(random_idx), BATCH):
        batch = [train_files[i] for i in random_idx[s:s + BATCH]]
        imgs, gts = [], []
        for f in batch:
            two, g = load_triplet('train', f, pos_channels=POS)
            imgs.append(two)
            gts.append(g)
        x = torch.from_numpy(np.stack(imgs)).float().to(device)          # [B, 2, H, W]: image + coarse
        y = torch.from_numpy(np.stack(gts)).float().unsqueeze(1).to(device)
        logits = refiner(x)
        loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(torch.sigmoid(logits), y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        tot += float(loss)
    # validation
    refiner.eval()
    vds = []
    with torch.no_grad():
        for f in val_files:
            two, g = load_triplet('val', f, pos_channels=POS)
            x = torch.from_numpy(two[None]).float().to(device)
            p = (torch.sigmoid(refiner(x))[0, 0] > 0.5).cpu().numpy()
            vds.append(dice(g, p))
    vd = float(np.mean(vds))
    print(f'epoch {epoch}/{EPOCHS}: train loss {tot/len(range(0, len(random_idx), BATCH)):.4f} | val 2D dice {vd:.4f}', flush=True)
    if vd > best_val:
        best_val = vd
        os.makedirs(os.path.dirname(REF_CKPT), exist_ok=True)
        torch.save(refiner.state_dict(), REF_CKPT)

print('best val 2D dice:', best_val)

# ---------- final 3D eval with refiner ----------
refiner.load_state_dict(torch.load(REF_CKPT, map_location=device))
refiner.eval()
cases = {}
for f in val_files:
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), f))
inter = union = 0
for cid, slices in sorted(cases.items()):
    for k, f in slices:
        two, g = load_triplet('val', f, pos_channels=POS)
        with torch.no_grad():
            p = (torch.sigmoid(refiner(torch.from_numpy(two[None]).float().to(device)))[0, 0] > 0.5).cpu().numpy()
        inter += np.logical_and(p, g > 0).sum()
        union += np.logical_or(p, g > 0).sum()
print(f'refined per-case 3D Dice: {2*inter/(inter+union):.4f} (raw was 0.8725)')
