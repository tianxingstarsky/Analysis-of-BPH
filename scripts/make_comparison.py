# -*- coding: utf-8 -*-
"""Side-by-side comparison figure: U-Net (auto) vs MedSAM2 zero-shot vs MedSAM2 fine-tuned.
Rows = validation cases (middle slice), columns = models. Dice labeled per panel."""
import glob
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "segmentation"))
sys.path.insert(0, os.path.join(ROOT, "MedSAM2") if os.path.isdir(os.path.join(ROOT, "MedSAM2", "sam2"))
                else os.path.join(os.path.dirname(ROOT), "MedSAM2"))
from sam2.build_sam import build_sam2_video_predictor
from unet import UNet
from predict import predict_img

CROP = 176
VAL = os.path.join(ROOT, "data", "prostate_slices", "val")
OUT = os.path.join(ROOT, "figures", "fig_compare_unet_medsam2.png")
UNET_CKPT = os.path.join(os.path.dirname(ROOT), "Pytorch-UNet", "checkpoints", "prostate_unet_best.pth")
MEDSAM2_ZS = os.path.join(ROOT, "MedSAM2", "checkpoints", "MedSAM2_latest.pt")
if not os.path.exists(MEDSAM2_ZS):
    MEDSAM2_ZS = os.path.join(os.path.dirname(ROOT), "MedSAM2", "checkpoints", "MedSAM2_latest.pt")
MEDSAM2_FT = os.path.join(ROOT, "MedSAM2", "checkpoints", "medsam2_ft_prostate.pt")
if not os.path.exists(MEDSAM2_FT):
    MEDSAM2_FT = os.path.join(os.path.dirname(ROOT), "MedSAM2", "checkpoints", "medsam2_ft_prostate.pt")
TMP = os.path.join(ROOT, "medsam2_frames")


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


try:
    f = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    fs = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 13)
except OSError:
    f = fs = ImageFont.load_default()

# ---- gather middle slice of each val case ----
cases = {}
for fp in sorted(glob.glob(os.path.join(VAL, 'imgs', '*.png'))):
    name = os.path.basename(fp)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), fp, fp.replace('imgs', 'masks').replace('.png', '_mask.png')))
targets = []
for cid in sorted(cases):
    slices = sorted(cases[cid])
    k, fp, gp = slices[len(slices) // 2]
    img = np.asarray(Image.open(fp))
    gt = np.asarray(Image.open(gp)) > 0
    ys, xs = np.nonzero(gt)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - CROP // 2, 0, img.shape[0] - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, img.shape[1] - CROP))
    targets.append((cid, img[y0:y0 + CROP, x0:x0 + CROP], gt[y0:y0 + CROP, x0:x0 + CROP]))

# ---- models ----
unet = UNet(n_channels=1, n_classes=1, bilinear=False, se=False).to('cuda')
sd = torch.load(UNET_CKPT, map_location='cuda')
sd.pop('mask_values', None)
unet.load_state_dict(sd)
unet.eval()

FT = os.path.join(ROOT, "MedSAM2", "checkpoints", "medsam2_ft_prostate.pt")
med_ft = build_sam2_video_predictor("configs/sam2.1_hiera_t512.yaml", MEDSAM2_FT, device='cuda')
med_zs = build_sam2_video_predictor("configs/sam2.1_hiera_t512.yaml", MEDSAM2_ZS, device='cuda')


def medsam2_boxall(predictor, frames_dir, frame_boxes):
    state = predictor.init_state(video_path=frames_dir)
    for j, box in frame_boxes.items():
        predictor.add_new_points_or_box(state, frame_idx=j, obj_id=1, box=box)
    preds = {}
    for fi, _, logits in predictor.propagate_in_video(state):
        preds[fi] = (logits[0][0] > 0).cpu().numpy()
    del state
    torch.cuda.empty_cache()
    return preds


rows = []
for ci, (cid, img, gt) in enumerate(targets):
    frames = os.path.join(TMP, cid)
    os.makedirs(frames, exist_ok=True)
    Image.fromarray(img).convert('RGB').save(os.path.join(frames, '000.jpg'), quality=98)
    ys, xs = np.nonzero(gt)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)

    unet_pred = predict_img(unet, Image.fromarray(img), 'cuda', 1.0)

    zs = medsam2_boxall(med_zs, frames, {0: box})[0]
    ft = medsam2_boxall(med_ft, frames, {0: box})[0]

    panels = [overlay(img, gt, unet_pred), overlay(img, gt, zs), overlay(img, gt, ft)]
    dices = [dice(gt, unet_pred), dice(gt, zs), dice(gt, ft)]
    rows.append((cid, panels, dices))
    print(f'{cid}: unet {dices[0]:.3f} | zs {dices[1]:.3f} | ft {dices[2]:.3f}')

shutil.rmtree(TMP, ignore_errors=True)

# ---- compose: 4 rows x 3 cols, column headers + dice labels ----
S = 176
pad, head = 4, 30
W = 3 * S + 2 * pad
H = head + 4 * (S + 22)
canvas = Image.new('RGB', (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
try:
    f = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
except OSError:
    f = ImageFont.load_default()
headers = ['U-Net（全自动）', 'MedSAM2 零样本', 'MedSAM2 安全微调']
for c, h in enumerate(headers):
    draw.text((c * (S + pad) + S // 2 - len(h) * 7, 4), h, fill=(30, 30, 30), font=f)
for r, (cid, panels, dices) in enumerate(rows):
    y = head + r * (S + 22)
    for c, (p, d) in enumerate(zip(panels, dices)):
        canvas.paste(Image.fromarray(p), (c * (S + pad), y))
        draw.text((c * (S + pad) + 4, y + S + 2), f'Dice {d:.3f}', fill=(180, 30, 30), font=fs)
    draw.text((W - 78, y + S + 2), cid.replace('prostate', 'p'), fill=(60, 60, 60), font=fs)
canvas.save(OUT)
print('saved', OUT)
