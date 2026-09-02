# -*- coding: utf-8 -*-
"""Generate MedSAM2 (zero-shot, box-per-slice) prediction overlays for README.
Same style as U-Net overlays: green=GT, red=prediction, overlap=yellow."""
import glob
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_medsam2_dir():
    cands = [
        os.environ.get('MEDSAM2_DIR'),
        os.path.join(ROOT, 'MedSAM2'),
        os.path.join(os.path.dirname(ROOT), 'MedSAM2'),
    ]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'sam2')):
            return c
    raise FileNotFoundError('clone MedSAM2 next to this repo or set MEDSAM2_DIR env var')


MEDSAM2_DIR = find_medsam2_dir()
sys.path.insert(0, MEDSAM2_DIR)
from sam2.build_sam import build_sam2_video_predictor

CFG = "configs/sam2.1_hiera_t512.yaml"
CKPT = os.path.join(MEDSAM2_DIR, "checkpoints", "MedSAM2_latest.pt")
VAL = os.path.join(ROOT, "data", "prostate_slices", "val")
FRAMES = os.path.join(ROOT, "medsam2_frames")
FIGDIR = os.path.join(ROOT, "figures")
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


predictor = build_sam2_video_predictor(CFG, CKPT, device='cuda')

cases = {}
for f in sorted(glob.glob(os.path.join(VAL, 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), f, f.replace('imgs', 'masks').replace('.png', '_mask.png')))

if os.path.exists(FRAMES):
    shutil.rmtree(FRAMES)

for cid, slices in sorted(cases.items()):
    if cid not in TARGETS:
        continue
    slices.sort()
    frames = os.path.join(FRAMES, cid)
    os.makedirs(frames, exist_ok=True)
    gts = {}
    for j, (k, f, gp) in enumerate(slices):
        Image.open(f).convert('RGB').save(os.path.join(frames, f'{j:03d}.jpg'), quality=98)
        gts[j] = np.asarray(Image.open(gp))
    state = predictor.init_state(video_path=frames)
    for j in gts:  # zero-shot box-per-slice protocol
        m = gts[j] > 0
        ys, xs = np.nonzero(m)
        predictor.add_new_points_or_box(state, frame_idx=j, obj_id=1,
                                        box=np.array([xs.min(), ys.min(), xs.max(), ys.max()]))
    preds = {}
    for fi, _, logits in predictor.propagate_in_video(state):
        preds[fi] = (logits[0][0] > 0).cpu().numpy()
    del state
    torch.cuda.empty_cache()

    mid = len(slices) // 2
    j, f, gp = slices[mid]
    img = np.asarray(Image.open(f))
    d = dice(gts[mid], preds[mid])
    Image.fromarray(overlay(img, gts[mid], preds[mid])).save(
        os.path.join(FIGDIR, f'overlay_medsam2_{TARGETS[cid]}.png'))
    print(f'medsam2 {TARGETS[cid]}: Dice={d:.3f}')

shutil.rmtree(FRAMES, ignore_errors=True)
print('done')
