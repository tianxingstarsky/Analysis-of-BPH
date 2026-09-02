# -*- coding: utf-8 -*-
"""MedSAM2 zero-shot evaluation on our prostate MRI validation cases.
Mode A: box prompt on every annotated slice (oracle-box 2D protocol).
Mode B: single box on the middle slice, propagate across the volume (3D tracking).
"""
import glob
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MedSAM2"))
from sam2.build_sam import build_sam2_video_predictor

CFG = "configs/sam2.1_hiera_t512.yaml"
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MedSAM2", "checkpoints", "MedSAM2_latest.pt")
VAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prostate_slices", "val")
TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medsam2_frames")

box_mode = 'gt'
if '--box' in sys.argv:
    i = sys.argv.index('--box')
    box_mode = sys.argv[i + 1]
    del sys.argv[i:i + 2]
ckpt_arg = [a for a in sys.argv[1:] if a.endswith('.pt')]
CKPT = ckpt_arg[0] if ckpt_arg else CKPT
rng = np.random.default_rng(42)


def jitter_box(box, w=320, h=320):
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    s = rng.uniform(0.85, 1.15)
    cx = (x0 + x1) / 2 + rng.uniform(-0.1, 0.1) * bw
    cy = (y0 + y1) / 2 + rng.uniform(-0.1, 0.1) * bh
    nw, nh = bw * s, bh * s
    return np.array([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]).clip([0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])


def dice(a, b):
    i = np.logical_and(a > 0, b > 0).sum()
    return 2 * i / (i + np.logical_or(a > 0, b > 0).sum())


predictor = build_sam2_video_predictor(CFG, CKPT, device='cuda')

cases = {}
for f in sorted(glob.glob(os.path.join(VAL, 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    cid, k = name.rsplit('_t2_', 1)
    cases.setdefault(cid, []).append((int(k), f, f.replace('imgs', 'masks').replace('.png', '_mask.png')))

for cid, slices in sorted(cases.items()):
    slices.sort()
    frames = os.path.join(TMP, cid)
    if os.path.exists(frames):
        shutil.rmtree(frames)
    os.makedirs(frames)
    gt = {}
    for j, (k, f, gp) in enumerate(slices):
        Image.open(f).convert('RGB').save(os.path.join(frames, f'{j:03d}.jpg'), quality=98)
        gt[j] = np.asarray(Image.open(gp))
    n = len(slices)
    mid = n // 2

    results = {}
    for mode, prompt_frames in (('boxall', range(n)), ('propagate', [mid])):
        state = predictor.init_state(video_path=frames)
        for j in prompt_frames:
            m = gt[j]
            ys, xs = np.nonzero(m)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            if box_mode == 'jitter':
                box = jitter_box(box)
            predictor.add_new_points_or_box(state, frame_idx=j, obj_id=1, box=box)
        preds = {}
        for fi, obj_ids, logits in predictor.propagate_in_video(state):
            preds[fi] = (logits[0][0] > 0).cpu().numpy()
        per_slice = {j: dice(gt[j], preds.get(j, np.zeros_like(gt[j]))) for j in gt}
        inter = sum(np.logical_and(gt[j], preds.get(j, 0)).sum() for j in gt)
        union = sum(np.logical_or(gt[j], preds.get(j, 0)).sum() for j in gt)
        d3 = 2 * inter / (inter + union)
        results[mode] = (float(np.mean(list(per_slice.values()))), d3)
        del state
        torch.cuda.empty_cache()

    print(f'{cid}: n={n} | '
          f'boxall 2D {results["boxall"][0]:.4f} / 3D {results["boxall"][1]:.4f} | '
          f'propagate 2D {results["propagate"][0]:.4f} / 3D {results["propagate"][1]:.4f}')

shutil.rmtree(TMP, ignore_errors=True)
print('done')
