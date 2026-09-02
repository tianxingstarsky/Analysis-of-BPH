# -*- coding: utf-8 -*-
"""SAFE fine-tuning of MedSAM2 on our prostate slices.
Frozen: image encoder, prompt encoder, memory attention, memory encoder.
Trained: mask decoder (+ output hypernetwork) only.
Anti-forgetting: no memory weights touched, low LR, box prompt jitter.
"""
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CFG = "configs/sam2.1_hiera_t512.yaml"
CKPT = os.path.join(MEDSAM2_DIR, "checkpoints", "MedSAM2_latest.pt")
TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prostate_slices", "train")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MedSAM2", "checkpoints", "medsam2_ft_prostate.pt")
EPOCHS = 20
LR = 2e-5
rng = np.random.default_rng(42)
random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda')
sam2_model = build_sam2(CFG, CKPT, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# ---- freeze everything except the mask decoder (+ its hypernetwork MLPs) ----
trainable = []
for n, p in sam2_model.named_parameters():
    p.requires_grad = ('mask_decoder' in n or 'output_hypernetworks_mlps' in n)
    if p.requires_grad:
        trainable.append(n)
sam2_model.sam_mask_decoder.train()
print(f'trainable params: {sum(p.numel() for n, p in sam2_model.named_parameters() if p.requires_grad)/1e6:.1f}M '
      f'({len(trainable)} tensors), e.g. {trainable[:2]}')

optim = torch.optim.AdamW([p for p in sam2_model.parameters() if p.requires_grad], lr=LR, weight_decay=0)


def jitter_box(box, w=320, h=320):
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    s = rng.uniform(0.85, 1.15)
    cx = (x0 + x1) / 2 + rng.uniform(-0.1, 0.1) * bw
    cy = (y0 + y1) / 2 + rng.uniform(-0.1, 0.1) * bh
    nw, nh = bw * s, bh * s
    return np.array([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]).clip([0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])


samples = []
for f in sorted(glob.glob(os.path.join(TRAIN, 'imgs', '*.png'))):
    name = os.path.basename(f)[:-4]
    gp = f.replace('imgs', 'masks').replace('.png', '_mask.png')
    samples.append((f, gp))
print(f'{len(samples)} training slices')

gt_t = {}


def dice_loss(p, g, eps=1.0):
    inter = (p * g).sum()
    union = p.sum() + g.sum()
    return 1 - (2 * inter + eps) / (union + eps)


for epoch in range(1, EPOCHS + 1):
    random.shuffle(samples)
    tot = n = 0
    for f, gp in samples:
        img = np.asarray(Image.open(f).convert('RGB'))
        g = np.asarray(Image.open(gp)) > 0
        ys, xs = np.nonzero(g)
        box = jitter_box(np.array([xs.min(), ys.min(), xs.max(), ys.max()]))
        gt = torch.from_numpy(g.astype(np.float32)).to(device)

        predictor.set_image(img)
        # _prep_prompts (undecorated) transforms the box; bypass the @torch.no_grad
        # decorators via __wrapped__ so gradients flow into the mask decoder
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(None, None, box, None, True)
        pred_fn = type(predictor)._predict.__wrapped__
        with torch.enable_grad():
            masks_out, ious, low_res_masks = pred_fn(
                predictor, unnorm_coords, labels, unnorm_box, None, False, True
            )
        logits_t = low_res_masks.squeeze(1)[0] if low_res_masks.dim() == 4 else low_res_masks[0]
        if logits_t.shape != gt.shape:
            logits_t = F.interpolate(logits_t[None, None], size=gt.shape, mode='bilinear',
                                     align_corners=False)[0, 0]
        if not logits_t.requires_grad:
            raise RuntimeError('no grad flow — check predictor path')

        prob = torch.sigmoid(logits_t)
        loss = F.binary_cross_entropy_with_logits(logits_t, gt) + dice_loss(prob, gt)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        tot += float(loss)
        n += 1
    print(f'epoch {epoch}/{EPOCHS}: loss {tot / max(n, 1):.4f}', flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(sam2_model.state_dict(), OUT)
print('saved', OUT, flush=True)
