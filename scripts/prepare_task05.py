# -*- coding: utf-8 -*-
"""MSD Task05 Prostate -> 2D slices for milesial/Pytorch-UNet.

This packaging: imagesTr/prostate_XX.nii.gz is 4D (320,320,N,2), ch0=T2, ch1=ADC.
labelsTr/prostate_XX.nii.gz: 0=bg, 1=PZ (peripheral zone), 2=TZ (transition zone).
Whole gland = label > 0. Only annotated slices exported; split BY CASE.
Usage: python prepare_task05.py t2|adc
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
SRC = ROOT / "dataset" / "Task05_Prostate"
OUT = ROOT / "data" / "prostate_slices"
RESIZE = 320
VAL_CASES = {"prostate_06", "prostate_18", "prostate_24", "prostate_40"}  # 4 of 32


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Per-CASE normalization (nnU-Net style): whole-volume stats so the same
    tissue maps consistently across all slices of one patient."""
    body = vol > (0.02 * vol.max())          # exclude pure-air background
    vals = vol[body]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    mu, sd = vals.mean(), vals.std() + 1e-8
    z = np.clip((np.clip(vol, lo, hi) - mu) / sd, -3, 3)
    return ((z + 3) / 6 * 255).astype(np.uint8)


def main():
    mod = sys.argv[1]  # "t2" or "adc"
    ch = {"t2": 0, "adc": 1}[mod]
    for split in ("train", "val"):
        (OUT / split / "imgs").mkdir(parents=True, exist_ok=True)
        (OUT / split / "masks").mkdir(parents=True, exist_ok=True)

    total = 0
    for img_file in sorted((SRC / "imagesTr").glob("prostate_*.nii.gz")):
        cid = img_file.stem.replace(".nii", "")
        vol = np.asarray(nib.load(img_file).dataobj, dtype=np.float32)[:, :, :, ch]
        vol = normalize_volume(vol)
        lbl = np.asarray(nib.load(SRC / "labelsTr" / f"{cid}.nii.gz").dataobj)
        assert vol.shape == lbl.shape, f"shape mismatch in {cid}"
        split = "val" if cid in VAL_CASES else "train"

        n_kept = 0
        for k in range(vol.shape[2]):
            m = lbl[:, :, k]
            if m.sum() == 0:
                continue
            im = Image.fromarray(vol[:, :, k]).resize((RESIZE, RESIZE), Image.BICUBIC)
            mk = Image.fromarray(((m > 0) * 255).astype(np.uint8)).resize((RESIZE, RESIZE), Image.NEAREST)
            im.save(OUT / split / "imgs" / f"{cid}_{mod}_{k:03d}.png")
            mk.save(OUT / split / "masks" / f"{cid}_{mod}_{k:03d}_mask.png")
            n_kept += 1
        total += n_kept
        print(f"{cid} [{split}]: {n_kept} slices")
    print(f"TOTAL {mod} slices: {total}")


if __name__ == "__main__":
    main()
