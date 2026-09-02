# -*- coding: utf-8 -*-
"""Phase 2: ROI texture features per case (methodology follows 杨建丽 2025).

For every MSD Task05 training case (4D NIfTI, ch0=T2, ch1=ADC), using GT mask:
  - whole-gland volume (mL), peripheral-zone volume (mL), PZ fraction
  - mean T2WI signal inside gland (mean-SI-T2WI analog)
  - mean ADC value inside gland
Output: features_gt.csv
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).parent.parent
SRC = ROOT / "dataset" / "Task05_Prostate"


def main():
    rows = []
    for img_file in sorted((SRC / "imagesTr").glob("prostate_*.nii.gz")):
        cid = img_file.stem.replace(".nii", "")
        data = np.asarray(nib.load(img_file).dataobj, dtype=np.float32)
        t2, adc = data[..., 0], data[..., 1]
        lbl = np.asarray(nib.load(SRC / "labelsTr" / f"{cid}.nii.gz").dataobj)
        whole, pz = lbl > 0, lbl == 1

        zooms = nib.load(img_file).header.get_zooms()[:3]
        vox_ml = float(np.prod(zooms)) / 1000.0
        gland_vol = float(whole.sum()) * vox_ml
        pz_vol = float(pz.sum()) * vox_ml
        mean_si = float(t2[whole].mean())
        mean_adc = float(adc[whole].mean())
        rows.append(f"{cid},{gland_vol:.2f},{pz_vol:.2f},{pz_vol / gland_vol:.3f},{mean_si:.2f},{mean_adc:.4f}")
        print(rows[-1])

    out = ROOT / "features_gt.csv"
    out.write_text("case,gland_vol_ml,pz_vol_ml,pz_frac,mean_si_t2,mean_adc\n" + "\n".join(rows), encoding="utf-8")
    print(f"\nsaved {out} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
