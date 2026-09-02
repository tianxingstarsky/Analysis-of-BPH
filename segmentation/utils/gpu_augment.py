# -*- coding: utf-8 -*-
"""Batched on-GPU augmentation: affine + elastic warps (grid_sample), intensity
jitter, background-only noise/bias field. Runs on the batch after .to(device),
so the GPU never waits on CPU image processing.

img: [B, C, H, W] float in [0,1]   mask: [B, H, W] long class indices
"""
import torch
import torch.nn.functional as F


def _smooth_field(b, h, w, device, cell=16):
    """Per-sample smooth random field [B, 1, H, W], peak |value| = 1."""
    f = torch.rand(b, 1, max(h // cell, 2), max(w // cell, 2), device=device) * 2 - 1
    f = F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
    return f / f.abs().flatten(1).amax(dim=1).view(b, 1, 1, 1).clamp_min(1e-6)


def gpu_augment(img, mask, scheme='full',
                p_affine=0.7, p_elastic=0.5, p_intensity=0.6,
                p_bg_noise=0.4, p_bias=0.2, max_disp=14):
    """scheme: 'none' | 'geo' (affine+intensity) | 'warp' (elastic family) |
    'bg' (background noise/bias) | 'full' (all)."""
    if scheme == 'none':
        return img, mask
    use = lambda g: scheme == 'full' or g == scheme
    b, c, h, w = img.shape
    dev = img.device
    r = lambda *s: torch.rand(*s, device=dev)

    on = r(b) < p_affine                                   # per-sample triggers
    ang = (r(b) * 40 - 20).deg2rad() * on * use('geo')
    sc = (1 + (r(b) * 0.2 - 0.1)) * (on & use('geo')) + 1 * ~(on & use('geo'))
    tx = ((r(b) * 32 - 16) / w) * on * use('geo')
    ty = ((r(b) * 32 - 16) / h) * on * use('geo')
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, sin, tx], 1),
                         torch.stack([-sin, cos, ty], 1)], 1)          # [B,2,3]
    grid = F.affine_grid(theta, [b, c, h, w], align_corners=False)

    if use('warp') and r(1).item() < p_elastic:                        # shared elastic field
        disp_y, disp_x = _smooth_field(b, h, w, dev), _smooth_field(b, h, w, dev)
        disp_y, disp_x = disp_y[:, 0] * max_disp, disp_x[:, 0] * max_disp   # pixels, [B, H, W]
        grid[..., 0] = grid[..., 0] + disp_x * 2 / (w - 1)
        grid[..., 1] = grid[..., 1] + disp_y * 2 / (h - 1)

    img = F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    mask = F.grid_sample(mask[:, None].float(), grid, mode='nearest',
                         padding_mode='reflection', align_corners=False).squeeze(1).round().long()

    if use('geo'):                                                       # flips
        if r(1).item() < 0.5:
            img, mask = torch.flip(img, [3]), torch.flip(mask, [2])
        if r(1).item() < 0.5:
            img, mask = torch.flip(img, [2]), torch.flip(mask, [1])

    if use('geo') and r(1).item() < p_intensity:                       # gamma / brightness / contrast
        g = (1 + (r(b, 1, 1, 1) * 0.45 - 0.225)).clamp(0.8, 1.25)
        img = img.clamp(1e-4, 1).pow(g)
        img = img * r(b, 1, 1, 1).uniform_(0.85, 1.15) + r(b, 1, 1, 1).uniform_(-0.05, 0.05)

    fg = (mask > 0).float()[:, None]                                    # background-only perturbation
    bg_w = 1 - F.max_pool2d(fg, 13, stride=1, padding=6)                # dilate gland ~6px
    bg_w = F.avg_pool2d(bg_w, 5, stride=1, padding=2)                   # smooth the ramp
    if use('bg') and r(1).item() < p_bg_noise:
        img = img + torch.randn_like(img) * 0.15 * bg_w
    if use('bg') and r(1).item() < p_bias:
        img = img * (1 + _smooth_field(1, h, w, dev, cell=40) * 0.35 * bg_w)

    return img.clamp(0, 1), mask
