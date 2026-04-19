"""GPU-side sequence augmentation. All ops work on (B, T, C, H, W) tensors.
One transform per batch item applied to all T frames."""

import torch
import torch.nn.functional as F


_MEAN = None
_INV_STD = None


def _init_constants(device, dtype):
    """Lazily build normalization constants in the target device/dtype.
    We precompute inv_std so the final step is mul_ rather than div_."""
    global _MEAN, _INV_STD
    if _MEAN is None or _MEAN.device != device or _MEAN.dtype != dtype:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 1, 3, 1, 1) * 255.0
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 1, 3, 1, 1) * 255.0
        _MEAN = mean
        _INV_STD = 1.0 / std


def color_jitter_(x, brightness=0.20, contrast=0.20, saturation=0.15):
    """In-place color jitter. Brightness and contrast are algebraically fused into
    a single affine step. x: (B, T, C, H, W) float tensor."""
    B = x.shape[0]
    device, dtype = x.device, x.dtype

    b = 1.0 + (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * brightness
    c = 1.0 + (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * contrast
    s = 1.0 + (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * saturation
    b = b.view(B, 1, 1, 1, 1)
    c = c.view(B, 1, 1, 1, 1)
    s = s.view(B, 1, 1, 1, 1)

    # Fuse brightness + contrast.
    # Original: x1 = x*b; mean1 = mean(x1); x2 = (x1 - mean1)*c + mean1
    # Identity: x2 = x * (b*c) + b * mean(x) * (1 - c)
    # One mean kernel + one chained mul+add instead of mul, mean, sub, mul, add.
    mean_x = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    x.mul_(b * c).add_(b * mean_x * (1.0 - c))

    # Saturation: build grayscale as (B,T,1,H,W), then x = x*s + gray*(1-s)
    gray = x[:, :, 0:1].mul(0.299)
    gray.add_(x[:, :, 1:2], alpha=0.587)
    gray.add_(x[:, :, 2:3], alpha=0.114)
    x.mul_(s).add_(gray * (1.0 - s))

    x.clamp_(0, 255)
    return x


def random_affine_flip(x, degrees=5.0, shear=5.0, flip_p=0.5):
    """Rotate + x-shear + horizontal flip all folded into a single grid_sample.
    x: (B, T, C, H, W) float tensor."""
    B, T, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    angle_deg = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * degrees
    shear_deg = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * shear

    a = angle_deg * (torch.pi / 180.0)
    sh = torch.tan(shear_deg * (torch.pi / 180.0))
    cos_a = torch.cos(a)
    sin_a = torch.sin(a)

    # +1 or -1 per item; -1 negates the x-axis row of theta so grid_sample flips the image
    flip_sign = torch.ones(B, device=device, dtype=dtype)
    flip_sign[torch.rand(B, device=device) < flip_p] = -1.0

    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = flip_sign * cos_a
    theta[:, 0, 1] = flip_sign * (cos_a * sh - sin_a)
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = sin_a * sh + cos_a

    # broadcast per-item theta across T frames
    theta = theta.unsqueeze(1).expand(B, T, 2, 3).reshape(B * T, 2, 3)

    x_flat = x.reshape(B * T, C, H, W)
    grid = F.affine_grid(theta, x_flat.shape, align_corners=False)
    out = F.grid_sample(x_flat, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    del grid, x_flat

    return out.reshape(B, T, C, H, W)


def augment_and_normalize(images_uint8, train=True, dtype=torch.float16):
    """Full GPU pipeline. Input: (B, T, C, H, W) uint8. Output: normalized float."""
    _init_constants(images_uint8.device, dtype)
    x = images_uint8.to(dtype=dtype)

    if train:
        x = color_jitter_(x, brightness=0.20, contrast=0.20, saturation=0.15)
        x = random_affine_flip(x, degrees=5.0, shear=5.0, flip_p=0.5)

    # normalize as the final op, mul by inv_std beats div_
    x.sub_(_MEAN).mul_(_INV_STD)
    return x
