"""Transform operators (Appendix A.3 of the paper).

All operators take a batch of images in [0, 1] with shape (B, 3, 224, 224).
"""

import torch
import torch.nn.functional as F


def _disk(h, w, radius, device):
    yy, xx = torch.meshgrid(torch.arange(h, device=device) - h // 2,
                            torch.arange(w, device=device) - w // 2, indexing='ij')
    return ((xx ** 2 + yy ** 2).float().sqrt() <= radius).float()


def _fft(x):
    return torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))


def _ifft(f):
    return torch.fft.ifft2(torch.fft.ifftshift(f, dim=(-2, -1)), norm='ortho').real.clamp(0, 1)


def lpf(x, radius):
    """Circular hard low-pass filter centered at DC (radius in frequency-grid pixels)."""
    return _ifft(_fft(x) * _disk(x.shape[-2], x.shape[-1], radius, x.device))


def graft(host, donor, radius=50):
    """Low frequencies of `host` + high frequencies of `donor` (frequency grafting)."""
    m = _disk(host.shape[-2], host.shape[-1], radius, host.device)
    return _ifft(m * _fft(host) + (1 - m) * _fft(donor))


def center_crop_resize(x, crop, antialias=False):
    """Center crop of `crop` pixels, bilinearly resized back (align_corners=False)."""
    h, w = x.shape[-2:]
    top, left = int(round((h - crop) / 2)), int(round((w - crop) / 2))  # torchvision center_crop convention
    c = x[..., top:top + crop, left:left + crop]
    return F.interpolate(c, size=(h, w), mode='bilinear', align_corners=False,
                         antialias=antialias).clamp(0, 1)


def zoom(x, crop):
    """Center zoom: crop `crop` pixels and resize back, e.g. crop=192 is a 224/192 = 1.17x zoom."""
    return center_crop_resize(x, crop)


def edge_crop(x, keep):
    """Remove the border so that a fraction `keep` of the side length remains (e.g. 0.98),
    then resize back with antialiased bilinear interpolation."""
    return center_crop_resize(x, int(round(x.shape[-1] * keep)), antialias=True)


def build(name):
    """Parses transform names used on the command line.

    none | lpf<r> (e.g. lpf50) | crop<p> (remove p% of the side, e.g. crop2)
    | zoom<c> (center crop of c pixels resized to 224, e.g. zoom192 = 1.17x)
    """
    if name == 'none':
        return lambda x: x
    if name.startswith('lpf'):
        return lambda x, r=float(name[3:]): lpf(x, r)
    if name.startswith('crop'):
        return lambda x, p=float(name[4:]): edge_crop(x, 1 - p / 100)
    if name.startswith('zoom'):
        return lambda x, c=int(name[4:]): zoom(x, c)
    raise ValueError(f'unknown transform: {name}')
