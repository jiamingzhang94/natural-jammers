"""Standard models and robust lenses.

Robust CNN lenses are the public MadryLab checkpoints (Salman et al., 2020);
the ViT lenses are released with this repository.
"""

import os

import torch
from torchvision import models

# name -> (architecture, file name, source)
LENSES = {
    # MadryLab release (https://huggingface.co/madrylab/robust-imagenet-models)
    **{f'resnet18_{t}': ('resnet18', f'resnet18_{t}.ckpt', 'madrylab')
       for t in ['l2_eps0.5', 'l2_eps1', 'l2_eps3', 'linf_eps1.0', 'linf_eps4.0']},
    **{f'resnet50_{t}': ('resnet50', f'resnet50_{t}.ckpt', 'madrylab')
       for t in ['l2_eps0.5', 'l2_eps1', 'l2_eps3', 'linf_eps1.0', 'linf_eps4.0']},
    **{f'wide_resnet50_2_{t}': ('wide_resnet50_2', f'wide_resnet50_2_{t}.ckpt', 'madrylab')
       for t in ['l2_eps0.5', 'l2_eps1', 'l2_eps3', 'linf_eps1.0', 'linf_eps4.0']},
    # released with this repository
    'vit_b_16_linf_eps1.0': ('vit_b_16', 'vit_b_16_linf_eps1.0.safetensors', 'ours'),
    'vit_b_32_linf_eps1.0': ('vit_b_32', 'vit_b_32_linf_eps1.0.safetensors', 'ours'),
}

PRIMARY_LENS = 'resnet50_l2_eps1'


def standard_model(arch):
    """torchvision ImageNet-1K (V1) weights, as used throughout the paper."""
    return getattr(models, arch)(weights='IMAGENET1K_V1')


def _state_dict_from_file(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
    # MadryLab checkpoints wrap the classifier as `module.model.*` (plus an
    # attacker copy and a normalizer); keep only the classifier weights.
    if any(k.startswith('module.model.') for k in sd):
        sd = {k[len('module.model.'):]: v for k, v in sd.items() if k.startswith('module.model.')}
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def robust_model(arch, path):
    model = getattr(models, arch)(weights=None)
    model.load_state_dict(_state_dict_from_file(path), strict=True)
    return model


def load_lens(name, weights_dir):
    """Returns (f_std, f_rob) for a lens name in LENSES, both in eval mode."""
    arch, fname, _ = LENSES[name]
    path = os.path.join(weights_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found; run `python download_weights.py --lens {name}`')
    return standard_model(arch).eval(), robust_model(arch, path).eval()
