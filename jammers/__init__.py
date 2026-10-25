"""Natural Jammers: When Non-Robust Features Become Antagonistic (NeurIPS 2026)."""

import torch


def deterministic_inference():
    """Full-FP32, deterministic cuDNN inference.

    PyTorch enables TF32 convolutions by default and cuDNN autotuning picks
    batch-size-dependent algorithms; both flip the argmax of a few borderline
    images between runs. Set membership is an argmax comparison, so we disable them.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
