"""Natural Jammers: When Non-Robust Features Become Antagonistic (NeurIPS 2026)."""

import torch


def deterministic_inference():
    """Full-FP32, deterministic cuDNN inference, so that set membership is identical across runs."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
