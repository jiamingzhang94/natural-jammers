"""ImageNet data pipeline shared by all scripts.

Every transform in the paper acts on the standard 224x224 center crop in [0, 1]
pixel space; ImageNet normalization is applied immediately before the model.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def normalize(x):
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def imagenet_folder(root):
    """ImageNet split in torchvision ImageFolder layout (one sub-folder per wnid)."""
    return datasets.ImageFolder(root, transform=eval_transform())


def make_loader(dataset, indices=None, batch_size=256, num_workers=16):
    if indices is not None:
        dataset = Subset(dataset, list(indices))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
