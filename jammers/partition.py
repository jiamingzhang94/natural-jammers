"""Quad-set partition (Section 2 of the paper) and helpers.

    Common      C: f_std correct, f_rob correct
    Spurious    S: f_std correct, f_rob wrong
    Jammed      J: f_std wrong,   f_rob correct
    Uninformative G: f_std wrong, f_rob wrong
"""

import csv

import numpy as np
import torch
from tqdm import tqdm

from .data import normalize

SETS = ('C', 'S', 'J', 'G')
SET_NAMES = {'C': 'Common', 'S': 'Spurious', 'J': 'Jammed', 'G': 'Uninformative'}


def assign(pred_std, pred_rob, labels):
    """Vectorized set assignment; returns an array of 'C'/'S'/'J'/'G'."""
    pred_std, pred_rob, labels = map(np.asarray, (pred_std, pred_rob, labels))
    s_ok, r_ok = pred_std == labels, pred_rob == labels
    return np.select([s_ok & r_ok, s_ok & ~r_ok, ~s_ok & r_ok], ['C', 'S', 'J'], default='G')


def iou(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


@torch.no_grad()
def predict(models, loader, transforms, device, desc='predict'):
    """Runs every model on every transform of every batch in a single data pass.

    models: dict name -> nn.Module; transforms: dict name -> callable on [0,1] images.
    Returns (labels, preds) with preds[(model_name, transform_name)] = int64 array.
    """
    preds = {(m, t): [] for m in models for t in transforms}
    labels = []
    for x, y in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        labels.append(y.numpy())
        for t, fn in transforms.items():
            xt = normalize(fn(x))
            for m, model in models.items():
                preds[(m, t)].append(model(xt).argmax(1).cpu().numpy())
    return np.concatenate(labels), {k: np.concatenate(v) for k, v in preds.items()}


def save_csv(path, files, labels, pred_std, pred_rob, sets):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'file', 'label', 'pred_std', 'pred_rob', 'set'])
        for i, row in enumerate(zip(files, labels, pred_std, pred_rob, sets)):
            w.writerow([i, *row])


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        'file': [r['file'] for r in rows],
        'label': np.array([int(r['label']) for r in rows]),
        'pred_std': np.array([int(r['pred_std']) for r in rows]),
        'pred_rob': np.array([int(r['pred_rob']) for r in rows]),
        'set': np.array([r['set'] for r in rows]),
    }
