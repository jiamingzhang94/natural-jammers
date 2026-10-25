# Natural Jammers: When Non-Robust Features Become Antagonistic

**NeurIPS 2026 (spotlight)** · [Project page](https://jiamingzhang.netlify.app/naturaljammers) · [Weights](https://huggingface.co/jiamingzz/natural-jammers)

Jiaming Zhang, Fuyao Zhang, Pengjun Xie, Wei Yang Bryan Lim

On some **clean** ImageNet images, a standard classifier is wrong while an adversarially trained
("robust") classifier is right. We call them the **Jammed Set**. This repository reproduces the
paper's story in four steps, on the primary lens: standard ResNet-50 vs. robust ResNet-50 (ℓ2, ε = 1).

## Setup

```bash
uv venv && source .venv/bin/activate
# install the torch build that matches your CUDA driver, e.g. CUDA 12.8:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

ImageNet must be obtained from [image-net.org](https://image-net.org) in the standard `ImageFolder`
layout (`val/<wnid>/*.JPEG`). All steps use resize 256, center crop 224, and one GPU.

## 1. Find the jammed images

```bash
python download_weights.py --lens resnet50_l2_eps1
python build_partition.py --data /path/to/imagenet/val --lens resnet50_l2_eps1
```

This writes `data/partitions/resnet50_l2_eps1.csv` with one row per validation image
(`index, file, label, pred_std, pred_rob, set`), where `set` is `C` (Common: both right), `S` (Spurious:
only the standard model right), `J` (Jammed: only the robust model right) or `G` (Uninformative).
The list shipped in this repository is the one used in the paper:

| Common | Spurious | **Jammed** | Uninformative |
| --- | --- | --- | --- |
| 33,071 | 5,002 | **2,140** (4.28%) | 9,787 |

Rebuilding it gives 33,075 / 4,998 / 2,141 / 9,786: our scripts run in deterministic FP32
(`jammers.deterministic_inference`), while the paper's runs used PyTorch's default TF32
convolutions, which change the prediction of 11 of the 50,000 images. Steps 2–4 use the shipped list.

**Model weights.** Standard models are the torchvision ImageNet-1K (V1) weights. The robust
ResNet-18, ResNet-50 and WideResNet-50-2 models are the public MadryLab checkpoints
([madrylab/robust-imagenet-models](https://huggingface.co/madrylab/robust-imagenet-models));
our robust ViT-B/16 and ViT-B/32 are at [jiamingzz/natural-jammers](https://huggingface.co/jiamingzz/natural-jammers).
`download_weights.py` fetches any of them, e.g. `--lens vit_b_32_linf_eps1.0` or `--lens all`;
lens names are listed in `jammers/models.py`.

## 2. Remove the high frequencies, and the right answer comes back

```bash
python evaluate.py --data /path/to/imagenet/val --tasks lpf zoom
```

Accuracy (%) of the standard model on the sets defined on clean images (Fig. 2). Low-pass
filtering with radius r:

| r | 100 | 70 | 50 | 30 |
| --- | --- | --- | --- | --- |
| **Jammed** | 15.33 | 30.05 | **31.68** | 25.37 |
| Spurious | 87.68 | 60.68 | 43.80 | 23.85 |
| Common | 98.50 | 92.82 | 85.04 | 64.67 |

Center zoom (crop, then resize back to 224):

| Zoom | 1.05× | 1.17× | 1.33× | 1.67× | 2.00× |
| --- | --- | --- | --- | --- | --- |
| **Jammed** | 26.87 | 31.54 | 34.86 | 35.42 | 31.45 |
| Spurious | 79.67 | 74.05 | 66.55 | 55.56 | 46.90 |
| Common | 97.18 | 95.61 | 93.02 | 87.16 | 79.88 |

## 3. The jamming signal only works in its own image

```bash
python evaluate.py --data /path/to/imagenet/val --tasks graft
```

Grafting a jammer's high frequencies (r > 50) onto a random Common image passes on its wrong label
in **0.17% ± 0.11%** of the cases (five random pairings; Table 2). Putting them back into the jammer's
own image keeps the wrong label in 99.86% of the cases.

## 4. Break the jammers, and new ones appear

```bash
python evaluate.py --data /path/to/imagenet/val --tasks substitution
```

The partition is recomputed after each transform (Table 5 and Appendix F; originally |J| = 2,140 and
|S| = 5,002). The sets change a lot (low IoU) while their sizes stay close to the original:

| Transform | \|J_T\| | \|S_T\| | IoU(J) | IoU(S) |
| --- | --- | --- | --- | --- |
| crop 2% | 2,274 | 5,165 | 0.465 | 0.596 |
| crop 4% | 2,270 | 5,201 | 0.416 | 0.541 |
| crop 6% | 2,271 | 5,188 | 0.381 | 0.514 |
| crop 8% | 2,294 | 5,162 | 0.353 | 0.488 |
| LPF r = 100 | 2,291 | 4,847 | 0.650 | 0.769 |

The numbers in steps 2–4 come from a deterministic FP32 run on one A100 (about 30 minutes for all
three steps). Because the paper's runs used TF32, a few borderline images can differ: set sizes by a
few images and accuracies by at most 0.2 points. All transforms act on the 224×224 center crop in
[0, 1] pixel space (`jammers/transforms.py`).

## Citation

```bibtex
@inproceedings{zhang2026natural,
  title     = {Natural Jammers: When Non-Robust Features Become Antagonistic},
  author    = {Zhang, Jiaming and Zhang, Fuyao and Xie, Pengjun and Lim, Wei Yang Bryan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Code: MIT. The MadryLab robust checkpoints are MIT-licensed. ImageNet images are not redistributed;
the membership list contains file names only, and use of ImageNet is subject to its terms of access.
