"""Step 1 (data): build the quad-set partition of ImageNet-val for one or more lenses.

For each lens (f_std, f_rob) every validation image is assigned to
Common / Spurious / Jammed / Uninformative from the two models' clean predictions,
and the membership list is written to <out-dir>/<lens>.csv.

    python build_partition.py --data /path/to/imagenet/val --lens resnet50_l2_eps1
    python build_partition.py --data /path/to/imagenet/val --lens all
"""

import argparse
import json
import os

import torch

from jammers import deterministic_inference
from jammers.data import imagenet_folder, make_loader
from jammers.models import LENSES, PRIMARY_LENS, load_lens
from jammers.partition import SETS, SET_NAMES, assign, predict, save_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='ImageNet validation folder (ImageFolder layout)')
    ap.add_argument('--lens', nargs='+', default=[PRIMARY_LENS], help=f'lens names or "all": {list(LENSES)}')
    ap.add_argument('--weights-dir', default='weights')
    ap.add_argument('--out-dir', default='data/partitions')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    lenses = list(LENSES) if args.lens == ['all'] else args.lens
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deterministic_inference()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = imagenet_folder(args.data)
    files = [os.path.relpath(p, args.data) for p, _ in dataset.samples]
    loader = make_loader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    summary = {}
    for name in lenses:
        f_std, f_rob = load_lens(name, args.weights_dir)
        labels, preds = predict({'std': f_std.to(device), 'rob': f_rob.to(device)}, loader,
                                {'clean': lambda x: x}, device, desc=name)
        sets = assign(preds[('std', 'clean')], preds[('rob', 'clean')], labels)
        save_csv(os.path.join(args.out_dir, f'{name}.csv'), files, labels,
                 preds[('std', 'clean')], preds[('rob', 'clean')], sets)

        n = len(labels)
        row = {SET_NAMES[k]: int((sets == k).sum()) for k in SETS}
        row['std_acc'] = round(100 * float((preds[('std', 'clean')] == labels).mean()), 2)
        row['rob_acc'] = round(100 * float((preds[('rob', 'clean')] == labels).mean()), 2)
        summary[name] = row
        print(f'{name}: ' + ', '.join(f'{SET_NAMES[k]} {row[SET_NAMES[k]]:,} ({100 * row[SET_NAMES[k]] / n:.2f}%)'
                                     for k in SETS) + f' | std acc {row["std_acc"]}% | rob acc {row["rob_acc"]}%')
        del f_std, f_rob
        torch.cuda.empty_cache()

    path = os.path.join(args.out_dir, 'summary.json')
    merged = {}
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
    merged.update(summary)
    with open(path, 'w') as f:
        json.dump(merged, f)
        f.write('\n')


if __name__ == '__main__':
    main()
