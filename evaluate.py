"""Step 4 (evaluation): the core experiments of the paper on a quad-set partition.

Tasks (default: all):
  lpf           Cleansing Paradox, Fig. 2(a): f_std accuracy on fixed C/J/S under low-pass filtering
  zoom          Sensitivity Inversion, Fig. 2(b): same under center zoom
  graft         Frequency grafting, Table 2: HF of each jammer onto a random Common host (5 seeds)
                plus the identity-reconstruction control
  substitution  Substitution Paradox, Table 5 and Appendix F: partition recomputed under small transforms

    python evaluate.py --data /path/to/imagenet/val --partition data/partitions/resnet50_l2_eps1.csv
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from jammers import deterministic_inference
from jammers import transforms as T
from jammers.data import imagenet_folder, make_loader, normalize
from jammers.models import LENSES, PRIMARY_LENS, load_lens
from jammers.partition import assign, iou, load_csv, predict

LPF_RADII = [112, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
ZOOM_RATIOS = [0.95, 0.9, 0.857, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
SUBSTITUTION = ['crop2', 'crop4', 'crop6', 'crop8', 'lpf110', 'lpf105', 'lpf100']
GRAFT_SEEDS = [101, 102, 103, 104, 105]


def fixed_membership(part, preds, names):
    """Protocol P1: accuracy (%) of f_std on the clean-defined sets C, J, S."""
    out = {}
    for name in names:
        correct = preds[('std', name)] == part['label_sub']
        out[name] = {s: round(100 * float(correct[part['set_sub'] == s].mean()), 2) for s in 'CJS'}
    return out


class PairSet(Dataset):
    def __init__(self, base, pairs):
        self.base, self.pairs = base, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        h, d = self.pairs[i]
        host = self.base[h][0]
        return host, (host if h == d else self.base[d][0])


@torch.no_grad()
def run_graft(dataset, part, f_std, device, args):
    j_idx = np.flatnonzero(part['set'] == 'J')
    c_idx = np.flatnonzero(part['set'] == 'C')
    donor_wrong = part['pred_std'][j_idx]

    def rates(pairs, host_labels, identity=False):
        loader = DataLoader(PairSet(dataset, pairs), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True)
        preds = []
        for host, donor in tqdm(loader, desc='graft'):
            host, donor = host.to(device), donor.to(device)
            x = T.lpf(host, 1e9) if identity else T.graft(host, donor, 50)
            preds.append(f_std(normalize(x)).argmax(1).cpu().numpy())
        preds = np.concatenate(preds)
        return 100 * float((preds == donor_wrong).mean()), 100 * float((preds == host_labels).mean())

    donor, _ = rates([(j, j) for j in j_idx], part['label'][j_idx], identity=True)
    out = {'identity_reconstruction_donor_label': round(donor, 2), 'random_host': []}
    for seed in GRAFT_SEEDS:  # five random pairings, as in the paper (Appendix D.3)
        hosts = c_idx[np.random.RandomState(seed).randint(0, len(c_idx), size=len(j_idx))]
        d, h = rates(list(zip(hosts, j_idx)), part['label'][hosts])
        out['random_host'].append({'seed': seed, 'donor_label': round(d, 2), 'host_label': round(h, 2)})
    d = [r['donor_label'] for r in out['random_host']]
    out['random_host_donor_label_mean_std'] = [round(float(np.mean(d)), 2), round(float(np.std(d)), 2)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='ImageNet validation folder (ImageFolder layout)')
    ap.add_argument('--lens', default=PRIMARY_LENS, choices=list(LENSES))
    ap.add_argument('--partition', help='CSV from build_partition.py (default: data/partitions/<lens>.csv)')
    ap.add_argument('--weights-dir', default='weights')
    ap.add_argument('--tasks', nargs='+', default=['lpf', 'zoom', 'graft', 'substitution'])
    ap.add_argument('--out', default=None, help='write results as JSON')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deterministic_inference()
    part = load_csv(args.partition or f'data/partitions/{args.lens}.csv')
    dataset = imagenet_folder(args.data)
    assert len(dataset) == len(part['label']), 'partition does not match the dataset'
    f_std, f_rob = (m.to(device) for m in load_lens(args.lens, args.weights_dir))
    results = {'lens': args.lens, 'set_sizes': {s: int((part['set'] == s).sum()) for s in 'CSJG'}}
    print('set sizes:', results['set_sizes'])

    # Protocol P1 tasks: f_std on the fixed sets C, J, S (one pass over C u J u S).
    p1 = {}
    if 'lpf' in args.tasks:
        p1.update({f'lpf{r}': T.build(f'lpf{r}') for r in LPF_RADII})
    if 'zoom' in args.tasks:
        p1.update({f'zoom{int(224 * a)}': T.build(f'zoom{int(224 * a)}') for a in ZOOM_RATIOS})
    if p1:
        keep = np.flatnonzero(part['set'] != 'G')
        part['set_sub'], part['label_sub'] = part['set'][keep], part['label'][keep]
        _, preds = predict({'std': f_std}, make_loader(dataset, keep, args.batch_size, args.workers),
                           p1, device, desc='fixed-membership')
        if 'lpf' in args.tasks:
            results['lpf'] = fixed_membership(part, preds, [f'lpf{r}' for r in LPF_RADII])
        if 'zoom' in args.tasks:
            results['zoom'] = fixed_membership(part, preds, [f'zoom{int(224 * a)}' for a in ZOOM_RATIOS])

    if 'graft' in args.tasks:
        results['graft'] = run_graft(dataset, part, f_std, device, args)

    # Protocol P2: both models on every image under each transform, partition recomputed.
    if 'substitution' in args.tasks:
        labels, preds = predict({'std': f_std, 'rob': f_rob}, make_loader(dataset, None, args.batch_size, args.workers),
                                {t: T.build(t) for t in SUBSTITUTION}, device, desc='substitution')
        orig_J, orig_S = np.flatnonzero(part['set'] == 'J'), np.flatnonzero(part['set'] == 'S')
        results['substitution'] = {}
        for t in SUBSTITUTION:
            sets = assign(preds[('std', t)], preds[('rob', t)], labels)
            new_J, new_S = np.flatnonzero(sets == 'J'), np.flatnonzero(sets == 'S')
            results['substitution'][t] = {
                '|J_T|': len(new_J), '|S_T|': len(new_S),
                'IoU_J': round(iou(orig_J, new_J), 3), 'IoU_S': round(iou(orig_S, new_S), 3)}

    print(json.dumps(results, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
