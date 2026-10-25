"""Step 3 (weights): download robust lenses into ./weights.

  * MadryLab robust CNNs (Salman et al., 2020; MIT license) from
    https://huggingface.co/madrylab/robust-imagenet-models
  * Our robust ViTs and seed-baseline lenses from https://huggingface.co/jiamingzz/natural-jammers

    python download_weights.py --lens resnet50_l2_eps1        # primary lens
    python download_weights.py --lens all
"""

import argparse

from huggingface_hub import hf_hub_download

from jammers.models import LENSES, PRIMARY_LENS

MADRYLAB_REPO = 'madrylab/robust-imagenet-models'
OUR_REPO = 'jiamingzz/natural-jammers'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lens', nargs='+', default=[PRIMARY_LENS], help=f'lens names or "all": {list(LENSES)}')
    ap.add_argument('--weights-dir', default='weights')
    ap.add_argument('--our-repo', default=OUR_REPO)
    args = ap.parse_args()

    for name in (list(LENSES) if args.lens == ['all'] else args.lens):
        _, fname, source = LENSES[name]
        repo = MADRYLAB_REPO if source == 'madrylab' else args.our_repo
        path = hf_hub_download(repo_id=repo, filename=fname, local_dir=args.weights_dir)
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
