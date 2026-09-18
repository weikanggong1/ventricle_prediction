"""Export an inference checkpoint without subject IDs, image paths or annotations."""
import argparse
from pathlib import Path
import torch

FIELDS = ('format_version', 'task', 'targets', 'architecture', 'state_dict', 'shift_index',
          'train_mean', 'train_std', 'mask', 'affine', 'seed', 'parameter_count',
          'training_config', 'epoch', 'validation_r')


def export_bundle(source, output, cohort):
    private = torch.load(source, map_location='cpu', weights_only=True)
    public = {key: private[key] for key in FIELDS if key in private}
    public.update(cohort=cohort, inference_only=True,
                  source=f'{cohort.upper()} pretrained model; trained from random initialization on CPU',
                  split_counts={'train': len(private['train_ids']), 'val': len(private['validation_ids'])})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public, output)
    return public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cohort', choices=['ukb', 'adni'], required=True)
    args = p.parse_args()
    export_bundle(args.checkpoint, args.output, args.cohort)
    print(f'Saved public inference bundle: {args.output}')


if __name__ == '__main__':
    main()
