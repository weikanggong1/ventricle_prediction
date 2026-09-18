"""Exercise random initialization, training, selection, reload and held-out testing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='smoke_run')
    p.add_argument('--model-size', choices=['original', 'small'], default='original')
    args = p.parse_args()
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError('Choose a new smoke-test output directory.')
    out.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    def call(script, *arguments):
        subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], check=True, env=env)
    call('make_example.py', '--output', out / 'data')
    call('pipeline.py', 'run', '--manifest', out / 'data/subjects.csv',
         '--masks', out / 'data/masks', '--output', out / 'experiment',
         '--train-count', 12, '--val-count', 4, '--epochs', 2, '--warmup-epochs', 1,
         '--model-size', args.model_size)
    exp = out / 'experiment'
    split = pd.read_csv(exp / 'split_manifest.csv')
    assert split['split'].value_counts().to_dict() == {'train': 12, 'val': 4, 'test': 4}
    metrics = json.loads((exp / 'test_predictions.metrics.json').read_text())
    assert set(metrics) == {'EI', 'z-EI', 'BVR_AC', 'BVR_PC', 'Width'}
    assert all(m['n'] == 4 and np.isfinite(m['mae']) and np.isfinite(m['rmse']) for m in metrics.values())
    for task in ['masked', 'width']:
        b = torch.load(exp / 'checkpoints' / f'{task}.pt', map_location='cpu', weights_only=True)
        assert len(b['train_ids']) == 12 and len(b['validation_ids']) == 4
        assert not set(split.loc[split['split'] == 'test', 'eid']).intersection(b['train_ids'] + b['validation_ids'])
        assert b['epoch'] in [1, 2]
    # Fresh process reloads the chosen models and produces the same held-out results.
    call('pipeline.py', 'evaluate', '--manifest', exp / 'split_manifest.csv',
         '--checkpoints', exp / 'checkpoints', '--output', out / 'repeat_test.csv')
    first = pd.read_csv(exp / 'test_predictions.csv')
    repeated = pd.read_csv(out / 'repeat_test.csv')
    pd.testing.assert_frame_equal(first, repeated, check_exact=True)
    train = split.loc[split['split'] == 'train'].drop(columns='split')
    train.to_csv(out / 'leaking_test.csv', index=False)
    bad = subprocess.run([sys.executable, str(ROOT / 'pipeline.py'), 'evaluate',
                          '--manifest', str(out / 'leaking_test.csv'), '--checkpoints', str(exp / 'checkpoints'),
                          '--output', str(out / 'should_not_exist.csv')], capture_output=True, text=True, env=env)
    assert bad.returncode != 0 and 'overlap' in bad.stderr and not (out / 'should_not_exist.csv').exists()
    report = dict(status='passed', initialization='random; no existing checkpoint read for training',
                  models=['masked', 'width'], epochs_per_model=2, subjects={'train':12, 'val':4, 'test':4},
                  architecture=b['architecture'],
                  best_model_reload_identical=True, train_test_overlap_rejected=True,
                  test_metrics=list(metrics), device='cpu', synthetic_data=True)
    (out / 'smoke_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
