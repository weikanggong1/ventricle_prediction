"""Train both ventricle models from scratch and evaluate on a held-out test set."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model_multimodal import UViT
from utils import BWAS_correlation, compute_loss_multi_conti

ROOT = Path(__file__).resolve().parent
TASKS = {'masked': ['EI', 'z-EI', 'BVR_AC', 'BVR_PC'], 'width': ['Width']}
MASKS = {'masked': 'all_ven.nii.gz', 'width': 'inf_ven.nii.gz'}
ARCH = dict(patch_size=256, depth=18, embed_dim=128, num_heads=8)


def read_manifest(path):
    path = Path(path).resolve()
    table = pd.read_csv(path, dtype={'eid': str})
    if not {'eid', 'image'}.issubset(table) or table.empty:
        raise ValueError('Manifest needs nonempty eid,image columns.')
    if table.eid.isna().any() or table.eid.duplicated().any():
        raise ValueError('eid must be present and unique within this manifest.')
    table['image'] = [str((path.parent / str(p)).resolve()) for p in table.image]
    if table.image.duplicated().any():
        raise ValueError('Each image path must occur only once.')
    return table


def split_table(table, seed=42, train_count=None, val_count=None):
    table = table.copy()
    if 'split' not in table:
        n = len(table)
        nt = int(n * .7) if train_count is None else train_count
        nv = max(2, int(n * .15)) if val_count is None else val_count
        if min(nt, nv, n - nt - nv) < 2:
            raise ValueError('Need >=2 subjects in each split; supply counts or more subjects.')
        order = np.random.RandomState(seed).permutation(n)
        table['split'] = 'test'
        table.loc[table.index[order[:nt]], 'split'] = 'train'
        table.loc[table.index[order[nt:nt + nv]], 'split'] = 'val'
    if not table['split'].isin(['train', 'val', 'test']).all():
        raise ValueError('split values must be train, val or test.')
    for name in ['train', 'val', 'test']:
        if (table['split'] == name).sum() < 2:
            raise ValueError(f'Need at least two subjects in split {name}.')
    return table


def regression_metrics(truth, pred, names):
    result = {}
    for i, name in enumerate(names):
        y, p = truth[:, i], pred[:, i]
        r = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and y.std() > 0 and p.std() > 0 else None
        denom = float(np.sum((y - y.mean()) ** 2))
        result[name] = {'n': len(y), 'mae': float(np.mean(np.abs(p - y))),
                        'rmse': float(np.sqrt(np.mean((p - y) ** 2))), 'pearson_r': r,
                        'r2': float(1 - np.sum((p - y) ** 2) / denom) if denom > 0 else None}
    return result


def ensure_held_out(table, bundle):
    used_ids = set(bundle['train_ids'] + bundle['validation_ids'])
    used_images = set(bundle['train_images'] + bundle['validation_images'])
    if used_ids.intersection(table.eid) or used_images.intersection(table.image):
        raise ValueError('Test subjects/images overlap training or validation data.')


def labels(table, names):
    values = []
    for name in names:
        if name in table:
            value = table[name].to_numpy(dtype=float)
        else:
            cols = [f'{name}_reader{i}' for i in range(1, 7)]
            value = table[cols].mean(axis=1).to_numpy(dtype=float)
        values.append(value)
    result = np.stack(values, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Every subject/target requires at least one finite annotation.')
    return result


class Images(Dataset):
    def __init__(self, table, mask_image, targets=None):
        self.table = table.reset_index(drop=True)
        self.mask = mask_image.get_fdata() > 0
        self.affine = mask_image.affine
        self.targets = targets
        if not self.mask.any():
            raise ValueError('Empty mask.')

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        path = self.table.iloc[index].image
        img = nib.load(path)
        if img.shape != self.mask.shape or not np.allclose(img.affine, self.affine, atol=1e-4, rtol=0):
            raise ValueError(f'Image grid differs from mask (register beforehand): {path}')
        x = np.asarray(img.dataobj, dtype=np.float32)[self.mask]
        std = x.std()
        if not np.isfinite(x).all() or not np.isfinite(std) or std <= 0:
            raise ValueError(f'Nonfinite or constant masked image: {path}')
        # Population SD reproduces the notebooks' existing NPY-cache path.
        x = torch.from_numpy((x - x.mean()) / std).unsqueeze(0)
        y = np.zeros(1, dtype=np.float32) if self.targets is None else self.targets[index].astype(np.float32)
        return x, torch.from_numpy(y), index


def build_model(nvoxel, ntask, arch):
    return UViT(input_size=[nvoxel], patch_size=[arch['patch_size']], in_chans=1,
                out_chans=1, embed_dim=arch['embed_dim'], depth=arch['depth'],
                num_heads=arch['num_heads'], mlp_ratio=4., qkv_bias=False,
                norm_layer=torch.nn.LayerNorm, use_checkpoint=False, conv=True,
                skip=False, attn_drop=0.1, proj_drop=0.1, pred_drop=0.1,
                out_class=[1] * ntask, cov_dim=0, brain_mask=True)


def mask_payload(img):
    return {'mask': torch.from_numpy((img.get_fdata() > 0).astype(np.uint8)),
            'affine': torch.from_numpy(img.affine.copy())}


def load_bundle(path, device):
    b = torch.load(path, map_location='cpu', weights_only=True)
    if b.get('format_version') != 1 or b.get('task') not in TASKS:
        raise ValueError(f'Unsupported checkpoint bundle: {path}')
    if b['targets'] != TASKS[b['task']]:
        raise ValueError('Checkpoint target order does not match task.')
    mask = b['mask'].numpy().astype(bool)
    order = b['shift_index'].numpy()
    if order.ndim != 1 or len(order) != mask.sum() or not np.array_equal(np.sort(order), np.arange(mask.sum())):
        raise ValueError('Invalid voxel permutation in checkpoint.')
    if not torch.isfinite(b['train_std']).all() or (b['train_std'] <= 0).any():
        raise ValueError('Invalid label normalization.')
    model = build_model(len(order), len(b['targets']), b['architecture']).to(device)
    model.load_state_dict(b['state_dict'], strict=True)
    model.eval()
    img = nib.Nifti1Image(mask.astype(np.uint8), b['affine'].numpy())
    return b, model, img


def predict_loader(model, loader, order, mean, std, device):
    model.eval()
    result = np.zeros((len(loader.dataset), len(mean)), dtype=np.float32)
    with torch.inference_mode():
        for x, _, idx in loader:
            # Original forward ignores pos_init (passed as cov_feature, cov_dim=0).
            pred = torch.cat(model([x.to(device)[:, :, order]]), dim=1)
            result[idx.numpy()] = pred.cpu().numpy() * std + mean
    if not np.isfinite(result).all():
        raise ValueError('Model produced nonfinite predictions.')
    return result


def predict(args):
    table = read_manifest(args.manifest)
    evaluating = args.command == 'evaluate'
    if evaluating and 'split' in table:
        table = table.loc[table['split'] == 'test'].reset_index(drop=True)
    if table.empty:
        raise ValueError('No subjects to predict/evaluate.')
    result = table[['eid']].copy()
    provenance, metrics = [], {}
    for task in (TASKS if args.task == 'all' else [args.task]):
        path = Path(args.checkpoints) / f'{task}.pt'
        if not path.exists():
            raise FileNotFoundError(f'{path}: train {task} first. No predictions written.')
        b, model, img = load_bundle(path, args.device)
        if b['task'] != task:
            raise ValueError('Checkpoint filename/task mismatch.')
        if evaluating:
            ensure_held_out(table, b)
            truth = labels(table, b['targets'])
        loader = DataLoader(Images(table, img), batch_size=args.batch_size, num_workers=args.workers)
        pred = predict_loader(model, loader, b['shift_index'].to(args.device),
                              b['train_mean'].numpy(), b['train_std'].numpy(), args.device)
        for i, name in enumerate(b['targets']):
            result[name] = pred[:, i]
            if evaluating:
                result[name + '_true'] = truth[:, i]
        if evaluating:
            metrics.update(regression_metrics(truth, pred, b['targets']))
        provenance.append({'task': task, 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'source': b.get('source', 'trained'), 'synthetic': b.get('synthetic', False)})
        del model
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2))
    if evaluating:
        output.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(f'Saved {len(result)} rows to {output}', flush=True)


def train(args):
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError('Need 0 <= warmup-epochs < epochs.')
    table = read_manifest(args.manifest)
    if 'split' not in table:
        raise ValueError('Training requires split column; use run to generate a split automatically.')
    table = split_table(table)
    ti = np.flatnonzero(table['split'] == 'train')
    vi = np.flatnonzero(table['split'] == 'val')
    for task in (TASKS if args.task == 'all' else [args.task]):
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Test annotations do not participate in training or model selection.
        y = np.zeros((len(table), len(TASKS[task])), dtype=np.float64)
        y[ti] = labels(table.iloc[ti], TASKS[task])
        y[vi] = labels(table.iloc[vi], TASKS[task])
        mean, std = y[ti].mean(axis=0).astype('float32'), y[ti].std(axis=0).astype('float32')
        if (std <= 0).any():
            raise ValueError('Constant training target.')
        img = nib.load(Path(args.masks) / MASKS[task])
        ds = Images(table.iloc[ti], img, y[ti])
        fixed = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        # Match BWAS_correlation and the final shift_index assignment in cell 6.
        features = np.zeros((len(ds), int(ds.mask.sum())), dtype=np.float64)
        for x, _, idx in fixed:
            features[idx.numpy()] = x[:, 0].numpy()
        r = np.abs(BWAS_correlation(features, y[ti].astype(np.float32).astype(np.float64))).mean(axis=1)
        if not np.isfinite(r).all():
            raise ValueError('Undefined voxel/target correlation; check constant training voxels.')
        order = np.argsort(r); del features
        train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(Images(table.iloc[vi], img, y[vi]), batch_size=args.batch_size, num_workers=args.workers)
        model = build_model(len(order), len(TASKS[task]), ARCH).to(args.device)
        loss_fn = compute_loss_multi_conti(len(TASKS[task])).to(args.device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()),
                                      lr=args.lr, weight_decay=.01, betas=(.9, .999))
        total, warmup = args.epochs * len(train_loader), args.warmup_epochs * len(train_loader)
        def lr_factor(step):
            step = step % total
            amplitude = step / warmup if step < warmup else (1 + math.cos(math.pi * (step - warmup) / (total - warmup))) / 2
            return (1e-6 + (args.lr - 1e-6) * amplitude) / args.lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        mt, st = torch.from_numpy(mean).to(args.device), torch.from_numpy(std).to(args.device)
        ot = torch.from_numpy(order).to(args.device)
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        history, best = [], -float('inf')
        for epoch in range(args.epochs):
            model.train(); losses = []
            for x, target, _ in train_loader:
                optimizer.zero_grad()
                x = torch.nn.functional.dropout(x.to(args.device)[:, :, ot], p=.25, training=True)
                pred = torch.cat(model([x]), dim=1)
                loss = loss_fn(pred, (target.to(args.device) - mt) / st)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); scheduler.step(); losses.append(loss.item())
            pred = predict_loader(model, val_loader, ot, mean, std, args.device)
            vm = regression_metrics(y[vi], pred, TASKS[task])
            corr = [vm[name]['pearson_r'] for name in TASKS[task]]
            if any(r is None for r in corr):
                raise ValueError('Validation correlation undefined; check constant targets/predictions.')
            score = float(np.mean(corr))
            record = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'validation_r': corr}
            history.append(record); print(task, record, flush=True)
            if np.isfinite(score) and score > best:
                best = score
                b = dict(format_version=1, task=task, targets=TASKS[task], architecture=ARCH,
                         state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                         shift_index=torch.from_numpy(order), train_mean=torch.from_numpy(mean),
                         train_std=torch.from_numpy(std), source='unified training', seed=args.seed,
                         train_ids=table.iloc[ti].eid.tolist(), validation_ids=table.iloc[vi].eid.tolist(),
                         train_images=table.iloc[ti].image.tolist(), validation_images=table.iloc[vi].image.tolist(),
                         epoch=epoch + 1, validation_r=corr, **mask_payload(img))
                torch.save(b, output / f'{task}.pt')
            (output / f'{task}_history.json').write_text(json.dumps(history, indent=2))
        if best == -float('inf'):
            raise RuntimeError('No finite validation correlation; no checkpoint selected.')


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / 'checkpoints' / f'{task}.pt').exists() for task in TASKS):
        raise ValueError('Run directory already contains trained models; choose a new output directory.')
    table = split_table(read_manifest(args.manifest), args.seed, args.train_count, args.val_count)
    manifest = out / 'split_manifest.csv'
    table.to_csv(manifest, index=False)
    training = argparse.Namespace(**vars(args))
    training.manifest = str(manifest)
    training.output = str(out / 'checkpoints')
    training.command = 'train'
    train(training)
    testing = argparse.Namespace(**vars(args))
    testing.manifest = str(manifest)
    testing.command = 'evaluate'
    testing.checkpoints = training.output
    testing.output = str(out / 'test_predictions.csv')
    predict(testing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for name in ['run', 'train', 'evaluate', 'predict']:
        s = commands.add_parser(name)
        s.add_argument('--manifest', required=True)
        s.add_argument('--task', choices=['all', *TASKS], default='all')
        s.add_argument('--device', default='cpu')
        s.add_argument('--batch-size', type=int, default=4)
        s.add_argument('--workers', type=int, default=0)
        s.add_argument('--output', required=True)
        if name in ['predict', 'evaluate']:
            s.add_argument('--checkpoints', required=True)
        else:
            s.add_argument('--masks', default=str(ROOT / 'assets'))
            s.add_argument('--epochs', type=int, default=50)
            s.add_argument('--warmup-epochs', type=int, default=5)
            s.add_argument('--lr', type=float, default=1e-4)
            s.add_argument('--seed', type=int, default=42)
            if name == 'run':
                s.add_argument('--train-count', type=int)
                s.add_argument('--val-count', type=int)
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        p.error('batch-size >=1 and workers >=0 required')
    {'run': run, 'train': train, 'evaluate': predict, 'predict': predict}[args.command](args)


if __name__ == '__main__':
    main()
