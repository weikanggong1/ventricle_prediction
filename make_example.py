"""Create synthetic images/labels only; no checkpoint is needed or generated."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='example')
    args = p.parse_args()
    out = Path(args.output)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(exist_ok=True)
    rng = np.random.RandomState(20260918)
    shape = (12, 12, 12)
    grid = np.indices(shape)
    mask = ((grid - 5.5) ** 2).sum(axis=0) < 25
    for name, region in [('all_ven', mask), ('inf_ven', mask & (grid[2] < 5))]:
        nib.save(nib.Nifti1Image(region.astype('uint8'), np.eye(4)), out / 'masks' / f'{name}.nii.gz')
    rows = []
    for i in range(20):
        data = rng.normal(size=shape).astype('float32') + i * grid[0].astype('float32') / 12
        nib.save(nib.Nifti1Image(data, np.eye(4)), out / 'images' / f'sim_{i:03}.nii.gz')
        row = {'eid': f'sim_{i:03}', 'image': f'images/sim_{i:03}.nii.gz'}
        row.update({n: float(i + j + rng.normal(scale=.2)) for j, n in enumerate(['EI', 'z-EI', 'BVR_AC', 'BVR_PC', 'Width'])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'subjects.csv', index=False)
    (out / 'README.txt').write_text('Synthetic random arrays and labels, seed=20260918. '
        'No original subject data, anatomical simulation, or pretrained weights. '
        'For software execution tests only; not for measuring prediction accuracy.\n')
    print(out / 'subjects.csv')


if __name__ == '__main__':
    main()
