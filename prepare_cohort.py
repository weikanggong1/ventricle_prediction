"""Build a train/val/test manifest from the UKB or ADNI annotation tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import split_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', choices=['ukb', 'adni'], required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    ukb = args.cohort == 'ukb'
    idcol = 'eid' if ukb else 'PTID'
    annotation = pd.read_csv(args.labels, sep=',' if ukb else '\t', dtype={idcol: str})
    if annotation[idcol].isna().any() or annotation[idcol].duplicated().any():
        raise ValueError('Annotation subject IDs must be present and unique.')
    names = ['EI', 'z-EI', 'BVR_AC', 'BVR_PC', 'Width']
    filename = 'T1_brain_linear_1mm.nii.gz' if ukb else 'T1_MNI.nii.gz'
    rows = []; missing = invalid = 0
    for _, row in annotation.iterrows():
        image = Path(args.image_root) / row[idcol] / filename
        if not image.is_file():
            missing += 1; continue
        y = {name: float(row[[f'{name}_reader{i}' for i in range(1, 7)] if ukb
                             else [f'{name}_1', f'{name}_2']].mean()) for name in names}
        if not np.isfinite(list(y.values())).all():
            invalid += 1; continue
        rows.append(dict(eid=row[idcol], image=str(image.resolve()), **y))
    table = split_table(pd.DataFrame(rows), args.seed)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    audit = dict(cohort=args.cohort, source_rows=len(annotation), available_rows=len(table),
                 missing_images=missing, invalid_targets=invalid, seed=args.seed,
                 split=table['split'].value_counts().to_dict(),
                 label_file_sha256=hashlib.sha256(Path(args.labels).read_bytes()).hexdigest())
    out.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
