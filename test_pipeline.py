"""Input-contract and checkpoint integrity checks; no original images required."""
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from pipeline import (Images, labels, read_manifest, build_model, mask_payload,
                      load_bundle, TASKS, split_table, regression_metrics, ensure_held_out)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype='uint8'), np.eye(4))
        self.table = pd.DataFrame({'eid': ['sim'], 'image': [str(self.root / 'image.nii.gz')]})

    def save(self, data, affine=None):
        nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), self.table.image[0])

    def test_normalization_and_grid(self):
        self.save(np.arange(64, dtype='float32').reshape(4, 4, 4))
        x, _, _ = Images(self.table, self.mask)[0]
        self.assertAlmostEqual(x.mean().item(), 0, places=6)
        self.assertAlmostEqual(x.std(unbiased=False).item(), 1, places=6)
        affine = np.eye(4); affine[0, 3] = 1
        self.save(np.arange(64, dtype='float32').reshape(4, 4, 4), affine)
        with self.assertRaisesRegex(ValueError, 'grid'):
            Images(self.table, self.mask)[0]

    def test_invalid_images(self):
        for data in [np.ones((4, 4, 4), dtype='float32'), np.full((4, 4, 4), np.nan, dtype='float32')]:
            self.save(data)
            with self.assertRaisesRegex(ValueError, 'constant'):
                Images(self.table, self.mask)[0]

    def test_annotations_and_ids(self):
        table = pd.DataFrame({f'Width_reader{i}': [float(i), np.nan] for i in range(1, 7)})
        self.assertEqual(labels(table.iloc[:1], ['Width'])[0, 0], 3.5)
        with self.assertRaises(ValueError):
            labels(table, ['Width'])
        pd.concat([self.table, self.table]).to_csv(self.root / 'manifest.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'unique'):
            read_manifest(self.root / 'manifest.csv')

    def test_checkpoint_reload_and_permutation(self):
        arch = dict(patch_size=16, depth=2, embed_dim=16, num_heads=2)
        model = build_model(64, 1, arch); model.eval()
        b = dict(format_version=1, task='width', targets=TASKS['width'], architecture=arch,
                 state_dict=model.state_dict(), shift_index=torch.arange(64),
                 train_mean=torch.zeros(1), train_std=torch.ones(1), **mask_payload(self.mask))
        path = self.root / 'width.pt'; torch.save(b, path)
        _, reloaded, _ = load_bundle(path, 'cpu')
        x = torch.randn(2, 1, 64)
        with torch.inference_mode():
            torch.testing.assert_close(model([x])[0], reloaded([x])[0], rtol=0, atol=0)
        b['shift_index'][1] = 0; torch.save(b, path)
        with self.assertRaisesRegex(ValueError, 'permutation'):
            load_bundle(path, 'cpu')

    def test_split_and_holdout(self):
        table = pd.DataFrame({'eid': [str(i) for i in range(20)], 'image': [f'/img/{i}' for i in range(20)]})
        a = split_table(table, train_count=12, val_count=4)
        pd.testing.assert_frame_equal(a, split_table(table, train_count=12, val_count=4))
        self.assertEqual(a['split'].value_counts().to_dict(), {'train': 12, 'val': 4, 'test': 4})
        b = dict(train_ids=['0'], validation_ids=['1'], train_images=['/img/0'], validation_images=['/img/1'])
        ensure_held_out(table.iloc[2:], b)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            ensure_held_out(table, b)
        # Relabeling a duplicate image does not turn it into an independent test subject.
        with self.assertRaisesRegex(ValueError, 'overlap'):
            ensure_held_out(pd.DataFrame({'eid':['new'], 'image':['/img/0']}), b)

    def test_metrics(self):
        y = np.array([[1.], [2.], [3.]])
        result = regression_metrics(y, np.full_like(y, 2.), ['Width'])['Width']
        self.assertAlmostEqual(result['mae'], 2 / 3)
        self.assertAlmostEqual(result['rmse'], np.sqrt(2 / 3))
        self.assertEqual(result['r2'], 0)
        self.assertIsNone(result['pearson_r'])
        self.assertAlmostEqual(regression_metrics(y, y, ['Width'])['Width']['pearson_r'], 1)


if __name__ == '__main__':
    unittest.main()
