"""Projection quality gates and REST identity boundaries; no model downloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


encoder = load('nemotron_encoder', 'kimodo/model/nemotron_text_encoder.py')
trainer = load('nemotron_trainer', 'kimodo/scripts/train_nemotron_projection.py')


class NemotronTests(unittest.TestCase):
    def bundle(self, root):
        np.savez(root / 'projection.npz', weight=np.ones((3, 4096), np.float32),
                 source_mean=np.zeros(3, np.float32), target_mean=np.zeros(4096, np.float32))
        m = {'schema': 'kimodo-nemotron-projection/1', 'source_dimension': 3, 'target_dimension': 4096,
             'source_model': 'test-only', 'source_revision': 'pinned', 'pooling': 'LAST',
             'weights_sha256': hashlib.sha256((root / 'projection.npz').read_bytes()).hexdigest()}
        (root / 'manifest.json').write_text(json.dumps(m))
        return m

    def test_rejects_modified_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root)
            with (root / 'projection.npz').open('ab') as stream:
                stream.write(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                encoder.NemotronProjection(root)

    def test_rejects_nonfinite_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root)
            with self.assertRaisesRegex(ValueError, 'features'):
                encoder.NemotronProjection(root)([[0, float('nan'), 1]])

    def test_rest_identity_batch_order_and_single_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m = self.bundle(root)
            client = encoder.NemotronTextEncoder(root, 'http://test.invalid/v1')
            response = {**m, 'data': [{'index': 1, 'embedding': [1, 1, 1]}, {'index': 0, 'embedding': [2, 2, 2]}]}
            with patch.object(encoder, 'urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
                values, lengths = client(['first', 'second'])
            self.assertEqual(tuple(values.shape), (2, 1, 4096))
            self.assertEqual(lengths, [1, 1])
            self.assertEqual(values[:, 0, 0].tolist(), [6, 3])
            response['data'] = [{'index': 0, 'embedding': [1, 1, 1]}]
            with patch.object(encoder, 'urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
                values, length = client('single')
            self.assertEqual(tuple(values.shape), (1, 4096))
            self.assertEqual(length, 1)
            response['source_revision'] = 'wrong-checkpoint'
            with patch.object(encoder, 'urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    client('single')

    def test_fit_recovers_held_out_linear_map(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(40, 3))
        y = x @ rng.normal(size=(3, 4096)) + 0.1
        split = np.array(['train'] * 24 + ['validation'] * 8 + ['test'] * 8)
        metadata = dict(source_model='test-only', source_revision='pinned', pooling='LAST',
                        teacher_model='test-only', teacher_revision='pinned', corpus_sha256='test-only')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bundle'
            report = trainer.fit(x, y, split, metadata, output)
            self.assertLess(report['test']['mse'], 1e-6)
            prediction = encoder.NemotronProjection(output)(x[-8:])
            np.testing.assert_allclose(prediction, y[-8:], atol=0.001)

    def test_rejects_projection_that_cannot_beat_mean(self):
        x = np.zeros((12, 3))
        y = np.ones((12, 4096))
        split = np.array(['train'] * 4 + ['validation'] * 4 + ['test'] * 4)
        metadata = dict(source_model='test-only', source_revision='pinned', pooling='LAST',
                        teacher_model='test-only', teacher_revision='pinned', corpus_sha256='test-only')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'baseline'):
                trainer.fit(x, y, split, metadata, Path(directory) / 'bundle')


if __name__ == '__main__':
    unittest.main()
