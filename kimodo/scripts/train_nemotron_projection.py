# SPDX-License-Identifier: Apache-2.0
"""Fit a ridge projection to paired Nemotron/LLM2Vec features, with held-out evaluation.

Input NPZ: source[N,D], target[N,4096], split[N] (train/validation/test).
Input JSON: source_model, source_revision, pooling, teacher_model, teacher_revision,
corpus_sha256. Split related prompt variants together before extracting features.
This produces an experimental bundle, not evidence of downstream motion quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(prediction, target):
    norm = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    return {'mse': float(np.mean((prediction - target) ** 2)),
            'cosine': float(np.mean(np.sum(prediction * target, axis=1) / np.maximum(norm, 1e-12)))}


def fit(source, target, split, metadata, output):
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    split = np.asarray(split)
    if (source.ndim != 2 or target.shape != (len(source), 4096) or split.shape != (len(source),)
            or not np.isfinite(source).all() or not np.isfinite(target).all()
            or set(split.tolist()) != {'train', 'validation', 'test'}):
        raise ValueError('Expected finite paired features with train/validation/test partitions')
    for key in ('source_model', 'source_revision', 'pooling', 'teacher_model', 'teacher_revision', 'corpus_sha256'):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError('Missing feature provenance: ' + key)
    train, validation, test = (split == name for name in ('train', 'validation', 'test'))
    xm, ym = source[train].mean(0), target[train].mean(0)
    x, y = source[train] - xm, target[train] - ym
    # Dual form supports fewer training prompts than feature dimensions.
    kernel = x @ x.T
    scale = max(float(np.trace(kernel) / len(kernel)), 1e-12)
    best = None
    for alpha in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0):
        weight = x.T @ np.linalg.solve(kernel + np.eye(len(kernel)) * scale * alpha, y)
        score = metrics((source[validation] - xm) @ weight + ym, target[validation])
        if best is None or score['mse'] < best[0]:
            best = (score['mse'], alpha, weight, score)
    _, alpha, weight, validation_metrics = best
    test_metrics = metrics((source[test] - xm) @ weight + ym, target[test])
    baseline = metrics(np.broadcast_to(ym, target[test].shape), target[test])
    if test_metrics['mse'] >= baseline['mse']:
        raise ValueError('Held-out projection does not beat the constant teacher-mean baseline')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'projection.npz'
    np.savez(weights, weight=weight.astype(np.float32), source_mean=xm.astype(np.float32), target_mean=ym.astype(np.float32))
    manifest = {**metadata, 'schema': 'kimodo-nemotron-projection/1', 'source_dimension': source.shape[1],
                'target_dimension': 4096, 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                'ridge_alpha': alpha, 'validation': validation_metrics, 'test': test_metrics,
                'baseline_test': baseline, 'counts': {name: int(np.sum(split == name)) for name in ('train', 'validation', 'test')},
                'status': 'experimental; downstream motion validation required'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with np.load(args.pairs, allow_pickle=False) as pairs:
        report = fit(pairs['source'], pairs['target'], pairs['split'], json.loads(Path(args.metadata).read_text()), args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
