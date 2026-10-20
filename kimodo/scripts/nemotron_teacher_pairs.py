# SPDX-License-Identifier: Apache-2.0
"""Pair offline Nemotron features with the installed compatible LLM2Vec teacher."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--teacher-dir', required=True)
    p.add_argument('--teacher-id', required=True)
    p.add_argument('--teacher-revision', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    corpus_path = Path(args.corpus)
    corpus = json.loads(corpus_path.read_text())
    group_splits = {}
    for row in corpus:
        if row['split'] not in ('train', 'validation', 'test') or not row['group']:
            raise ValueError('Each corpus record needs a group and partition')
        if group_splits.setdefault(row['group'], row['split']) != row['split']:
            raise ValueError('Related motion prompts cannot cross evaluation partitions')
    with np.load(args.source, allow_pickle=False) as data:
        source = data['embeddings'].copy()
    metadata = json.loads(Path(args.source + '.json').read_text())
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if metadata.get('corpus_sha256') != digest or len(source) != len(corpus):
        raise ValueError('Nemotron features must match this exact corpus and ordering')
    from kimodo.model.local_text_encoder import LocalLLM2VecEncoder
    teacher = LocalLLM2VecEncoder(args.teacher_dir)
    target, _ = teacher([row['text'] for row in corpus])
    np.savez(args.output, source=source, target=target[:, 0].float().cpu().numpy(),
             split=np.asarray([row['split'] for row in corpus]))
    metadata.update(teacher_model=args.teacher_id, teacher_revision=args.teacher_revision)
    Path(args.output + '.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
