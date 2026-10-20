# SPDX-License-Identifier: Apache-2.0
"""Experimental REST Nemotron features followed by a learned Kimodo projection.

The projection must be trained against Kimodo's LLM2Vec teacher. A chat-completion
endpoint, zero padding, or an untrained matrix cannot substitute for this bundle.
"""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


class NemotronProjection:
    def __init__(self, directory):
        root = Path(directory)
        self.manifest = json.loads((root / 'manifest.json').read_text())
        m = self.manifest
        if m.get('schema') != 'kimodo-nemotron-projection/1' or m.get('target_dimension') != 4096:
            raise ValueError('Expected a trained Nemotron-to-Kimodo projection bundle')
        path = root / 'projection.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != m.get('weights_sha256'):
            raise ValueError('Projection checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            self.weight = data['weight'].copy()
            self.source_mean = data['source_mean'].copy()
            self.target_mean = data['target_mean'].copy()
        dimension = m['source_dimension']
        if (self.weight.shape != (dimension, 4096) or self.source_mean.shape != (dimension,)
                or self.target_mean.shape != (4096,) or any(
                    not np.isfinite(x).all() for x in (self.weight, self.source_mean, self.target_mean))):
            raise ValueError('Invalid projection arrays')

    def __call__(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.weight.shape[0] or not np.isfinite(x).all():
            raise ValueError('Invalid Nemotron features')
        result = (x - self.source_mean) @ self.weight + self.target_mean
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite projected conditioning')
        return result


class NemotronTextEncoder:
    """Kimodo callable using a revision-identified embeddings REST service."""
    def __init__(self, directory, url, device='cpu', timeout=120):
        self.projection = NemotronProjection(directory)
        self.url = url.rstrip('/') + '/embeddings'
        self.timeout = timeout
        self.device = device
        self.dtype = None
        self.llm_dim = 4096

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def eval(self):
        return self

    def get_device(self):
        import torch
        return torch.device(self.device)

    def __call__(self, text):
        import torch
        single = isinstance(text, str)
        texts = [text] if single else text
        if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
            raise ValueError('Expected a prompt or nonempty list of prompts')
        manifest = self.projection.manifest
        headers = {'Content-Type': 'application/json'}
        token = os.environ.get('NEMOTRON_EMBEDDINGS_API_KEY')
        if token:
            headers['Authorization'] = 'Bearer ' + token
        request = Request(self.url, data=json.dumps({
            'model': manifest['source_model'], 'input': texts, 'encoding_format': 'float',
        }).encode(), headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        # Generic OpenAI responses lack revision/pooling evidence. Reject them rather
        # than silently using a different checkpoint or pooling strategy.
        if any(result.get(key) != manifest[key] for key in ('source_model', 'source_revision', 'pooling')):
            raise ValueError('Nemotron feature service identity differs from trained projection')
        rows = result.get('data', [])
        if len(rows) != len(texts) or sorted(r.get('index', -1) for r in rows) != list(range(len(texts))):
            raise ValueError('Incomplete or duplicate Nemotron embeddings')
        rows.sort(key=lambda r: r['index'])
        values = self.projection([r['embedding'] for r in rows])
        embeddings = torch.as_tensor(values, device=self.device, dtype=self.dtype or torch.float32)[:, None]
        return (embeddings[0], 1) if single else (embeddings, [1] * len(texts))
