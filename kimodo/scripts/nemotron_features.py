# SPDX-License-Identifier: Apache-2.0
"""Extract pinned Nemotron pooling features, offline or through an isolated REST server.

Never attach this to or restart a shared chat engine. The model path must be an
existing local snapshot. Serving is opt-in and binds loopback by default.
"""
import argparse
import atexit
import hashlib
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--corpus')
    p.add_argument('--output')
    p.add_argument('--serve', action='store_true')
    p.add_argument('--port', type=int, default=9551)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.25)
    args = p.parse_args()
    if not (Path(args.model_path) / 'config.json').is_file():
        p.error('An existing local model snapshot is required')
    if not args.serve and (not args.corpus or not args.output):
        p.error('Offline extraction requires --corpus and --output')
    from vllm import LLM
    llm = LLM(model=args.model_path, runner='pooling', convert='embed', trust_remote_code=True,
              language_model_only=True, max_model_len=256, max_num_seqs=1,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
              enable_prefix_caching=False, moe_backend='cutlass')
    # vLLM workers otherwise outlive a SIGTERM'd parent and retain GPU memory.
    atexit.register(llm.llm_engine.engine_core.shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    identity = {'source_model': args.model_id, 'source_revision': args.revision, 'pooling': 'LAST'}

    def encode(texts):
        if (not isinstance(texts, list) or not 1 <= len(texts) <= 64
                or any(not isinstance(t, str) or len(t) > 700 for t in texts)):
            raise ValueError('Expected 1–64 short motion descriptions')
        return [x.outputs.embedding for x in llm.embed(texts)]

    if not args.serve:
        corpus_path = Path(args.corpus)
        corpus = json.loads(corpus_path.read_text())
        identity['corpus_sha256'] = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        values = []
        for offset in range(0, len(corpus), 32):
            values.extend(encode([row['text'] for row in corpus[offset:offset + 32]]))
        np.savez(args.output, embeddings=np.asarray(values, dtype=np.float32))
        Path(args.output + '.json').write_text(json.dumps(identity, indent=2) + '\n')
        return

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != '/v1/embeddings':
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(size))
                if request.get('model') != args.model_id:
                    raise ValueError('Unexpected model')
                texts = request['input']
                texts = [texts] if isinstance(texts, str) else texts
                rows = encode(texts)
                body = json.dumps({**identity, 'model': args.model_id, 'object': 'list',
                                   'data': [{'object': 'embedding', 'index': i, 'embedding': row}
                                            for i, row in enumerate(rows)]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, KeyError, TypeError) as error:
                self.send_error(400, str(error))
            except Exception:
                self.send_error(503, 'Feature extraction failed')

    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
