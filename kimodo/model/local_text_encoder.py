# SPDX-License-Identifier: Apache-2.0
"""Explicit local LLM2Vec bundles, including prequantized Kimodo exports.

No remote base-model fallback, synthetic embeddings, dimension padding or chat-model substitution.
"""
import json
import os
from pathlib import Path


def inspect_bundle(directory):
    root = Path(directory).resolve()
    config = json.loads((root / 'config.json').read_text())
    if config.get('hidden_size') != 4096 or config.get('model_type') != 'llama' or config.get('architectures') != ['LlamaBiModel']:
        raise ValueError('Kimodo requires the trained Llama-3 LLM2Vec representation (4096 dimensions)')
    if not (root / 'model.safetensors').is_file() or not (root / 'tokenizer_config.json').is_file():
        raise ValueError('Local LLM2Vec bundle is incomplete')
    adapter = root / 'supervised_adapter'
    settings = json.loads((adapter / 'adapter_config.json').read_text())
    if settings.get('peft_type') != 'LORA' or not (adapter / 'adapter_model.safetensors').is_file():
        raise ValueError('The compatible supervised LLM2Vec adapter is required')
    quantized = bool(config.get('quantization_config', {}).get('load_in_4bit'))
    return root, adapter, quantized


class LocalLLM2VecEncoder:
    """Same callable interface as upstream LLM2VecEncoder, with an explicit local bundle."""
    def __init__(self, directory, device='auto'):
        root, adapter, self.quantized = inspect_bundle(directory)
        import torch
        from .llm2vec.llm2vec import LLM2Vec
        self.torch = torch
        self.llm_dim = 4096
        device = os.environ.get('TEXT_ENCODER_DEVICE', device)
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self.quantized and not str(device).startswith('cuda'):
            raise ValueError('This prequantized encoder requires a CUDA device')
        self._device = torch.device(device)
        kwargs = dict(torch_dtype=torch.bfloat16, local_files_only=True)
        if self.quantized:
            kwargs['device_map'] = {'': str(self._device)}
        self.model = LLM2Vec.from_pretrained(str(root), str(adapter), **kwargs)
        # Preserve upstream's exact instruction formatting: this is a semantic model identity, not a Hub fetch.
        self.model.model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
        if not self.quantized:
            self.model = self.model.to(self._device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def to(self, device=None, dtype=None):
        target = self.torch.device(device) if device is not None else self._device
        if self.quantized:
            if target.type != self._device.type or (target.index or 0) != (self._device.index or 0):
                raise ValueError('Cannot relocate a prequantized encoder after loading')
        else:
            self.model = self.model.to(device=target, dtype=dtype)
        self._device = target
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self._device

    def __call__(self, text):
        single = isinstance(text, str)
        texts = [text] if single else text
        if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
            raise ValueError('Expected a prompt string or nonempty list of prompt strings')
        with self.torch.no_grad():
            values = self.model.encode(texts, batch_size=1, show_progress_bar=False, device=self._device)
            embeddings = self.torch.as_tensor(values, device=self._device)
        if embeddings.shape != (len(texts), 4096) or not self.torch.isfinite(embeddings).all():
            raise ValueError('Invalid trained LLM2Vec output; expected finite [batch,4096] embeddings')
        embeddings = embeddings[:, None]
        return (embeddings[0], 1) if single else (embeddings, [1] * len(texts))
