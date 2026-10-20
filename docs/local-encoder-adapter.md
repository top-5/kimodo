# Local compatible encoder bundles

This fork adds `LOCAL_LLM2VEC_DIR` for a complete, locally installed LLM2Vec
bundle. It keeps the upstream text-encoder callable contract and lets applications
keep their existing REST orchestration. No additional API server is necessary.

The bundle must contain the merged MNTP LlamaBiModel base, tokenizer files, and
`supervised_adapter/adapter_config.json` plus its adapter weights. The loader
requires a LlamaBiModel configuration with 4096 hidden dimensions. It never pads
other model embeddings or redirects a missing base model to another Hub repo.

A published candidate is `matbee/kimodo-llm2vec-nf4`, revision
`268b2f7bbbb8361bce76cc3f34d61b5467643a59`. It contains a quantized merged MNTP base
and the separate supervised adapter. The original model's license still applies.

```sh
hf download matbee/kimodo-llm2vec-nf4 \
  --revision 268b2f7bbbb8361bce76cc3f34d61b5467643a59 \
  --local-dir /path/to/encoder
TEXT_ENCODER_MODE=local LOCAL_LLM2VEC_DIR=/path/to/encoder \
  kimodo_gen 'A person dances with wide side steps.' --duration 4
```

Prequantized loading requires bitsandbytes and CUDA. Device placement occurs at
load time; attempts to relocate quantized weights afterward are rejected. All
model/tokenizer loads use local-files-only mode. Instruction formatting and the
supervised adapter remain consistent with upstream's Llama-3 LLM2Vec encoder.

Nemotron can author choreography text through an application REST API. It does
not replace this pretrained embedding representation. The adapter does not train
or claim a Nemotron-to-LLM2Vec translation.

Boundary tests: `python -m unittest discover -s tests -v`. Real embedding and
motion generation must be checked with the installed bundle before deployment.
