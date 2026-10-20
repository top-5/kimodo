# Experimental Nemotron conditioning adapter

This fork implements **Nemotron pooled hidden states → trained ridge projection
→ Kimodo conditioning**. It uses real feature vectors, rather than parsing chat
responses as embeddings. The REST client preserves Kimodo's `[batch,1,4096]`
callable contract. The existing compatible LLM2Vec route remains the default.

The existing chat-only vLLM endpoint did not expose `/v1/embeddings`. An isolated
vLLM pooling engine successfully extracted 2688-dimensional LAST-token features
from `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`, revision
`16993199e436da4ba75ddc410855f87e0d996ee6`. That engine was tested without
restarting or modifying the shared chat service. It loaded about 18.23 GiB of
weights; loading reported temporary allocation failures, then recovered. Budget
GPU memory before running it alongside other workloads: the test engine reserved
about 25 GiB including caches at utilization 0.25. vLLM is an optional,
separate runtime; use its existing supported environment.

## Reproduce calibration

`examples/nemotron/pilot-corpus.json` contains 256 short, synthetic descriptions
across 32 motion families and eight modifiers. Whole families are assigned to
train/validation/test: 192/32/32 descriptions. This is a small calibration pilot,
not a broad human-motion dataset. Corpus SHA256:
`a426a7806a25730e1d2ad70346d6e1c22f120f81b326a8b6b03a94146dc3eda2`.

Run feature and teacher extraction sequentially to release GPU memory between
models. Use a local pinned model snapshot; no download is performed by the
feature script. Substitute installed environment paths below.

```sh
python -m kimodo.scripts.nemotron_features \
  --model-path /path/to/pinned/nemotron-snapshot \
  --model-id nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --revision 16993199e436da4ba75ddc410855f87e0d996ee6 \
  --corpus examples/nemotron/pilot-corpus.json --output /data/source.npz
python -m kimodo.scripts.nemotron_teacher_pairs \
  --corpus examples/nemotron/pilot-corpus.json --source /data/source.npz \
  --teacher-dir /path/to/compatible-llm2vec-bundle \
  --teacher-id matbee/kimodo-llm2vec-nf4 \
  --teacher-revision 268b2f7bbbb8361bce76cc3f34d61b5467643a59 \
  --output /data/pairs.npz
python -m kimodo.scripts.train_nemotron_projection \
  --pairs /data/pairs.npz --metadata /data/pairs.npz.json \
  --output /data/projection
```

The trainer selects regularization on validation MSE and reports an untouched
test partition against a constant teacher-mean baseline. It refuses to export
when test MSE fails to beat that baseline. The paired-feature collector rejects
corpus identity mismatches and groups crossing partitions. Teacher labels come
from the installed compatible quantized LLM2Vec bundle, not a Nemotron teacher.

The first actual pilot selected ridge alpha 0.1. Test cosine similarity was
0.89260 versus baseline 0.80050; MSE was 0.89424 versus baseline 1.57821, a 43.34%
reduction. Validation cosine was 0.91507. Exported projection SHA256:
`fa5a325b335285d610281ce49fd4f9d864f6f9a023b13693fb1fbf6c0e376640`.
These are embedding metrics. A high absolute cosine partly reflects common
teacher components; it does not establish prompt adherence or choreography quality.

## REST inference

Start the same feature script with `--serve` instead of `--corpus/--output`, using
the pinned model arguments above. It binds `127.0.0.1:9551` and exposes
`POST /v1/embeddings`. Requests use `model`, `input`, and float embeddings.
Responses include `source_model`, `source_revision`, and `pooling` for identity
checking. This is a local inference component, not a public authenticated service.

Select the trained adapter explicitly:

```sh
TEXT_ENCODER_MODE=local TEXT_ENCODER=nemotron \
NEMOTRON_PROJECTION_DIR=/data/projection \
NEMOTRON_EMBEDDINGS_URL=http://127.0.0.1:9551/v1 \
  kimodo_gen 'A person dances with wide side steps.' --duration 4
```

`NEMOTRON_EMBEDDINGS_API_KEY` optionally supplies a bearer token when using an
authenticated compatible feature service. It is never stored in a bundle.
Checkpoint/pooling mismatch, incomplete responses, checksum changes and nonfinite
features fail explicitly. There is no automatic fallback to teacher embeddings.

A real REST → projection → SOMA-RP-v1.2 generation produced 120 finite frames at
30 FPS, 77 joints and about 1.99 metres of root travel for a four-second dance
prompt. This proves execution. Broader dance-prompt adherence, foot sliding,
transition quality and equivalence to teacher-conditioned motion remain unmeasured.
The feature probe service is temporary; this change does not install a new
always-running model server or switch application defaults.
