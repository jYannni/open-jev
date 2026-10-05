# Gemma 4 support

open-jev runs on Gemma 4 instruct models through the existing `--model <path|repo>` option.
`DEFAULT_MODEL`, the Makefile defaults and Gemma 3 behaviour are unchanged.

```sh
openjev serve --model mlx-community/gemma-4-e4b-it-4bit
openjev serve --model mlx-community/gemma-4-12B-it-qat-4bit
make serve MODEL=mlx-community/gemma-4-e4b-it-4bit
```

## Checkpoints this was written against

| Model | MLX repo (ungated) | `model_type` | Layout |
|---|---|---|---|
| Gemma 4 E4B | `mlx-community/gemma-4-e4b-it-4bit` (also `-8bit`) | `gemma4` | multimodal wrapper; text: hidden 2560, 42 layers, last 18 layers share K/V, per-layer input embeddings |
| Gemma 4 12B | `mlx-community/gemma-4-12B-it-qat-4bit` | `gemma4_unified` | encoder-free multimodal wrapper; text: hidden 3840, 48 layers, no K/V sharing, K = V on full-attention layers |

Both use vocab 262144, BOS id 2, final-logit soft-capping 30, and tied embeddings.
`mlx-community/gemma-4-12B-it-OptiQ-4bit` has the same `gemma4_unified` type but a different
chat template (not reviewed here).

**mlx-lm version.** `gemma4` loads from mlx-lm 0.31, but `gemma4_unified` (the 12B) is only
mapped onto the `gemma4` implementation from **mlx-lm 0.32.0**; older versions fail at load with an
unsupported model type. `pyproject.toml` now requires `mlx-lm>=0.32.0` and `uv.lock` pins 0.32.0
(previously 0.31.3; nothing else in the lock moved). Run `uv sync` on the Mac mini to pick it up. PyTorch: transformers 5.17
(locked) maps both types to their native `*ForConditionalGeneration` wrappers through
`AutoModelForCausalLM`, so `torch_backend.py` needed no change.

## Code changes

- **K/V cache sizing (MLX).** mlx-lm's Gemma 4 model expects one cache per layer that *owns*
  K/V (`model.make_cache()`) and pads the list with `None` for the K/V-sharing layers. open-jev
  used to build one `KVCache` per layer. On E4B that made the sharing layers append the shared
  keys a second time, and the first cached call failed with a shape mismatch.
  `OptionScorer.new_cache()` now returns the model's own `make_cache()`: the right number of
  caches, with a `RotatingKVCache` on sliding-window layers. `_expand()` copies each cache with
  its type and trims rotating caches to their window first, so the per-option batch copies
  stay bounded by the window instead of growing with the full context. Scores are unchanged:
  cached and naive scoring still agree exactly on CPU, for Gemma 3 too.
- **Chat template.** `context_ids(chat=True)` passes `enable_thinking=False`. Templates that
  don't read the flag (Gemma 3) ignore it.
- **Opt-in chat for `system_one`.** `system_one(..., chat=None)` now follows the scorer's
  `chat` setting (off by default) instead of hard-coding `chat=False`. An explicit `chat=`
  argument overrides it. `openjev serve --chat` turns it on for `/v1/systemone`, and `/health`
  reports it. `/score` keeps its per-request `chat` field. See "Gemma 4 12B needs chat scoring".
- **Dependency.** `mlx-lm>=0.32.0` (see above).
- **Feature norms.** `train` prints and records (`head.json` → `feature_norms`) the mean L2
  norm of context-token and option features. See below.

## Template details

By default the benchmark path (`systemone`, `norm="sum"`) scores raw text: the context is
encoded with the tokenizer's own BOS and no chat template. With a chat scorer
(`OptionScorer(..., chat=True)` or `serve --chat`) it uses the templates below.

With `chat=True` the Gemma 4 templates render one user turn as:

```
<bos><|turn>user\n{context, trimmed}<turn|>\n<|turn>model\n                              (E4B)
<bos><|turn>user\n{context, trimmed}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>  (12B qat)
```

The 12B template appends an empty thought channel whenever thinking is off, so options are
scored as the visible answer after it. That is the template's intended non-thinking reply
prefix. The template emits `<bos>` as text, which the tokenizer maps to id 2; `context_ids`
encodes with `add_special_tokens=False` and only prepends BOS when it is missing, so BOS is
never doubled (covered by a test). Options are encoded without special tokens, as before.

## Frozen-feature head (Route A)

`FeatureExtractor` takes `model.language_model.model` (`Gemma4TextModel`: embeddings, per-layer
inputs, decoder layers, final norm, no LM head), the same attribute path as Gemma 3. The hidden
size comes from `language_model.args.hidden_size`: 2560 for E4B and 3840 for 12B. Heads trained on
Gemma 3 (2560) features are not reusable, including on E4B: the dimension matches but the
feature space differs. Retrain.

**Learning rate.** The 5e-4 default (and `decision` 1e-4) was tuned on Gemma 3 4B features with
norms of about 115. I could not measure Gemma 4 norms without weights. The first `openjev train` run prints
`mean feature norms: ...`. AdamW's step size does not depend on the gradient scale, so the
head's effective change scales with the input norm. As a starting point, scale the learning rate
by `115 / measured_norm` and confirm on validation. This rule of thumb has not been verified on Gemma 4.

## Gemma 4 12B needs chat scoring

Measured on the Mac mini with `mlx-community/gemma-4-12B-it-qat-4bit`. The model itself works:
`mlx_lm.generate` reasons its way to the correct answer. Raw-text log-probabilities, however,
are badly distorted:

| Context → option | E4B raw | 12B raw | 12B `--chat` |
|---|---|---|---|
| "The capital of France is" → " Paris" (`check`) | −0.025 | −16.25 | — |
| "What is the capital of France? Answer in one word." → "Paris" | — | −24.87 | −0.000 |

With `serve --chat` the 12B quickstart answers are: department `technical` (0.9997),
frustration 2.00 ("Furious", 0.998), is_urgent 1.00. Raw quickstart answers from the 12B were wrong: department `billing` at 0.997, frustration mostly
"Calm and factual". This matches the OptiQ 12B model card, which warns that cloze-style
(log-likelihood) scoring underscores this reasoning model. The 12B is benchmarked with
`OptionScorer(model, chat=True)` or `serve --chat`. Its cached-vs-naive drift (`check`, raw
text) is similar to E4B's: 0.30 / 0.60 / 1.72 at 6 / 14 / 1501 tokens, 0.36% relative at
1501 tokens.

Route A feature extraction (`openjev decision`, `openjev features`) still encodes raw text. Hidden
states are not log-probabilities, so this may matter less there, but it is not verified for the 12B.

## Benchmark contract

`OptionScorer` and `openjev.systemone.SystemOneRequest` are unchanged. `system_one` gains an
optional trailing `chat: bool | None = None` argument. Existing calls behave as before unless the
scorer was itself built with `chat=True`, which now carries through instead of being ignored.
`OptionScorer.new_cache()` is a new public helper.

## Tests

```sh
uv run python -m unittest discover -s tests -p 'test_gemma4.py' -v
```

`tests/test_gemma4.py` builds tiny random Gemma 4 models in two shapes, E-series (K/V
sharing, per-layer inputs) and unified (K = V, no sharing). It runs the same checks as the Gemma 3
tests:

- prefix-cached vs naive scores across batch sizes;
- norms and repeated calls;
- chat and separator, with exactly one BOS;
- loading both multimodal wrappers through `TorchBackend`;
- on MLX: cache length, features vs an uncached forward pass, and Route A extract plus a one-epoch train.

The MLX tests run wherever mlx-lm imports. They were run with `mlx[cpu]` 0.32.3 and mlx-lm
0.32.0 on Linux. The PyTorch tests ran with the locked torch/transformers.

## Verified on real weights (Mac mini, Apple silicon)

`mlx-community/gemma-4-e4b-it-4bit`, mlx-lm 0.32.0: `tests/test_gemma4.py` passes on Metal; the
server loads the checkpoint and its warm-up scoring call succeeds (unchanged upstream crashes
there with the K/V-sharing shape mismatch). `examples/systemone-quickstart.json`:

| Question | Answer | Probabilities |
|---|---|---|
| department (choice) | technical | billing 1.1e-6, technical 0.999999, sales 6.1e-11 |
| frustration (score) | 1.98 | 0: 7.4e-5, 1: 0.016, 2: 0.984 |
| is_urgent (noul) | 0.905 | — |

Zero-shot `norm="sum"` probabilities are very sharp. "Furious, harsh wording" is arguably too
strong for this ticket.

### `openjev check --context-tokens 1500` (cached vs naive, MLX 4-bit on Metal)

| Model | 6-token context | 14-token context | 1501-token context | Result |
|---|---|---|---|---|
| `mlx-community/gemma-3-4b-it-4bit` | 0.43 (3.2%) | 0.17 (1.9%) | 0.31 (0.23%) | OK |
| `mlx-community/gemma-4-e4b-it-4bit` | 0.17 (1.1%) | 0.54 (4.4%) | 2.04 (0.71%) | MISMATCH at tol 0.5 |

The values are the worst absolute log-prob difference per case, with the relative difference in
brackets. Option rankings are identical between cached and naive in every case for both models.
On CPU, in float32 and bfloat16, cached and naive agree exactly for tiny Gemma 4 (E-series and
unified) and Gemma 3 models with a 1602-token context past a 64-token sliding window. The
difference is therefore attributed to Metal kernel rounding in bfloat16, not to the cache logic.
That attribution is inferred, not proven on Metal. Gemma 4's drift at long context is about 3×
Gemma 3's in relative terms, and its option scores are about 2.4× larger. E4B runs at 401 tokens (inside its 512-token window) and 701 tokens gave 1.55 (0.58%) and 1.35
(0.47%), with identical rankings. The drift is roughly constant per token, so it does not come
from the sliding window. Expect `check` to fail
its default `--tol 0.5` on Gemma 4 for long contexts. For short labels, as in `systemone`, the
drift matches Gemma 3's.

## Not verified without weights

- Feature norms and a working learning rate for Route A.
- Numerical agreement between MLX 4-bit and PyTorch bf16 on real weights.
- Gemma 4's 512/1024-token sliding windows with contexts longer than the window, on real
  weights. Plain `KVCache` plus the windowed mask is the same mechanism Gemma 3 uses, and it is
  covered by the tiny-model tests with an 8-token window.
