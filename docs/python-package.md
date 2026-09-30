# Use openjev from Python

The distribution includes the `openjev` Python API and CLI. Python 3.12 or newer
is required. From a checkout containing this version:

```sh
python -m pip install .
# For development, use: python -m pip install -e .
```

To build a wheel that others can install without a source checkout:

```sh
uv build --wheel
python -m pip install dist/openjev-0.1.0-py3-none-any.whl
```

These commands install declared dependencies. Building a wheel does not publish
to PyPI. Model weights, trained artifacts, datasets, and repository demo scripts
are not bundled. Imports do not download weights or initialize a GPU.

## Score candidates with a trained head

Decision heads currently require Apple silicon and MLX. Pass a head artifact
directory produced by `train_decision_head` or `openjev decision train`:

```python
from openjev import DecisionScorer

scorer = DecisionScorer(
    "artifacts/support-routing",
    model="models/gemma-3-4b-it",
)
result = scorer.predict(
    state={"message": "My item arrived broken. Please replace it."},
    question="Which department should handle this request?",
    candidates={
        "returns": "Returns and replacements",
        "billing": "Billing and payments",
    },
    request_id="ticket-001",
)
print(result["choice_id"])
for candidate in result["candidates"]:
    print(candidate["id"], candidate["utility"], candidate["probability"])
```

This example assumes a head trained for support routing. Reuse the scorer for
multiple requests. The optional `model` argument relocates the original backbone;
its weights and tokenizer must match the artifact's content fingerprint. When
omitted, the backbone path recorded during training is used. The feature extraction
library versions must match the artifact too. Share the head directory, compatible
backbone, and pinned environment together.

`predict` takes 2–255 unique nonempty candidate IDs and texts in a mapping. State
accepts a JSON string, object, or array. The result is a JSON-compatible dictionary
with `id`, `strategy`, `choice_id`, `candidates`, `artifact_id`, and `seconds`.
Candidate results retain input order. Probabilities sum to one over the supplied
candidates; they are not automatically calibrated confidence estimates.

For an existing native request dictionary, use `scorer.score(request)`. You can
also parse it explicitly with `ChoiceTask.parse(request)` and pass that object to
`score`. Invalid requests raise `ValueError`; file access errors retain the normal
Python filesystem exceptions. `Candidate` and `ChoiceTask` are available from
`openjev` for constructing and inspecting requests.

## Extract features and train from Python

Training uses the same validated JSONL format and artifact checks as the CLI:

```python
from openjev import extract_decision_features, train_decision_head

for split in ("train", "validation"):
    metadata = extract_decision_features(
        data=f"data/{split}.jsonl",
        model="models/gemma-3-4b-it",
        out=f"runs/support/{split}.npz",
        split=split,
        batch_size=8,
    )

training = train_decision_head(
    train_path="runs/support/train.npz",
    validation_path="runs/support/validation.npz",
    out="artifacts/support-routing",
    rank=256,
    epochs=8,
    batch_size=64,
    lr=1e-4,
    seed=7,
)
print(training["artifact"], training["best_val_top1"])
```

Feature extraction returns metadata; training returns the artifact path and best
validation top-1 score. Both can emit progress to stdout. Output paths must be new.
Training optimizes categorical cross-entropy on a frozen backbone. See the
[decision-head guide](decision-head.md) for the dataset schema, leakage checks,
token limits, evaluation, and artifact compatibility rules.

## Use continuation scoring without a trained head

The existing scorer supports MLX on Apple silicon and PyTorch on other platforms:

```python
from openjev import OptionScorer

scorer = OptionScorer(model_path="models/gemma-3-4b-it", backend="auto")
scores = scorer.score("The capital of France is", [" Paris", " Rome"])
for score in scores:
    print(score.option, score.probability)
```

Continuation scores and learned decision-head utilities use different objectives.
The Python API does not add new model-family support or change either objective.
