"""Native choice contract and artifact identities. Safe to import without MLX."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

RECIPE = {
    'version': 1,
    'context_rendering': 'state-question-json-v1',
    'chat': False,
    'separator': '',
    'candidate_encoding': 'independent-text-after-bos',
    'candidate_pooling': 'masked-mean-float32-excluding-bos',
    'context_pooling': 'all-final-hidden-states',
    'storage_dtype': 'float16',
    'candidate_ids_encoded': False,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str


@dataclass(frozen=True)
class ChoiceTask:
    id: str
    state: object
    question: str
    candidates: tuple[Candidate, ...]
    choice_id: str | None = None
    group_id: str | None = None

    @classmethod
    def parse(cls, value, labelled=False):
        if not isinstance(value, dict):
            raise ValueError('task must be an object')
        allowed = {'schema_version', 'id', 'state', 'question', 'candidates', 'target', 'group_id'}
        if set(value) - allowed:
            raise ValueError(f'unknown task fields: {sorted(set(value)-allowed)}')
        if type(value.get('schema_version')) is not int or value['schema_version'] != 1:
            raise ValueError('schema_version must be 1')
        for name in ('id', 'question'):
            if not isinstance(value.get(name), str) or not value[name].strip():
                raise ValueError(f'{name} must be a nonempty string')
        state = value.get('state')
        if not isinstance(state, (str, dict, list)):
            raise ValueError('state must be a string, object, or array')
        canonical(state)  # Reject non-JSON values and NaN/Infinity.
        candidates = value.get('candidates')
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 255:
            raise ValueError('need 2..255 candidates')
        parsed = []
        for c in candidates:
            if not isinstance(c, dict) or set(c) != {'id', 'text'}:
                raise ValueError('each candidate must contain only id and text')
            if any(not isinstance(c[k], str) or not c[k].strip() for k in ('id', 'text')):
                raise ValueError('candidate id and text must be nonempty strings')
            parsed.append(Candidate(**c))
        if len({c.id for c in parsed}) != len(parsed):
            raise ValueError('duplicate candidate IDs')
        if len({c.text for c in parsed}) != len(parsed):
            raise ValueError('duplicate candidate text cannot identify a unique choice')
        target = value.get('target')
        choice_id = None
        if target is not None:
            if not isinstance(target, dict) or set(target) != {'choice_id'}:
                raise ValueError('target must contain only choice_id')
            choice_id = target['choice_id']
            if not isinstance(choice_id, str) or choice_id not in {c.id for c in parsed}:
                raise ValueError('target choice_id is not in candidates')
        if labelled and choice_id is None:
            raise ValueError('a choice target is required')
        group_id = value.get('group_id')
        if group_id is not None and (not isinstance(group_id, str) or not group_id.strip()):
            raise ValueError('group_id must be a nonempty string')
        return cls(value['id'], state, value['question'], tuple(parsed), choice_id, group_id)

    @property
    def label(self):
        if self.choice_id is None:
            raise ValueError('task has no target')
        return next(i for i, c in enumerate(self.candidates) if c.id == self.choice_id)

    def to_dict(self):
        value = dict(schema_version=1, id=self.id, state=self.state, question=self.question,
                     candidates=[asdict(c) for c in self.candidates])
        if self.choice_id is not None:
            value['target'] = {'choice_id': self.choice_id}
        if self.group_id is not None:
            value['group_id'] = self.group_id
        return value

    @property
    def input_fingerprint(self):
        # Order and IDs do not disguise a repeated semantic example across splits.
        return digest(dict(state=self.state, question=self.question,
                           candidates=sorted(c.text for c in self.candidates)))


def render_context(task):
    return 'State:\n' + canonical(task.state) + '\n\nQuestion:\n' + task.question


def load_tasks(path, labelled=False):
    tasks, ids = [], set()
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                task = ChoiceTask.parse(json.loads(line), labelled)
                if task.id in ids:
                    raise ValueError('duplicate task ID')
                ids.add(task.id)
                tasks.append(task)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
    if not tasks:
        raise ValueError(f'{path}: empty dataset')
    return tasks


def provenance(tasks):
    return dict(ids=[t.id for t in tasks], inputs=[t.input_fingerprint for t in tasks],
                groups=sorted({t.group_id for t in tasks if t.group_id is not None}))


def check_disjoint(a, b):
    for key in ('ids', 'inputs', 'groups'):
        if set(a[key]) & set(b[key]):
            raise ValueError(f'dataset overlap in {key}; use independent splits')


def model_identity(model):
    root = Path(model).resolve()
    if not root.is_dir() or not (root/'config.json').is_file():
        raise ValueError('Phase 1 requires a local MLX-compatible model directory with config.json')
    files = sorted(p for p in root.rglob('*') if p.is_file() and
                   '.cache' not in p.relative_to(root).parts and
                   p.suffix in ('.json', '.safetensors', '.model', '.jinja', '.txt'))
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('model directory has no safetensors weights')
    hashes = {p.relative_to(root).as_posix(): file_digest(p) for p in files}
    return dict(path=str(root), files=hashes, fingerprint=digest(hashes))


def read_artifact(directory):
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('strategy') != 'candidate_head':
        raise ValueError('unsupported decision artifact')
    if manifest.get('recipe') != RECIPE:
        raise ValueError('unsupported feature/rendering recipe')
    if manifest.get('head') != 'attention_head_v1':
        raise ValueError('unsupported decision head')
    for name in ('head.safetensors', 'head.json'):
        if file_digest(root/name) != manifest['files'][name]:
            raise ValueError(f'artifact integrity mismatch: {name}')
    backbone = manifest['backbone']
    signature = manifest['feature_signature']
    config = json.loads((root/'head.json').read_text())
    if (digest(backbone['files']) != backbone['fingerprint'] or
            signature['model_fingerprint'] != backbone['fingerprint'] or
            signature['recipe'] != RECIPE or signature['limits'] != manifest['limits'] or
            config['hidden'] != signature['hidden'] or
            config['features_meta']['signature_sha256'] != digest(signature)):
        raise ValueError('inconsistent artifact feature contract')
    return manifest


def choice_result(task, logits):
    if len(logits) != len(task.candidates) or not all(math.isfinite(x) for x in logits):
        raise ValueError('invalid decision utilities')
    shifted = [v-max(logits) for v in logits]
    z = sum(math.exp(v) for v in shifted)
    options = [dict(id=c.id, utility=u, probability=math.exp(v)/z)
               for c, u, v in zip(task.candidates, logits, shifted)]
    # Stable ID tie-breaking is output mapping only; IDs never enter the encoder.
    best = min(options, key=lambda o: (-o['utility'], o['id']))
    return dict(id=task.id, strategy='candidate_head', choice_id=best['id'], candidates=options)


def summarize(tasks, results):
    if not tasks or len(tasks) != len(results):
        raise ValueError('expected nonempty aligned results')
    hits = top3 = 0
    nll = brier = 0.
    bins = [[0, 0., 0.] for _ in range(10)]
    for task, result in zip(tasks, results):
        options = result['candidates']
        gold = next(o for o in options if o['id'] == task.choice_id)
        temperature = result.get('calibration', {}).get('temperature', 1.0)
        scores = [o['utility']/temperature for o in options]
        maximum = max(scores)
        # Stable NLL even when a probability underflows.
        nll += maximum + math.log(sum(math.exp(s-maximum) for s in scores)) - gold['utility']/temperature
        hit = result['choice_id'] == task.choice_id
        hits += hit
        ordered = sorted(options, key=lambda o: (-o['utility'], o['id']))
        top3 += task.choice_id in [o['id'] for o in ordered[:3]]
        brier += sum((o['probability']-int(o['id'] == task.choice_id))**2 for o in options)
        confidence = ordered[0]['probability']
        bucket = bins[min(9, int(confidence*10))]
        bucket[0] += 1
        bucket[1] += confidence
        bucket[2] += hit
    n = len(tasks)
    return dict(n=n, accuracy=hits/n, top3=top3/n, nll=nll/n, brier=brier/n,
                ece=sum(abs(sc-sh) for count, sc, sh in bins)/n,
                chance_accuracy=sum(1/len(t.candidates) for t in tasks)/n)
