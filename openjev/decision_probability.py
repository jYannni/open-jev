"""Probability evaluation and temperature fitting; no model backend required."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .decision import choice_result, digest, provenance


def aligned_logits(tasks, results):
    """Require exact example/candidate alignment; never silently zip away rows."""
    if not tasks or len(tasks) != len(results):
        raise ValueError('expected nonempty aligned labelled tasks and predictions')
    logits, labels = [], []
    for task, result in zip(tasks, results):
        ids = [c.id for c in task.candidates]
        values = result['candidates']
        if (task.choice_id not in ids or result['id'] != task.id or
                len(values) != len(ids) or {v['id'] for v in values} != set(ids)):
            raise ValueError('prediction/label alignment mismatch')
        by_id = {v['id']: v['utility'] for v in values}
        row = np.array([by_id[i] for i in ids], dtype=np.float64)
        if not np.isfinite(row).all():
            raise ValueError('utilities must be finite')
        logits.append(row)
        labels.append(ids.index(task.choice_id))
    return logits, np.array(labels)


def distribution(logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    x = (np.asarray(logits, dtype=np.float64) - np.max(logits)) / temperature
    logp = x - np.log(np.exp(x).sum())
    return np.exp(logp), logp


def _objective(logits, labels, temperature, objective):
    total = 0.0
    for row, label in zip(logits, labels):
        p, logp = distribution(row, temperature)
        total += -logp[label] if objective == 'nll' else float((p*p).sum() - 2*p[label] + 1)
    return total / len(labels)


def fit_temperature(logits, labels, objective='nll', bounds=(0.05, 20.0)):
    """Bounded log-temperature grid with local golden-section refinements.

    Always considers T=1 and both endpoints; fitting cannot worsen its own
    calibration-split objective. This is not a held-out improvement guarantee.
    """
    if objective not in ('nll', 'brier'):
        raise ValueError('temperature objective must be nll or brier')
    if (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or
            not 0 < bounds[0] <= 1 <= bounds[1] or bounds[0] == bounds[1]):
        raise ValueError('temperature bounds must be finite, positive, ordered, and contain 1')
    if len(logits) != len(labels) or not len(labels):
        raise ValueError('expected nonempty logits and labels')
    for row, label in zip(logits, labels):
        if (np.asarray(row).ndim != 1 or len(row) < 2 or not np.isfinite(row).all() or
                not isinstance(label, (int, np.integer)) or not 0 <= label < len(row)):
            raise ValueError('invalid logits or labels')
    grid = sorted(set(np.linspace(math.log(bounds[0]), math.log(bounds[1]), 65).tolist() + [0.0]))
    def score(log_t):
        return _objective(logits, labels, math.exp(log_t), objective)
    values = [score(x) for x in grid]
    candidates = list(zip(values, grid))
    ratio = (math.sqrt(5)-1)/2
    for i in range(1, len(grid)-1):
        if values[i] > values[i-1] or values[i] > values[i+1]:
            continue
        # Flat logits need no refinement.
        if values[i] == values[i-1] == values[i+1]:
            continue
        a, b = grid[i-1], grid[i+1]
        c, d = b-ratio*(b-a), a+ratio*(b-a)
        fc, fd = score(c), score(d)
        for _ in range(48):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b-ratio*(b-a)
                fc = score(c)
            else:
                a, c, fc = c, d, fd
                d = a+ratio*(b-a)
                fd = score(d)
        candidates.extend([(fc, c), (fd, d)])
    value, log_t = min(candidates, key=lambda pair: (pair[0], abs(pair[1])))
    temperature = min(bounds[1], max(bounds[0], math.exp(log_t)))
    return dict(temperature=temperature, objective=objective, bounds=list(bounds),
                raw_objective=score(0.0), fitted_objective=value,
                at_bound=any(math.isclose(temperature, b, rel_tol=1e-6) for b in bounds),
                optimizer='log_grid_65_golden_48_identity_and_endpoints')


class TemperatureCalibrator:
    """A scalar temperature bound to one exact head artifact and fitting split."""

    def __init__(self, metadata):
        self.metadata = metadata

    @classmethod
    def fit(cls, tasks, results, artifact_id, objective='nll', bounds=(0.05, 20.0)):
        if not artifact_id or any(r.get('artifact_id') != artifact_id or 'calibration' in r for r in results):
            raise ValueError('fit requires raw predictions from the specified artifact')
        logits, labels = aligned_logits(tasks, results)
        fitted = fit_temperature(logits, labels, objective, bounds)
        return cls(dict(schema_version=1, kind='scalar_temperature', artifact_id=artifact_id,
                        split='calibration', examples=len(tasks), provenance=provenance(tasks),
                        prediction_sha256=digest(results), brier_convention='sum_over_valid_candidates', **fitted))

    def save(self, path):
        with Path(path).open('x') as stream:
            json.dump(self.metadata, stream, indent=2, allow_nan=False)
            stream.write('\n')

    @classmethod
    def load(cls, path, artifact_id):
        value = json.loads(Path(path).read_text())
        if (value.get('schema_version') != 1 or value.get('kind') != 'scalar_temperature' or
                value.get('artifact_id') != artifact_id or value.get('split') != 'calibration' or
                value.get('objective') not in ('nll', 'brier') or
                value.get('brier_convention') != 'sum_over_valid_candidates'):
            raise ValueError('incompatible calibrator artifact')
        try:
            t, bounds = value['temperature'], value['bounds']
            valid = (math.isfinite(t) and len(bounds) == 2 and
                     all(math.isfinite(v) for v in bounds) and 0 < bounds[0] <= t <= bounds[1] and
                     bounds[0] <= 1 <= bounds[1] and bounds[0] < bounds[1] and
                     all(isinstance(value['provenance'][key], list) for key in ('ids', 'inputs', 'groups')))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('invalid calibrator temperature, bounds, or provenance')
        return cls(value)

    def apply(self, result):
        if result.get('artifact_id') != self.metadata['artifact_id'] or 'calibration' in result:
            raise ValueError('calibrator requires raw scores from its bound artifact')
        result = deepcopy(result)
        probs, _ = distribution([c['utility'] for c in result['candidates']], self.metadata['temperature'])
        for c, p in zip(result['candidates'], probs):
            c['raw_probability'], c['probability'] = c['probability'], float(p)
        result['calibration'] = {k: self.metadata[k] for k in ('temperature', 'objective')}
        result['calibration']['calibrator_id'] = digest(self.metadata)
        return result


def _rows(tasks, results):
    logits, labels = aligned_logits(tasks, results)
    rows = []
    for task, result, row, label in zip(tasks, results, logits, labels):
        p, logp = distribution(row, result.get('calibration', {}).get('temperature', 1.0))
        best = min(range(len(row)), key=lambda i: (-row[i], task.candidates[i].id))
        rows.append([float(best == label), -float(logp[label]),
                     float((p*p).sum()-2*p[label]+1), float(p[best])])
    return np.array(rows)


def _metrics(rows):
    hits, nll, brier, confidence = rows.T
    bucket = np.minimum(9, (confidence*10).astype(int))
    ece = sum(abs(float((confidence-hits)[bucket == b].sum())) for b in range(10))/len(rows)
    return dict(n=len(rows), accuracy=float(hits.mean()), nll=float(nll.mean()),
                brier=float(brier.mean()), ece=ece)


def probability_report(tasks, results, bootstrap_samples=1000, seed=7):
    """10-bin reliability, confidence-threshold coverage, cluster percentile CIs."""
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError('bootstrap_samples must be a nonnegative integer')
    rows = _rows(tasks, results)
    report = _metrics(rows)
    bins = np.minimum(9, (rows[:, 3]*10).astype(int))
    report['reliability'] = [dict(lower=b/10, upper=(b+1)/10, n=int((bins == b).sum()),
                                 confidence=float(rows[bins == b, 3].mean()) if (bins == b).any() else None,
                                 accuracy=float(rows[bins == b, 0].mean()) if (bins == b).any() else None)
                             for b in range(10)]
    report['selective'] = []
    for coverage in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        threshold = float(np.sort(rows[:, 3])[::-1][math.ceil(coverage*len(rows))-1])
        accepted = rows[:, 3] >= threshold
        report['selective'].append(dict(requested_coverage=coverage, threshold=threshold,
            coverage=float(accepted.mean()), n=int(accepted.sum()), error=float(1-rows[accepted, 0].mean())))
    strata = {'candidate_count': {}, 'question': {}}
    for name, keys in [('candidate_count', [str(len(t.candidates)) for t in tasks]),
                       ('question', [t.question for t in tasks])]:
        for key in sorted(set(keys)):
            strata[name][key] = _metrics(rows[np.array([k == key for k in keys])])
    report['strata'] = strata
    # Prefix types to keep an ungrouped example ID from colliding with a group ID.
    keys = [('group', t.group_id) if t.group_id is not None else ('example', t.id) for t in tasks]
    unique = sorted(set(keys))
    groups = [np.array([i for i, k in enumerate(keys) if k == key]) for key in unique]
    intervals = None
    if len(groups) >= 2 and bootstrap_samples:
        rng = np.random.default_rng(seed)
        draws = {k: [] for k in ('accuracy', 'nll', 'brier', 'ece')}
        for _ in range(bootstrap_samples):
            sample = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            metrics = _metrics(rows[sample])
            for k in draws:
                draws[k].append(metrics[k])
        intervals = {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in draws.items()}
    report['uncertainty'] = dict(method='group_bootstrap_percentile', confidence=0.95,
        samples=bootstrap_samples, seed=seed, groups=len(groups), intervals=intervals,
        fallback='ungrouped examples are singleton clusters',
        limitation='sampling uncertainty only; does not include training-seed uncertainty')
    return report


def baseline_predictions(train_tasks, tasks):
    """Uniform and add-one-smoothed chosen-text prior, restricted to each request."""
    counts = Counter(next(c.text for c in t.candidates if c.id == t.choice_id) for t in train_tasks)
    return {
        'uniform': [choice_result(t, [0.0]*len(t.candidates)) for t in tasks],
        'train_text_prior': [choice_result(t, [math.log(counts[c.text]+1) for c in t.candidates]) for t in tasks],
    }
