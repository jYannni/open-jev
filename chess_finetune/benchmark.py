"""Analyze saved chess predictions without loading an LLM."""
import argparse
import json
import math
import random
from pathlib import Path

from .calibration import log_probs, metrics, validate
from .groups import clusters, game_identity

KEYS = ('accuracy', 'nll', 'brier')


def contributions(rows, temperature):
    result = []
    for r in rows:
        lp = log_probs(r['scores'], temperature)
        p = [math.exp(x) for x in lp]
        result.append([float(max(range(len(p)), key=p.__getitem__) == r['label']),
                       -lp[r['label']], sum((v-int(i == r['label']))**2 for i, v in enumerate(p))])
    return result


def percentile(values, q):
    values = sorted(values)
    index = (len(values)-1)*q
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def intervals(rows, values, samples=1000, seed=42):
    """Cluster percentile bootstrap, with position-weighted means in every draw."""
    groups = clusters(rows)
    point = {key: sum(v[j] for v in values)/len(values) for j, key in enumerate(KEYS)}
    if len(groups) < 2 or samples == 0:
        return dict(groups=len(groups), estimates=point, ci95=None)
    rng = random.Random(seed)
    sums = [(len(g), [sum(values[i][j] for i in g) for j in range(len(KEYS))]) for g in groups]
    draws = [[] for _ in KEYS]
    for _ in range(samples):
        selected = [sums[rng.randrange(len(sums))] for _ in sums]
        n = sum(s[0] for s in selected)
        for j in range(len(KEYS)):
            draws[j].append(sum(s[1][j] for s in selected)/n)
    return dict(groups=len(groups), estimates=point,
                ci95={k: [percentile(draws[j], .025), percentile(draws[j], .975)] for j, k in enumerate(KEYS)})


def load_predictions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    validate(rows)
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError(f'{path}: duplicate prediction id {row["id"]}')
        seen.add(row['id'])
        if len(row['candidates']) != len(row['scores']) or len(set(row['candidates'])) != len(row['candidates']):
            raise ValueError(f'{path}: invalid candidates')
    return rows


def enrich_predictions(variants, data_path, expected_hash):
    """Attach provenance only when original test bytes and example identity match."""
    from .data import fingerprint, read_split, validate_board
    if not expected_hash or fingerprint(data_path) != expected_hash:
        raise ValueError('test dataset hash does not match the source run manifest')
    source = {r['id']: r for r in read_split(data_path)}
    for rows in variants.values():
        for row in rows:
            original = source.get(row['id'])
            if original is None or original['candidates'] != row['candidates'] or original['label'] != row['label']:
                raise ValueError('prediction does not match original test example')
            if original['fen'] != row['fen'] and validate_board(original['fen'], original['candidates']) != validate_board(row['fen'], row['candidates']):
                raise ValueError('prediction FEN does not match original test example')
            for key in ('puzzle_id', 'game_id', 'game_url', 'GameUrl', 'rating', 'position'):
                if key in original:
                    row[key] = original[key]


def strata(rows, temperature):
    groups = {}
    for row in rows:
        count = len(row['candidates'])
        bucket = '1' if count == 1 else '2-10' if count <= 10 else '11-25' if count <= 25 else '26+'
        groups.setdefault('candidates:'+bucket, []).append(row)
        rating = row.get('rating')
        if isinstance(rating, (int, float)):
            bucket = '<1000' if rating < 1000 else '1000-1399' if rating < 1400 else '1400-1799' if rating < 1800 else '1800+'
            groups.setdefault('rating:'+bucket, []).append(row)
    return {key: {k: v for k, v in metrics(group, temperature).items()
                  if k in ('n', 'accuracy', 'nll', 'brier', 'ece')} for key, group in sorted(groups.items())}


def analyze(variants, temperatures, samples=1000, seed=42):
    if samples < 0:
        raise ValueError('bootstrap sample count must be nonnegative')
    if not variants:
        raise ValueError('no prediction variants')
    reference = next(iter(variants.values()))
    validate(reference)
    # Enforce paired identity including ordering of candidates before calculating deltas.
    signature = lambda r: (r['id'], r['fen'], r['candidates'], r['label'])
    for name, rows in variants.items():
        validate(rows)
        if [signature(r) for r in rows] != [signature(r) for r in reference]:
            raise ValueError(f'{name}: test examples/candidate ordering do not match')
    outputs, vectors = {}, {}
    for name, rows in variants.items():
        for mode, t in [('raw', 1.), ('calibrated', temperatures[name])]:
            key = name+'/'+mode
            vectors[key] = contributions(rows, t)
            outputs[key] = dict(metrics=metrics(rows, t),
                                uncertainty=intervals(reference, vectors[key], samples, seed),
                                strata=strata(rows, t))
    comparisons = {}
    for name in variants:
        pairs = [(name+'/raw', name+'/calibrated')]
        if name != 'base' and 'base' in variants:
            pairs += [('base/raw', name+'/raw'), ('base/calibrated', name+'/calibrated')]
        for a, b in pairs:
            delta = [[y-x for x, y in zip(va, vb)] for va, vb in zip(vectors[a], vectors[b])]
            comparisons[b+' minus '+a] = intervals(reference, delta, samples, seed)
    counts = [len(r['candidates']) for r in reference]
    return dict(schema_version=1, seed=seed, bootstrap_samples=samples,
                uncertainty_method='95% percentile bootstrap of connected game/puzzle/position groups; temperatures fixed, not refitted. Does not cover training-seed variance.',
                game_metadata_complete=all(game_identity(r) for r in reference),
                uniform_baseline=dict(expected_accuracy=sum(1/n for n in counts)/len(counts),
                                      nll=sum(math.log(n) for n in counts)/len(counts),
                                      brier=sum(1-1/n for n in counts)/len(counts)),
                variants=outputs, paired_differences=comparisons)


def write_report(out, analysis):
    out.mkdir(parents=True, exist_ok=False)
    (out/'benchmark.json').write_text(json.dumps(analysis, indent=2, allow_nan=False)+'\n')
    lines = ['# Strengthened chess benchmark', '',
             'SMOKE TEST: not a research benchmark.' if analysis.get('smoke_test') else 'Saved-prediction evaluation.', '',
             analysis['uncertainty_method'], '',
             'Intervals are conditional on this dataset and fitted models. Missing source-game IDs weaken independence guarantees.', '',
             '| Variant | N | Accuracy | NLL | Brier | ECE |', '|---|---:|---:|---:|---:|---:|']
    for name, result in analysis['variants'].items():
        m = result['metrics']
        lines.append(f"| {name} | {m['n']} | {m['accuracy']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} |")
    b = analysis['uniform_baseline']
    lines += ['', f"Uniform baseline: expected accuracy {b['expected_accuracy']:.4f}, NLL {b['nll']:.4f}, Brier {b['brier']:.4f}.", '', '## Paired differences', '',
              'Accuracy: positive is better. NLL/Brier: negative is better. Intervals crossing zero are inconclusive at this resolution.', '',
              '| Comparison | Metric | Difference | 95% interval |', '|---|---|---:|---|']
    for name, result in analysis['paired_differences'].items():
        for key, value in result['estimates'].items():
            bounds = result['ci95'][key] if result['ci95'] else None
            interval = f'[{bounds[0]:.4f}, {bounds[1]:.4f}]' if bounds else 'Unavailable'
            lines.append(f'| {name} | {key} | {value:.4f} | {interval} |')
    lines += ['', '## Breakdown by difficulty and candidate count', '', '| Variant | Stratum | N | Accuracy | NLL | ECE |', '|---|---|---:|---:|---:|---:|']
    for name, result in analysis['variants'].items():
        for bucket, m in result['strata'].items():
            lines.append(f"| {name} | {bucket} | {m['n']} | {m['accuracy']:.4f} | {m['nll']:.4f} | {m['ece']:.4f} |")
    if 'engine' in analysis:
        lines += ['', '## Engine reference move quality', '', 'Finite-search estimates, not ground truth. Centipawn losses exclude positions with any mate score.', '', '| Model | Positions | CP-eligible | Mean CP loss | Within tolerance |', '|---|---:|---:|---:|---:|']
        for name, m in analysis['engine']['metrics'].items():
            lines.append(f"| {name} | {m['n']} | {m['cp_n']} | {m['mean_cp_loss']} | {m['within_tolerance_rate']} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--data', type=Path, help='Original dataset directory; verified against source hashes before enriching old predictions with ratings/game IDs')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--engine', help='Optional local UCI engine executable; no downloads')
    ap.add_argument('--engine-nodes', type=int, default=50000)
    ap.add_argument('--cp-tolerance', type=int, default=50)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('output directory already exists')
    if args.bootstrap < 0 or args.engine_nodes < 1 or args.cp_tolerance < 0:
        ap.error('invalid bootstrap, node budget, or centipawn tolerance')
    variants, temperatures = {}, {}
    from .data import fingerprint
    sources = {}
    for path in sorted(args.run.glob('*-test-predictions.jsonl')):
        name = path.name.removesuffix('-test-predictions.jsonl')
        variants[name] = load_predictions(path)
        calibrator = args.run/f'{name}-calibrator.json'
        temperatures[name] = json.loads(calibrator.read_text())['temperature']
        sources[path.name] = fingerprint(path)
        sources[calibrator.name] = fingerprint(calibrator)
    manifest_path = args.run/'manifest.json'
    if args.data:
        manifest = json.loads(manifest_path.read_text())
        enrich_predictions(variants, args.data/'test.jsonl', manifest.get('data', {}).get('test'))
        sources['test_dataset'] = fingerprint(args.data/'test.jsonl')
    result = analyze(variants, temperatures, args.bootstrap, args.seed)
    result['source_run'] = str(args.run.resolve())
    result['source_hashes'] = sources
    if manifest_path.exists():
        result['source_manifest_sha256'] = fingerprint(manifest_path)
        result['smoke_test'] = json.loads(manifest_path.read_text()).get('smoke_test', False)
    if args.engine:
        from .engine import evaluate_engine
        result['engine'] = evaluate_engine(variants, args.engine, args.engine_nodes, args.cp_tolerance)
    write_report(args.out, result)
    print(f'Report: {args.out}/report.md')


if __name__ == '__main__':
    main()
