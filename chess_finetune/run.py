"""Run with python -m chess_finetune.run --help."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path

from .calibration import fit_temperature, log_probs, metrics
from .data import audit, fingerprint, read_split


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def artifact_identity(path):
    p = Path(path)
    if not p.is_dir():
        return dict(source=path, note='Non-local model ID; pin a revision externally for reproducibility.')
    # Hash weights as well as config so replacing an adapter cannot silently reuse identity.
    files = sorted(f for f in p.rglob('*') if f.is_file() and
                   f.suffix in ('.json', '.safetensors', '.model', '.txt'))
    hashes = {}
    for f in files:
        h = hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        hashes[str(f.relative_to(p))] = h.hexdigest()
    return dict(source=str(p.resolve()), files=hashes)


def score(scorer, rows):
    result = []
    for i, row in enumerate(rows):
        started = time.perf_counter()
        candidates = [' ' + m for m in row['candidates']]
        values = scorer.score(row['prompt'], candidates, norm='sum') if len(candidates) > 1 else None
        result.append(dict(id=row['id'], puzzle_id=row.get('puzzle_id'), fen=row['fen'],
                           game_id=row.get('game_id'), game_url=row.get('game_url', row.get('GameUrl')),
                           rating=row.get('rating'), position=row['position'],
                           candidates=row['candidates'], label=row['label'],
                           scores=[v.score for v in values] if values else [0.],
                           seconds=time.perf_counter() - started))
        if (i + 1) % 25 == 0:
            print(f'  scored {i+1}/{len(rows)}', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('data/chess'))
    ap.add_argument('--model', default='models/gemma-3-4b-it')
    ap.add_argument('--adapter', default='adapters/chess-lora')
    ap.add_argument('--base-only', action='store_true')
    ap.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='Smoke test only: first N rows of each evaluation split')
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--strict-data', action='store_true', help='Validate board legality and complete move lists (requires python-chess)')
    ap.add_argument('--require-game-metadata', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=1000, help='Cluster bootstrap samples; 0 disables intervals')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.bootstrap < 0:
        ap.error('limit/bootstrap must be nonnegative and batch-size positive')
    names = ['train', 'valid', 'test']
    fit_split = 'calibration' if (args.data/'calibration.jsonl').exists() else 'valid'
    if fit_split == 'calibration':
        names.append('calibration')
    splits = {name: read_split(args.data / f'{name}.jsonl', strict=args.strict_data) for name in names}
    audit_result = audit(splits)
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'split-audit.json', audit_result)
    if audit_result['overlaps']:
        raise SystemExit(f'Split overlap detected; see {args.out}/split-audit.json. Repair dataset before evaluation.')
    if args.require_game_metadata and not audit_result['game_metadata_complete']:
        raise SystemExit('Source-game metadata is required but missing; see split-audit.json.')
    if not audit_result['game_metadata_complete']:
        print('Warning: source-game metadata is missing; game-level independence cannot be verified.', flush=True)
    if args.audit_only:
        print(f'Split audit written to {args.out}')
        return
    versions = {}
    for package in ('mlx', 'mlx-lm', 'transformers', 'openjev'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    manifest = dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    target='Exact match to reference puzzle solution; not win probability',
                    fit_split=fit_split, normalization='sum', versions=versions, platform=platform.platform(),
                    python=platform.python_version(), git_revision=git.stdout.strip(),
                    dirty_worktree=bool(dirty.stdout),
                    data={name: fingerprint(args.data / f'{name}.jsonl') for name in splits},
                    model=artifact_identity(args.model),
                    adapter=None if args.base_only else artifact_identity(args.adapter),
                    smoke_test=bool(args.limit))
    save(args.out / 'manifest.json', manifest)
    from openjev.scorer import OptionScorer
    summary, benchmark_variants, temperatures = {}, {}, {}
    variants = [('base', None)] + ([] if args.base_only else [('lora', args.adapter)])
    for name, adapter in variants:
        print(f'Loading {name}', flush=True)
        scorer = OptionScorer(args.model, batch_size=args.batch_size, adapter_path=adapter)
        selected = {key: (splits[key][:args.limit] if args.limit else splits[key]) for key in (fit_split, 'test')}
        # Warm up outside timed records.
        warm = next((r for r in selected[fit_split] if len(r['candidates']) > 1), None)
        if warm:
            scorer.score(warm['prompt'], [' ' + m for m in warm['candidates']], norm='sum')
        validation = score(scorer, selected[fit_split])
        temperature = fit_temperature(validation)
        save(args.out / f'{name}-calibrator.json', dict(temperature=temperature,
             bounds=[.05, 20.], fit_split=fit_split, fit_n=len(validation),
             validation_sha256=manifest['data'][fit_split], normalization='sum', model=name,
             manifest='manifest.json', objective='categorical negative log likelihood'))
        test = score(scorer, selected['test'])
        for split, rows in [(fit_split, validation), ('test', test)]:
            with (args.out / f'{name}-{split}-predictions.jsonl').open('w') as stream:
                for r in rows:
                    record = dict(r, probabilities=[math.exp(x) for x in log_probs(r['scores'])],
                                  calibrated_probabilities=[math.exp(x) for x in log_probs(r['scores'], temperature)])
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
        benchmark_variants[name] = test
        temperatures[name] = temperature
        summary[name] = dict(raw=metrics(test), calibrated=metrics(test, temperature),
                             fit_split=fit_split, fit_raw_nll=metrics(validation)['nll'],
                             fit_calibrated_nll=metrics(validation, temperature)['nll'])
        if fit_split == 'valid':
            summary[name]['validation_raw_nll'] = summary[name]['fit_raw_nll']
            summary[name]['validation_calibrated_nll'] = summary[name]['fit_calibrated_nll']
        nonforced = [r for r in test if len(r['scores']) > 1]
        if nonforced:
            summary[name]['nonforced'] = dict(raw=metrics(nonforced), calibrated=metrics(nonforced, temperature))
        del scorer
        import gc
        gc.collect()
        save(args.out / 'metrics.json', summary)
    lines = ['# Chess calibration evaluation', '', manifest['target'], '',
             'Smoke test; not a benchmark.' if args.limit else f'Temperature fitted on {fit_split} only.', '',
             '| Model | Temperature | Accuracy | NLL | Brier | ECE |',
             '|---|---:|---:|---:|---:|---:|']
    for name, result in summary.items():
        for mode in ('raw', 'calibrated'):
            m = result[mode]
            lines.append(f"| {name} {mode} | {m['temperature']:.4f} | {m['accuracy']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} |")
    lines += ['', 'Brier is the sum over candidate classes, averaged over positions (range 0–2). ECE uses 10 equal-width top-confidence bins. Forced moves are included above; nonforced-only metrics are in metrics.json.', '', '## Reliability bins', '']
    for name, result in summary.items():
        for mode in ('raw', 'calibrated'):
            lines += [f'### {name} {mode}', '', '| Confidence interval | Count | Mean confidence | Accuracy |', '|---|---:|---:|---:|']
            for b in result[mode]['reliability']:
                if b['count']:
                    lines.append(f"| {b['lower']:.1f}–{b['upper']:.1f} | {b['count']} | {b['confidence']:.4f} | {b['accuracy']:.4f} |")
            lines.append('')
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    from .benchmark import analyze, write_report
    benchmark = analyze(benchmark_variants, temperatures, args.bootstrap, args.seed)
    benchmark['smoke_test'] = bool(args.limit)
    benchmark['source_manifest_sha256'] = fingerprint(args.out/'manifest.json')
    write_report(args.out/'benchmark', benchmark)
    print(f'Results: {args.out}/report.md; strengthened report: {args.out}/benchmark/report.md')


if __name__ == '__main__':
    main()
