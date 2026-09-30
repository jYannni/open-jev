"""CLI for the native choice contract; backend imports stay lazy."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

from .decision import (Candidate, ChoiceTask, check_disjoint, choice_result, file_digest,
                       load_tasks, provenance, read_artifact, render_context, summarize)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(args):
    if args.action == 'experiment':
        from .decision_experiment import run_experiment
        print(json.dumps(run_experiment(args.config), indent=2))
        return
    if args.action == 'convert':
        # Explicit bridge for the existing {context, options, label} datasets.
        if args.out.exists():
            raise ValueError('output already exists')
        rows = []
        with args.data.open() as stream:
            for i, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                options, label = value['options'], value['label']
                if not isinstance(options, list) or type(label) is not int or not 0 <= label < len(options):
                    raise ValueError(f'{args.data}:{i}: invalid options/label')
                candidates = [dict(id=f'option-{j}', text=text) for j, text in enumerate(options)]
                task = ChoiceTask.parse(dict(schema_version=1, id=f'{args.prefix}:{i}',
                    state=value['context'], question=args.question, candidates=candidates,
                    target=dict(choice_id=candidates[label]['id'])), labelled=True)
                rows.append(task.to_dict())
        if not rows:
            raise ValueError('empty input')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            stream.writelines(json.dumps(row)+'\n' for row in rows)
        print(json.dumps(dict(examples=len(rows), out=str(args.out))))
        return
    from .decision_mlx import DecisionScorer, extract_features, train_artifact
    if args.action == 'features':
        result = extract_features(args.data, args.model, args.out, args.split, args.batch_size,
                                  args.context_tokens, args.candidate_tokens)
        print(json.dumps({k: result[k] for k in ('n', 'hidden', 'split', 'signature_sha256', 'seconds')}))
        return
    if args.action == 'train':
        print(json.dumps(train_artifact(args.train, args.validation, args.out, args.rank,
                                       args.epochs, args.batch_size, args.learning_rate, args.seed,
                                       args.brier_weight, args.selection)))
        return
    if args.action == 'score':
        task = ChoiceTask.parse(json.loads(args.request.read_text()))
        scorer = DecisionScorer(args.artifact, args.model, args.batch_size, args.calibrator)
        print(json.dumps(scorer.score(task), indent=2))
        return
    if args.limit < 0:
        raise ValueError('limit cannot be negative')
    tasks = load_tasks(args.data, labelled=args.action == 'eval')
    if args.action == 'eval' and args.out.exists():
        raise ValueError('evaluation output directory already exists')
    if args.action == 'eval':
        manifest = read_artifact(args.artifact)
        for split in ('train', 'validation'):
            check_disjoint(provenance(tasks), manifest['training'][split])
        if args.calibrator:
            from .decision_probability import TemperatureCalibrator
            calibrator = TemperatureCalibrator.load(args.calibrator, file_digest(args.artifact/'manifest.json'))
            check_disjoint(provenance(tasks), calibrator.metadata['provenance'])
    if args.action == 'check' and (not math.isfinite(args.tolerance) or args.tolerance <= 0):
        raise ValueError('tolerance must be positive and finite')
    scorer = DecisionScorer(args.artifact, args.model, args.batch_size, args.calibrator)
    if args.limit:
        tasks = tasks[:args.limit]
    if args.action == 'check':
        worst = 0.
        for i, task in enumerate(tasks):
            original = scorer.score(task)
            # Interleave an unrelated context, then revisit the original request.
            scorer.score(replace(task, state={'control': i}, question='Select a candidate.'))
            repeated = scorer.score(task)
            reversed_result = scorer.score(replace(task, candidates=tuple(reversed(task.candidates))))
            renamed = scorer.score(replace(task, candidates=tuple(Candidate(f'new-{j}', c.text)
                                   for j, c in enumerate(task.candidates)), choice_id=None))
            expected = {c['id']: c for c in original['candidates']}
            for result in (repeated, reversed_result):
                if result['choice_id'] != original['choice_id']:
                    raise ValueError('decision check failed: selected ID changed')
                for c in result['candidates']:
                    worst = max(worst, *(abs(c[k]-expected[c['id']][k]) for k in ('utility','probability')))
            for a, b in zip(original['candidates'], renamed['candidates']):
                worst = max(worst, *(abs(a[k]-b[k]) for k in ('utility','probability')))
        print(json.dumps(dict(examples=len(tasks), max_absolute_difference=worst, tolerance=args.tolerance,
                              passed=worst <= args.tolerance)))
        if worst > args.tolerance:
            raise SystemExit(1)
        return
    predictions = []
    baselines = []
    for task in tasks:
        predictions.append(scorer.score(task))
        if args.baseline:
            started = time.perf_counter()
            scores = scorer.fx.s.score(render_context(task)+'\n\nAnswer:',
                                       [' '+c.text for c in task.candidates], norm='sum')
            result = choice_result(task, [s.score for s in scores])
            result.update(strategy='continuation_sum', seconds=time.perf_counter()-started)
            baselines.append(result)
    report = dict(schema_version=1, artifact_id=scorer.artifact_id,
                  data_sha256=file_digest(args.data), examples=len(tasks), smoke_test=bool(args.limit),
                  head=summarize(tasks, predictions),
                  baseline=summarize(tasks, baselines) if baselines else None,
                  baseline_contract='sum of token log probabilities; same state/question, leading-space candidate text',
                  probability_semantics='candidate-choice distribution, not calibrated outcome confidence',
                  dataset_overlap_check='IDs, semantic input hashes, and supplied group IDs against train/validation')
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/'metrics.json', report)
    save(args.out/'manifest.json', dict(artifact=scorer.manifest, data=str(args.data.resolve()),
                                       data_sha256=report['data_sha256'], limit=args.limit,
                                       batch_size=args.batch_size, baseline=args.baseline))
    with (args.out/'predictions.jsonl').open('x') as stream:
        for i, (task, result) in enumerate(zip(tasks, predictions)):
            stream.write(json.dumps(dict(id=task.id, target=task.choice_id, head=result,
                                          baseline=baselines[i] if baselines else None))+'\n')
    print(json.dumps(report, indent=2))


def register(subparsers):
    p = subparsers.add_parser('decision', help='native candidate-head features, training, and inference (MLX)')
    commands = p.add_subparsers(dest='action', required=True)
    experiment = commands.add_parser('experiment', help='four-split CE/Brier and temperature-scaling ablation')
    experiment.add_argument('--config', type=Path, required=True)
    c = commands.add_parser('convert', help='explicitly convert legacy context/options/label JSONL')
    c.add_argument('data', type=Path)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--prefix', required=True, help='unique ID prefix for this split')
    c.add_argument('--question', default='Select the best candidate for the state.')
    f = commands.add_parser('features', help='extract frozen features for native choice JSONL')
    f.add_argument('data', type=Path)
    f.add_argument('--model', required=True)
    f.add_argument('--out', type=Path, required=True)
    f.add_argument('--split', choices=('train','validation','calibration','test'), required=True)
    f.add_argument('--batch-size', type=int, default=8)
    f.add_argument('--context-tokens', type=int, default=4096)
    f.add_argument('--candidate-tokens', type=int, default=256)
    t = commands.add_parser('train', help='train a frozen-feature attention head with choice cross-entropy')
    t.add_argument('train', type=Path)
    t.add_argument('--validation', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--rank', type=int, default=256)
    t.add_argument('--epochs', type=int, default=8)
    t.add_argument('--batch-size', type=int, default=64)
    t.add_argument('--learning-rate', type=float, default=1e-4)
    t.add_argument('--seed', type=int, default=7)
    t.add_argument('--brier-weight', type=float, default=0.0)
    t.add_argument('--selection', choices=('top1', 'nll', 'brier'), default='top1')
    for name in ('score', 'eval', 'check'):
        cmd = commands.add_parser(name)
        cmd.add_argument('artifact', type=Path)
        cmd.add_argument('--model', help='relocated identical backbone directory; content hashes must match')
        cmd.add_argument('--batch-size', type=int, default=8)
        cmd.add_argument('--calibrator', type=Path, help='temperature artifact bound to this head')
        if name == 'score':
            cmd.add_argument('--request', type=Path, required=True, help='one native JSON task')
        else:
            cmd.add_argument('--data', type=Path, required=True)
            cmd.add_argument('--limit', type=int, default=3 if name == 'check' else 0)
        if name == 'eval':
            cmd.add_argument('--out', type=Path, required=True)
            cmd.add_argument('--baseline', action='store_true', help='also evaluate zero-shot continuation scoring')
        if name == 'check':
            cmd.add_argument('--tolerance', type=float, default=0.005)
    p.set_defaults(fn=run)
