"""Build new, group-disjoint four-way splits from legacy chess JSONL files."""
import argparse
import json
import random
from pathlib import Path

from .data import audit, fingerprint, read_split
from .groups import clusters, game_identity


def split_rows(rows, seed=42, fractions=(.7, .1, .1, .1), require_games=False):
    if len(fractions) != 4 or any(f <= 0 for f in fractions) or abs(sum(fractions)-1) > 1e-9:
        raise ValueError('four positive fractions must sum to one')
    if require_games and any(not game_identity(r) for r in rows):
        raise ValueError('source-game metadata is required for every example')
    # Keep the first occurrence, but reject contradictory labels rather than selecting one.
    unique, removed = {}, 0
    for index, row in enumerate(rows):
        key = row['position']
        if key in unique:
            previous = unique[key][1]
            if previous['candidates'][previous['label']] != row['candidates'][row['label']]:
                raise ValueError(f'conflicting reference labels for position {key}')
            removed += 1
        else:
            unique[key] = (index, row)
    # Group before deduplication so duplicate positions cannot erase a game/puzzle link.
    groups = clusters(rows)
    first_indices = {index for index, row in unique.values()}
    groups = [[i for i in g if i in first_indices] for g in groups]
    groups = [g for g in groups if g]
    if len(groups) < 4:
        raise ValueError('need at least four independent groups to make four nonempty splits')
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    names = ('train', 'valid', 'calibration', 'test')
    result = {name: [] for name in names}
    total = len(unique)
    targets = dict(zip(names, [total*f for f in fractions]))
    for offset, group in enumerate(groups):
        empty = [name for name in names if not result[name]]
        eligible = empty if len(groups)-offset == len(empty) else names
        name = max(eligible, key=lambda n: targets[n]-len(result[n]))
        result[name].extend(rows[i] for i in group)
    for subset in result.values():
        rng.shuffle(subset)
    return result, dict(removed_duplicate_positions=removed, independent_groups=len(groups))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fractions', type=float, nargs=4, default=(.7, .1, .1, .1))
    ap.add_argument('--require-game-metadata', action='store_true')
    args = ap.parse_args()
    rows = [row for path in args.input for row in read_split(path, strict=True)]
    result, details = split_rows(rows, args.seed, args.fractions, args.require_game_metadata)
    review = audit(result)
    if review['overlaps']:
        raise ValueError('internal error: split groups overlap')
    args.out.mkdir(parents=True, exist_ok=False)
    for name, subset in result.items():
        with (args.out/f'{name}.jsonl').open('w') as stream:
            for row in subset:
                # Preserve provenance and input text; derived fields are recomputed on read.
                export = {k: v for k, v in row.items() if k not in ('id', 'fen', 'position', 'candidates', 'label')}
                stream.write(json.dumps(export)+'\n')
    manifest = dict(seed=args.seed, requested_fractions=args.fractions,
                    sources={str(p.resolve()): fingerprint(p) for p in args.input},
                    outputs={name: fingerprint(args.out/f'{name}.jsonl') for name in result},
                    audit=review, **details,
                    warning='New split assignments require retraining from the base model. An adapter trained on the source pool is contaminated for these test splits.')
    (args.out/'dataset-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
