"""Validate legacy chess examples and audit split overlap without loading a model."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from .groups import game_identity, identities


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(path, strict=False):
    rows = []
    with Path(path).open() as f:
        for line, raw in enumerate(f, 1):
            try:
                row = json.loads(raw)
                prompt = row['prompt']
                fen = next(s[5:] for s in prompt.splitlines() if s.startswith('FEN: '))
                moves = next(s[len('Legal moves: '):] for s in prompt.splitlines()
                             if s.startswith('Legal moves: ')).split()
                target = row['completion'].strip()
                if len(fen.split()) != 6 or not moves or len(set(moves)) != len(moves) or target not in moves:
                    raise ValueError('invalid FEN fields, candidate list, or target')
                if strict:
                    fen = validate_board(fen, moves)
                rows.append(dict(row, id=f'{Path(path).stem}:{line}', fen=fen,
                                 position=' '.join(fen.split()[:4]), candidates=moves,
                                 label=moves.index(target)))
            except (ValueError, KeyError, TypeError, AttributeError, StopIteration) as e:
                raise ValueError(f'{path}:{line}: malformed chess example: {e}') from e
    if not rows:
        raise ValueError(f'{path}: empty split')
    return rows


def validate_board(fen, moves):
    try:
        import chess
    except ImportError as exc:
        raise RuntimeError("Strict chess validation requires the finetune extra (python-chess)") from exc
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError("invalid chess position")
    if set(moves) != {m.uci() for m in board.legal_moves}:
        raise ValueError("candidate list must contain exactly all legal moves")
    # Normalize non-actionable en-passant squares for position identity.
    return board.fen(en_passant='legal')


def audit(splits):
    seen, overlaps, duplicates = {}, [], {}
    for split, rows in splits.items():
        counts = Counter(r['position'] for r in rows)
        duplicates[split] = sum(n - 1 for n in counts.values() if n > 1)
        for key in sorted({key for r in rows for key in identities(r)}):
            if key in seen:
                overlaps.append(dict(field=key[0], splits=[seen[key], split], value=key[1]))
            else:
                seen[key] = split
    return dict(counts={k: len(v) for k, v in splits.items()}, overlaps=overlaps,
                duplicate_positions_within_split=duplicates,
                game_metadata_complete=all(game_identity(r) for rows in splits.values() for r in rows),
                position_policy='First four FEN fields; clocks ignored. Strict mode additionally normalizes en-passant legality. Near-duplicates and pretraining contamination are not detected.')
