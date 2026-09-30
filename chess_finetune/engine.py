"""Optional finite-search UCI reference. Mate scores never become centipawns."""
import shutil
from pathlib import Path

from .data import fingerprint, validate_board


def move_quality(rows, references, tolerance):
    losses, ranks, best_matches = [], [], []
    for row in rows:
        ref = references[row['fen']]
        chosen = row['candidates'][max(range(len(row['scores'])), key=row['scores'].__getitem__)]
        ranks.append(ref['moves'][chosen]['rank'])
        best_matches.append(ref['moves'][chosen]['rank'] == 1)
        if all(value['cp'] is not None for value in ref['moves'].values()):
            best = max(value['cp'] for value in ref['moves'].values())
            losses.append(best-ref['moves'][chosen]['cp'])
    return dict(n=len(rows), cp_n=len(losses), mate_positions=len(rows)-len(losses),
                mean_cp_loss=sum(losses)/len(losses) if losses else None,
                within_tolerance_rate=sum(x <= tolerance for x in losses)/len(losses) if losses else None,
                engine_best_match_rate=sum(best_matches)/len(rows), mean_rank=sum(ranks)/len(rows))


def evaluate_engine(variants, executable, nodes=50000, tolerance=50):
    import chess
    import chess.engine
    path = shutil.which(executable)
    if not path:
        raise ValueError(f'UCI engine executable not found: {executable}')
    rows = next(iter(variants.values()))
    references = {}
    with chess.engine.SimpleEngine.popen_uci(path, timeout=60) as engine:
        options = {key: value for key, value in {'Threads': 1, 'Hash': 64}.items() if key in engine.options}
        engine.configure(options)
        identity = dict(engine.id)
        for row in rows:
            if row['fen'] in references:
                continue
            validate_board(row['fen'], row['candidates'])
            board = chess.Board(row['fen'])
            infos = engine.analyse(board, chess.engine.Limit(nodes=nodes),
                                   multipv=len(row['candidates']), game=object(),
                                   root_moves=[chess.Move.from_uci(m) for m in row['candidates']])
            values = {info['pv'][0].uci(): info['score'].pov(board.turn) for info in infos}
            if set(values) != set(row['candidates']):
                raise ValueError('engine did not return all candidate moves; increase node budget')
            references[row['fen']] = dict(moves={move: dict(cp=value.score(), mate=value.mate(),
                rank=1+sum(other > value for other in values.values())) for move, value in values.items()})
            if len(references) % 25 == 0:
                print(f'Engine analyzed {len(references)} positions', flush=True)
    return dict(identity=identity, executable=str(Path(path).resolve()), sha256=fingerprint(path),
                options=options, node_budget=nodes, cp_tolerance=tolerance,
                score_perspective='Side to move in original position',
                metrics={name: move_quality(group, references, tolerance) for name, group in variants.items()},
                positions=references)
