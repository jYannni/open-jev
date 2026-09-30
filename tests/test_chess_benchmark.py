import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from chess_finetune.benchmark import analyze, intervals, write_report, enrich_predictions
from chess_finetune.data import audit, read_split
from chess_finetune.engine import move_quality
from chess_finetune.groups import clusters, game_identity
from chess_finetune.prepare import split_rows

try:
    import chess
    import chess.engine
except ImportError:
    chess = None


def row(i, **extra):
    return dict(id=str(i), fen=f'position-{i}', position=f'position-{i}',
                puzzle_id=f'puzzle-{i}', candidates=['a', 'b'], label=0,
                scores=[math.log(.8), math.log(.2)], prompt='example', completion='a', **extra)


class BenchmarkTests(unittest.TestCase):
    def test_group_aliases_and_transitive_links(self):
        a, b, c = row(0), row(1), row(2)
        a['game_url'] = 'https://lichess.org/abcdefgh1234/black#12'
        b['game_id'] = 'abcdefgh'
        c['puzzle_id'] = b['puzzle_id']
        self.assertEqual(game_identity(a), 'abcdefgh')
        self.assertEqual(clusters([a,b,c]), [[0,1,2]])
        self.assertEqual(audit({'train':[a], 'test':[b]})['overlaps'][0]['field'], 'game')

    def test_dedup_split_and_reproducibility(self):
        rows = [row(i, game_id=f'game{i//2}') for i in range(20)]
        rows += [rows[0]]
        result, meta = split_rows(rows, seed=13)
        repeated, _ = split_rows(rows, seed=13)
        self.assertEqual(result, repeated)
        self.assertEqual(meta['removed_duplicate_positions'], 1)
        self.assertEqual(sum(map(len,result.values())),20)
        self.assertFalse(audit(result)['overlaps'])
        self.assertTrue(all(result.values()))

    def test_duplicate_bridge_does_not_split_games(self):
        rows = [row(i, game_id=f'game{i}') for i in range(12)]
        bridge = dict(rows[0], game_id='game1')
        result, _ = split_rows(rows+[bridge])
        names = {r['id']: name for name, group in result.items() for r in group}
        self.assertEqual(names['0'], names['1'])

    def test_split_rejects_conflicts_and_missing_metadata(self):
        rows = [row(i) for i in range(8)]
        with self.assertRaisesRegex(ValueError, 'metadata'):
            split_rows(rows, require_games=True)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            split_rows(rows+[dict(rows[0], label=1)])
        with self.assertRaisesRegex(ValueError, 'fractions'):
            split_rows(rows, fractions=(.7,.1,.1,.2))

    def test_paired_intervals_and_uniform_baseline(self):
        base = [row(i) for i in range(6)]
        better = copy.deepcopy(base)
        for r in base:
            r['scores'] = [-2, 0]
        report = analyze({'base':base, 'lora':better}, {'base':1, 'lora':1}, samples=100, seed=4)
        diff = report['paired_differences']['lora/raw minus base/raw']
        self.assertEqual(diff['ci95']['accuracy'], [1,1])
        self.assertEqual(report['uniform_baseline']['expected_accuracy'], .5)
        self.assertAlmostEqual(report['uniform_baseline']['nll'], math.log(2))
        self.assertEqual(report['uniform_baseline']['brier'], .5)
        self.assertEqual(report, analyze({'base':base, 'lora':better}, {'base':1, 'lora':1},100,4))
        with tempfile.TemporaryDirectory() as d:
            write_report(Path(d)/'out', report)
            self.assertIn('Paired differences', (Path(d)/'out/report.md').read_text())

    def test_pair_mismatch_rejected(self):
        a, b = [row(0)], [row(1)]
        with self.assertRaisesRegex(ValueError, 'do not match'):
            analyze({'base':a,'lora':b}, {'base':1,'lora':1})

    def test_metadata_enrichment_requires_matching_hash_and_examples(self):
        from chess_finetune.data import fingerprint
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'test.jsonl'
            path.write_text(json.dumps(dict(prompt='FEN: x w - - 0 1\nLegal moves: a b', completion='a', rating=1200))+'\n')
            rows = [dict(id='test:1', fen='x w - - 0 1', candidates=['a','b'], label=0, scores=[1.,0.])]
            with self.assertRaisesRegex(ValueError, 'hash'):
                enrich_predictions({'base':rows}, path, 'wrong')
            enrich_predictions({'base':rows}, path, fingerprint(path))
            self.assertEqual(rows[0]['rating'],1200)
            rows[0]['label'] = 1
            with self.assertRaisesRegex(ValueError, 'does not match'):
                enrich_predictions({'base':rows}, path, fingerprint(path))

    def test_single_cluster_has_no_interval(self):
        rows = [row(i, game_id='same') for i in range(4)]
        result = intervals(rows, [[1,0,0]]*4, samples=20)
        self.assertEqual(result['groups'],1)
        self.assertIsNone(result['ci95'])

    def test_mate_scores_not_counted_as_cp_loss(self):
        rows = [row(0), row(1)]
        refs = {'position-0': {'moves': {'a':{'cp':10,'rank':2}, 'b':{'cp':100,'rank':1}}},
                'position-1': {'moves': {'a':{'cp':None,'rank':1}, 'b':{'cp':100,'rank':2}}}}
        m = move_quality(rows, refs, 50)
        self.assertEqual(m['cp_n'],1)
        self.assertEqual(m['mean_cp_loss'],90)
        self.assertEqual(m['within_tolerance_rate'],0)
        self.assertEqual(m['mate_positions'],1)

    def test_runner_uses_separate_calibration_split(self):
        import sys
        import types
        from chess_finetune.run import main
        class FakeScorer:
            def __init__(self, *args, **kwargs):
                pass
            def score(self, prompt, candidates, norm):
                return [types.SimpleNamespace(score=x) for x in (0., -4.)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split, targets in [('train',['a']), ('valid',['a']), ('calibration',['a','b']), ('test',['b'])]:
                records = [dict(prompt=f'FEN: {split} w - - 0 1\nLegal moves: a b', completion=t) for t in targets]
                (root/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            argv = ['run','--data',d,'--base-only','--model',d,'--bootstrap','10','--out',str(root/'out')]
            with patch.object(sys,'argv',argv), patch.dict(sys.modules, {'openjev.scorer':types.SimpleNamespace(OptionScorer=FakeScorer)}):
                main()
            artifact = json.loads((root/'out/base-calibrator.json').read_text())
            self.assertEqual(artifact['fit_split'],'calibration')
            self.assertEqual(artifact['fit_n'],2)
            self.assertEqual(artifact['temperature'],20.)
            self.assertTrue((root/'out/base-calibration-predictions.jsonl').exists())
            self.assertFalse((root/'out/base-valid-predictions.jsonl').exists())

    @unittest.skipUnless(chess, 'python-chess required')
    def test_strict_legality_and_ep_canonicalization(self):
        board = chess.Board()
        board.push_uci('e2e4')
        moves = [m.uci() for m in board.legal_moves]
        record = dict(prompt=f'FEN: {board.fen(en_passant="fen")}\nLegal moves: '+ ' '.join(moves), completion=moves[0])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'valid.jsonl'
            p.write_text(json.dumps(record)+'\n')
            rows = read_split(p, strict=True)
            self.assertEqual(rows[0]['position'].split()[3], '-')
            record['prompt'] += ' a1a8'
            p.write_text(json.dumps(record)+'\n')
            with self.assertRaisesRegex(ValueError, 'exactly all legal'):
                read_split(p, strict=True)

    @unittest.skipUnless(chess, 'python-chess required')
    def test_engine_side_to_move_and_coverage(self):
        from chess_finetune.engine import evaluate_engine
        board = chess.Board()
        moves = list(board.legal_moves)
        record = dict(id='0', fen=board.fen(), candidates=[m.uci() for m in moves],
                      scores=list(range(len(moves))), label=0)
        engine = MagicMock()
        engine.id = {'name':'test-engine'}
        engine.options = {'Threads':None, 'Hash':None}
        engine.analyse.return_value = [dict(pv=[move], score=chess.engine.PovScore(chess.engine.Cp(i), chess.BLACK)) for i, move in enumerate(moves)]
        context = MagicMock()
        context.__enter__.return_value = engine
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d)/'engine'
            exe.write_text('fixture')
            with patch('chess_finetune.engine.shutil.which', return_value=str(exe)), patch('chess.engine.SimpleEngine.popen_uci',return_value=context):
                result = evaluate_engine({'base':[record]}, 'fixture')
                self.assertEqual(result['metrics']['base']['mean_cp_loss'], len(moves)-1)
                engine.analyse.return_value = engine.analyse.return_value[:1]
                with self.assertRaisesRegex(ValueError, 'all candidate'):
                    evaluate_engine({'base':[record]}, 'fixture')


if __name__ == '__main__':
    unittest.main()
