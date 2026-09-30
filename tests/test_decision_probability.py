import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openjev.decision import ChoiceTask, choice_result, summarize
from openjev.decision_experiment import ExperimentConfig, validate_splits
from openjev.decision_probability import (TemperatureCalibrator, baseline_predictions,
    fit_temperature, probability_report)


def task(i, label='a', group=None):
    value = dict(schema_version=1, id=f'row-{i}', state=f'state-{i}', question='Choose',
                 candidates=[dict(id='a', text='A'), dict(id='b', text='B')], target=dict(choice_id=label))
    if group is not None:
        value['group_id'] = group
    return ChoiceTask.parse(value, labelled=True)


def predictions(tasks, utilities=(math.log(4), 0)):
    return [dict(choice_result(t, utilities), artifact_id='head-sha') for t in tasks]


class ProbabilityTests(unittest.TestCase):
    def test_temperature_matches_analytic_binary_optimum(self):
        tasks = [task(i, 'b' if i % 4 == 0 else 'a') for i in range(100)]
        raw = predictions(tasks)
        for objective in ('nll', 'brier'):
            c = TemperatureCalibrator.fit(tasks, raw, 'head-sha', objective)
            self.assertAlmostEqual(c.metadata['temperature'], math.log(4)/math.log(3), places=5)
            self.assertLess(c.metadata['fitted_objective'], c.metadata['raw_objective'])
            scored = [c.apply(r) for r in raw]
            report = probability_report(tasks, scored, bootstrap_samples=20)
            self.assertAlmostEqual(report['brier'], 0.375, places=8)
            self.assertAlmostEqual(report['ece'], 0, places=7)
            self.assertAlmostEqual(report['nll'], summarize(tasks, scored)['nll'])
            self.assertEqual([r['choice_id'] for r in raw], [r['choice_id'] for r in scored])
            self.assertNotIn('raw_probability', raw[0]['candidates'][0])

    def test_bounds_flat_logits_and_extreme_nll(self):
        self.assertEqual(fit_temperature([[0, 0]], [0])['temperature'], 1)
        fitted = fit_temperature([[4, 0]], [0], bounds=(0.5, 2))
        self.assertEqual(fitted['temperature'], 0.5)
        self.assertTrue(fitted['at_bound'])
        t = task(1, 'b')
        report = probability_report([t], predictions([t], (1000, -1000)), bootstrap_samples=0)
        self.assertEqual(report['nll'], 2000)
        self.assertEqual(report['brier'], 2)
        for bounds in ((0, 2), (2, 4), (1, 1), (0.1, float('inf'))):
            with self.assertRaises(ValueError):
                fit_temperature([[1, 0]], [0], bounds=bounds)

    def test_artifact_binding_roundtrip_and_alignment(self):
        tasks = [task(1), task(2, 'b')]
        raw = predictions(tasks)
        c = TemperatureCalibrator.fit(tasks, raw, 'head-sha')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'calibrator.json'
            c.save(path)
            self.assertEqual(TemperatureCalibrator.load(path, 'head-sha').apply(raw[0]), c.apply(raw[0]))
            with self.assertRaises(ValueError):
                TemperatureCalibrator.load(path, 'another-head')
            value = json.loads(path.read_text())
            value['temperature'] = -1
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                TemperatureCalibrator.load(path, 'head-sha')
        with self.assertRaises(ValueError):
            c.apply(c.apply(raw[0]))
        with self.assertRaises(ValueError):
            TemperatureCalibrator.fit(tasks, list(reversed(raw)), 'head-sha')
        with self.assertRaises(ValueError):
            probability_report(tasks, raw[:1])

    def test_group_bootstrap_and_confidence_ties(self):
        tasks = [task(i, 'a' if i < 4 else 'b', group='one' if i < 4 else 'two') for i in range(8)]
        raw = predictions(tasks)
        a = probability_report(tasks, raw, bootstrap_samples=100, seed=4)
        self.assertEqual(a, probability_report(tasks, raw, bootstrap_samples=100, seed=4))
        self.assertEqual(a['uncertainty']['groups'], 2)
        self.assertEqual(a['uncertainty']['intervals']['accuracy'], [0, 1])
        self.assertTrue(all(s['coverage'] == 1 for s in a['selective']))
        self.assertEqual(sum(b['n'] for b in a['reliability']), 8)
        self.assertEqual(a['strata']['candidate_count']['2']['accuracy'], 0.5)

    def test_prior_is_train_only_and_handles_unseen_text(self):
        train = [task(i) for i in range(3)]
        test = [task(10, 'b')]
        result = baseline_predictions(train, test)
        self.assertAlmostEqual(result['train_text_prior'][0]['candidates'][0]['probability'], 0.8)
        self.assertEqual(result['uniform'][0]['candidates'][0]['probability'], 0.5)

    def test_config_typing_relative_paths_and_four_way_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = '''schema_version=1
model="model"
out="out"
[datasets]
train="train.jsonl"
validation="validation.jsonl"
calibration="calibration.jsonl"
test="test.jsonl"
'''
            path = root/'config.toml'
            path.write_text(base)
            config = ExperimentConfig.load(path)
            self.assertEqual(config.model, str((root/'model').resolve()))
            for i, split in enumerate(config.datasets):
                (root/f'{split}.jsonl').write_text(json.dumps(task(i).to_dict())+'\n')
            self.assertEqual(len(validate_splits(config)), 4)
            (root/'calibration.jsonl').write_text((root/'test.jsonl').read_text())
            with self.assertRaisesRegex(ValueError, 'overlap'):
                validate_splits(config)
            for settings in ('rank=true', 'brier_weights=[0.0]', 'selection="test"', 'lr=nan', 'typo=1'):
                path.write_text(base+'\n[training]\n'+settings+'\n')
                with self.assertRaises(ValueError):
                    ExperimentConfig.load(path)


if __name__ == '__main__':
    unittest.main()
