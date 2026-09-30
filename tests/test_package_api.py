"""Public Python workflow without loading model weights or an accelerator."""
import json
import subprocess
import sys
import unittest
from unittest.mock import Mock

from openjev import ChoiceTask, DecisionScorer


class PackageAPITests(unittest.TestCase):
    def test_imports_do_not_initialize_backends(self):
        code = '''
import json, sys
from openjev import (Candidate, ChoiceTask, DecisionScorer, OptionScorer,
                     extract_decision_features, train_decision_head)
print(json.dumps([name for name in sys.modules
                  if name.split('.')[0] in ('mlx', 'mlx_lm', 'torch')]))
'''
        result = subprocess.run([sys.executable, '-c', code], check=True,
                                capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), [])

    def test_predict_preserves_user_ids_and_structured_state(self):
        scorer = object.__new__(DecisionScorer)
        scorer.score = Mock(return_value={'choice_id': 'returns'})
        result = scorer.predict(state={'message': 'broken'}, question='Route?',
                                candidates={'returns': 'Replace', 'billing': 'Pay'},
                                request_id='ticket-17')
        task = scorer.score.call_args.args[0]
        self.assertIsInstance(task, ChoiceTask)
        self.assertEqual(task.id, 'ticket-17')
        self.assertEqual(task.state, {'message': 'broken'})
        self.assertEqual([(c.id, c.text) for c in task.candidates],
                         [('returns', 'Replace'), ('billing', 'Pay')])
        self.assertEqual(result, {'choice_id': 'returns'})

    def test_bad_requests_fail_before_backend_or_inference(self):
        scorer = object.__new__(DecisionScorer)
        for candidates in (['a', 'b'], {'a': 'one'}, {'a': 'same', 'b': 'same'}):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                scorer.predict(state='test', question='Choose?', candidates=candidates)
        with self.assertRaises(ValueError):
            scorer.score({'schema_version': 1})


if __name__ == '__main__':
    unittest.main()
