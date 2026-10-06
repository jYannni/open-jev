"""system_one chat plumbing: off by default, follows the scorer, explicit argument wins."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from openjev.scorer import OptionScore
from openjev.systemone import SystemOneRequest, system_one

EXAMPLE = Path(__file__).resolve().parent.parent / 'examples' / 'systemone-quickstart.json'


class RecordingScorer:
    def __init__(self, chat=False):
        self.chat = chat
        self.calls = []
        self.last_timing = {}

    def score(self, context, options, norm='mean', chat=None, sep=None):
        self.calls.append(self.chat if chat is None else chat)
        self.last_timing = {'context_tokens': 1, 'option_tokens': len(options)}
        p = 1 / len(options)
        return [OptionScore(o, 1, -1.0, -1.0, None, -1.0, p) for o in options]


class SystemOneChatTests(unittest.TestCase):
    def request(self):
        return SystemOneRequest(**json.loads(EXAMPLE.read_text()))

    def test_default_scorer_stays_raw(self):
        scorer = RecordingScorer()
        system_one(scorer, self.request(), model_name='m')
        self.assertEqual(scorer.calls, [False, False, False])

    def test_follows_scorer_chat_setting(self):
        scorer = RecordingScorer(chat=True)
        system_one(scorer, self.request(), model_name='m')
        self.assertEqual(scorer.calls, [True, True, True])

    def test_explicit_argument_wins(self):
        for scorer_chat, chat in [(False, True), (True, False)]:
            scorer = RecordingScorer(chat=scorer_chat)
            system_one(scorer, self.request(), model_name='m', chat=chat)
            self.assertEqual(scorer.calls, [chat] * 3)

    def test_serve_cli_and_app_forward_chat(self):
        from fastapi.testclient import TestClient
        from openjev.cli import main
        from openjev.server import create_app
        for argv, expected in [([], False), (['--chat'], True)]:
            with self.subTest(argv=argv):
                with patch.object(sys, 'argv', ['openjev', 'serve', *argv]), patch('openjev.server.serve') as serve:
                    main()
                self.assertEqual(serve.call_args.kwargs['chat'], expected)
                with patch('openjev.server.OptionScorer') as scorer:
                    with TestClient(create_app(model_path='test-model', chat=expected)) as client:
                        self.assertEqual(client.get('/health').json()['chat'], expected)
                    self.assertEqual(scorer.call_args.kwargs['chat'], expected)


if __name__ == '__main__':
    unittest.main()
