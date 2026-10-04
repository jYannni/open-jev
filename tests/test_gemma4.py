"""Offline Gemma 4 checks using random, tiny weights (no downloads).

Two shapes are covered: an E-series layout (per-layer embeddings, K/V shared by the last
layers; Gemma 4 E2B/E4B) and the unified dense layout (K = V on full attention, no K/V
sharing; Gemma 4 12B). The MLX tests run wherever mlx-lm imports, including mlx[cpu].
"""
import importlib.util
import math
import tempfile
import unittest
from unittest.mock import patch

from openjev.scorer import OptionScorer

LAYERS = ['sliding_attention', 'full_attention', 'sliding_attention', 'full_attention']
COMMON = dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
              num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
              sliding_window=8, max_position_embeddings=256, layer_types=LAYERS)
E_SERIES = dict(hidden_size_per_layer_input=8, vocab_size_per_layer_input=32, num_kv_shared_layers=2)
UNIFIED = dict(attention_k_eq_v=True, num_global_key_value_heads=1, num_kv_shared_layers=0)
OPTIONS = ['a', 'longer option', 'xyz', 'q']
CONTEXTS = ['hi', 'context longer than the sliding window']


class Tokenizer:
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [2 + ord(c) % 29 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get('enable_thinking') is False
        return 'user:' + messages[0]['content'] + '\nmodel:'


def make_scorer(model, engine=None, batch_size=2):
    s = OptionScorer.__new__(OptionScorer)
    s._engine = engine
    s.model, s.tok = model, Tokenizer()
    s.backend, s.device = ('torch', 'cpu') if engine else ('mlx', 'cpu')
    s.batch_size, s.chat, s.sep = batch_size, False, ''
    s.pad_id, s.bos_id, s.last_timing = 0, 1, {}
    return s


class ScoringChecks:
    """Shared assertions; subclasses provide scorer(name, batch_size)."""

    def test_cached_matches_naive_and_batch_sizes(self):
        for name in ('e_series', 'unified'):
            for context in CONTEXTS:
                expected = self.scorer(name).score_naive(context, OPTIONS)
                for batch_size in [1, 2, 8]:
                    with self.subTest(model=name, context=context, batch_size=batch_size):
                        actual = self.scorer(name, batch_size).score(context, OPTIONS, norm='sum')
                        for a, b in zip(actual, expected):
                            self.assertAlmostEqual(a.logprob_sum, b, places=4)
                        self.assertAlmostEqual(sum(x.probability for x in actual), 1)

    def test_normalization_and_repeated_calls(self):
        for name in ('e_series', 'unified'):
            s = self.scorer(name)
            for norm in ['sum', 'mean', 'pmi']:
                with self.subTest(model=name, norm=norm):
                    a = s.score('context', ['x', 'long option'], norm=norm)
                    b = s.score('context', ['x', 'long option'], norm=norm)
                    for x, y in zip(a, b):
                        self.assertAlmostEqual(x.score, y.score, places=5)
                        self.assertTrue(math.isfinite(x.score))

    def test_chat_and_separator(self):
        for name in ('e_series', 'unified'):
            s = self.scorer(name)
            for chat in [False, True]:
                s.chat, s.sep = chat, '\nAnswer: '
                with self.subTest(model=name, chat=chat):
                    ids = s.context_ids('hello')
                    self.assertEqual((ids[0], ids.count(1)), (1, 1))  # exactly one BOS, first
                    actual = s.score('hello', ['a', 'bbb'])
                    expected = s.score_naive('hello', ['a', 'bbb'])
                    for a, b in zip(actual, expected):
                        self.assertAlmostEqual(a.logprob_sum, b, places=4)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'requires the torch extra')
class TorchGemma4Tests(ScoringChecks, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import (Gemma4ForCausalLM, Gemma4TextConfig, Gemma4UnifiedForCausalLM,
                                  Gemma4UnifiedTextConfig)
        from openjev.torch_backend import TorchBackend
        torch.manual_seed(42)
        cls.engines = {}
        for name, model in [('e_series', Gemma4ForCausalLM(Gemma4TextConfig(**COMMON, **E_SERIES))),
                            ('unified', Gemma4UnifiedForCausalLM(Gemma4UnifiedTextConfig(**COMMON, **UNIFIED)))]:
            engine = TorchBackend.__new__(TorchBackend)
            engine.torch, engine.device = torch, torch.device('cpu')
            engine.model, engine.tok = model.eval(), Tokenizer()
            cls.engines[name] = engine

    def scorer(self, name, batch_size=2):
        engine = self.engines[name]
        return make_scorer(engine.model, engine, batch_size)

    def test_load_multimodal_checkpoints(self):
        import torch
        from transformers import (AutoModelForCausalLM, Gemma4Config, Gemma4TextConfig,
                                  Gemma4UnifiedConfig, Gemma4UnifiedTextConfig)
        from openjev.torch_backend import TorchBackend
        configs = {
            'Gemma4ForConditionalGeneration': Gemma4Config(
                text_config=Gemma4TextConfig(**COMMON, **E_SERIES).to_dict(), vision_config=None, audio_config=None),
            'Gemma4UnifiedForConditionalGeneration': Gemma4UnifiedConfig(
                text_config=Gemma4UnifiedTextConfig(**COMMON, **UNIFIED).to_dict(), vision_config=None, audio_config=None),
        }
        for cls_name, config in configs.items():
            with self.subTest(cls=cls_name), tempfile.TemporaryDirectory() as directory:
                original = AutoModelForCausalLM.from_config(config).eval()
                original.save_pretrained(directory)
                with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
                    loaded = TorchBackend(directory, 'cpu', None)
                self.assertEqual(type(loaded.model).__name__, cls_name)
                ids = torch.tensor([[1, 4, 5]])
                with torch.inference_mode():
                    torch.testing.assert_close(loaded.model(input_ids=ids).logits, original(input_ids=ids).logits)
                prefix, last = loaded.prefill([1, 4, 5])
                cached = loaded.score_with_prefix(prefix, last, [[6], [7, 8, 9]], 2, 0)
                naive = loaded.score_naive([1, 4, 5], [[6], [7, 8, 9]])
                torch.testing.assert_close(torch.tensor(cached), torch.tensor(naive))


@unittest.skipUnless(importlib.util.find_spec('mlx') and importlib.util.find_spec('mlx_lm'), 'requires mlx-lm')
class MLXGemma4Tests(ScoringChecks, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        from mlx_lm.models import gemma4
        from mlx_lm.models.cache import KVCache
        import openjev.scorer as scorer_module
        # OptionScorer binds these globals when it loads an MLX model.
        scorer_module.mx, scorer_module.KVCache = mx, KVCache
        mx.random.seed(42)
        cls.models = {}
        # The 12B config states hidden_size_per_layer_input = 0; mlx-lm defaults it to 256.
        for name, extra in [('e_series', E_SERIES), ('unified', {**UNIFIED, 'hidden_size_per_layer_input': 0})]:
            model = gemma4.Model(gemma4.ModelArgs(model_type='gemma4', text_config={**COMMON, **extra},
                                                  vocab_size=COMMON['vocab_size']))
            mx.eval(model.parameters())
            cls.models[name] = model

    def scorer(self, name, batch_size=2):
        return make_scorer(self.models[name], None, batch_size)

    def test_cache_covers_only_layers_that_own_kv(self):
        self.assertEqual(len(self.scorer('e_series').new_cache()), 2)
        self.assertEqual(len(self.scorer('unified').new_cache()), 4)

    def test_features_match_uncached_forward(self):
        import mlx.core as mx
        import numpy as np
        from openjev.features import FeatureExtractor
        for name in ('e_series', 'unified'):
            for contextual in (False, True):
                with self.subTest(model=name, contextual=contextual):
                    s = self.scorer(name)
                    fx = FeatureExtractor(s, contextual=contextual)
                    self.assertEqual(fx.hidden, COMMON['hidden_size'])
                    ctx, opts = fx.extract(CONTEXTS[1], OPTIONS)
                    prefix = s.context_ids(CONTEXTS[1]) if contextual else [s.bos_id]
                    np.testing.assert_allclose(
                        ctx, np.array(fx.core(mx.array(s.context_ids(CONTEXTS[1]))[None])[0]), atol=2e-2)
                    for i, option in enumerate(OPTIONS):
                        ids = s.option_ids(option)
                        h = fx.core(mx.array(prefix + ids)[None])[0, len(prefix):].astype(mx.float32).mean(0)
                        np.testing.assert_allclose(opts[i], np.array(h), atol=2e-2)

    def test_route_a_extract_and_train_records_feature_norms(self):
        import json
        from pathlib import Path
        from openjev.features import extract_dataset
        from openjev.train import train
        rows = [{'context': f'question {i}', 'options': ['yes', 'no', 'later'], 'label': i % 3} for i in range(6)]
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / 'rows.jsonl'
            data.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            for split in ('train', 'validation'):
                meta = extract_dataset(self.scorer('e_series'), str(data), f'{d}/{split}.npz')
                self.assertEqual(meta['hidden'], COMMON['hidden_size'])
            result = train(f'{d}/train.npz', f'{d}/validation.npz', f'{d}/head.safetensors',
                           rank=4, epochs=1, batch_size=4)
            self.assertTrue(math.isfinite(result['best_val_top1']))
            norms = json.loads(Path(f'{d}/head.json').read_text())['feature_norms']
            self.assertTrue(norms['context'] > 0 and norms['option'] > 0)


if __name__ == '__main__':
    unittest.main()
