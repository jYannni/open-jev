import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import numpy as np

from openjev.decision import (RECIPE, Candidate, ChoiceTask, check_disjoint, choice_result,
    digest, file_digest, load_tasks, model_identity, provenance, read_artifact, render_context, summarize)
from openjev.decision_mlx import inspect_features, train_artifact


def example():
    return dict(schema_version=1, id='ticket-1', state={'message':'broken package'},
                question='Choose the responsible department.',
                candidates=[{'id':'return','text':'Returns and replacements'},
                            {'id':'billing','text':'Billing inquiries'}], target={'choice_id':'return'})


def write_cache(path, split='train', label=0, identity='train'):
    backbone = {'path':'fake', 'files':{'model':'abc'}, 'fingerprint':digest({'model':'abc'})}
    limits = {'context_tokens':100,'candidate_tokens':20}
    signature = dict(model_fingerprint=backbone['fingerprint'], recipe=RECIPE, hidden=4,
                     limits=limits, runtime={'test':'1'})
    meta = dict(schema_version=1, strategy='candidate_head', n=1, hidden=4, split=split,
                recipe=RECIPE, limits=limits, backbone=backbone, signature=signature,
                signature_sha256=digest(signature),
                provenance={'ids':[identity], 'inputs':[identity], 'groups':[]})
    np.savez(path, meta=np.array(json.dumps(meta)), labels=np.array([label],np.int32),
             ctx_0=np.ones((3,4),np.float16), opt_0=np.ones((2,4),np.float16))
    return meta


class DecisionTests(unittest.TestCase):
    def test_contract_roundtrip_and_render_does_not_encode_ids_or_targets(self):
        task = ChoiceTask.parse(example(), labelled=True)
        self.assertEqual(task.to_dict(),example())
        changed = replace(task,id='other',choice_id=None,
                          candidates=tuple(Candidate('x'+c.id,c.text) for c in task.candidates))
        self.assertEqual(render_context(task),render_context(changed))
        self.assertEqual(task.input_fingerprint,changed.input_fingerprint)
        self.assertNotIn('return',render_context(task))
        self.assertEqual(task.input_fingerprint,replace(task,candidates=task.candidates[::-1]).input_fingerprint)

    def test_reject_ambiguous_or_invalid_labels(self):
        for change in ({'schema_version':True},{'target':{'choice_id':'missing'}},
                       {'question':''},{'state':float('nan')},{'unexpected':1}):
            value = dict(example(),**change)
            with self.assertRaises(ValueError):
                ChoiceTask.parse(value, labelled=True)
        value = example()
        value['candidates'][1]['id'] = 'return'
        with self.assertRaisesRegex(ValueError,'duplicate'):
            ChoiceTask.parse(value)
        value = example()
        value['candidates'][1]['text'] = value['candidates'][0]['text']
        with self.assertRaisesRegex(ValueError,'duplicate'):
            ChoiceTask.parse(value)

    def test_permuted_probabilities_and_ties(self):
        task = ChoiceTask.parse(example())
        a = choice_result(task,[2.,-1.])
        b = choice_result(replace(task,candidates=task.candidates[::-1]),[-1.,2.])
        self.assertEqual(a['choice_id'],b['choice_id'])
        self.assertEqual({c['id']:c['probability'] for c in a['candidates']},
                         {c['id']:c['probability'] for c in b['candidates']})
        self.assertEqual(choice_result(task,[0.,0.])['choice_id'],
                         choice_result(replace(task,candidates=task.candidates[::-1]),[0.,0.])['choice_id'])

    def test_metrics_stable_for_extreme_logits(self):
        task = ChoiceTask.parse(example())
        result = choice_result(task,[-1000.,1000.])
        m = summarize([task],[result])
        self.assertEqual(m['accuracy'],0)
        self.assertEqual(m['nll'],2000)
        self.assertEqual(m['brier'],2)
        self.assertEqual(m['ece'],1)

    def test_split_overlap_ignores_ids_and_order(self):
        task = ChoiceTask.parse(example())
        another = replace(task,id='test',candidates=task.candidates[::-1])
        with self.assertRaisesRegex(ValueError,'inputs'):
            check_disjoint(provenance([task]),provenance([another]))
        a, b = replace(task,group_id='same'), replace(another,state='new state',group_id='same')
        with self.assertRaisesRegex(ValueError,'groups'):
            check_disjoint(provenance([a]),provenance([b]))

    def test_file_errors_have_context_and_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'data.jsonl'
            p.write_text(json.dumps(example())+'\n'+json.dumps(example())+'\n')
            with self.assertRaisesRegex(ValueError,'data.jsonl:2: duplicate'):
                load_tasks(p)

    def test_cache_validation_and_test_split_training_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_cache(root/'train.npz')
            write_cache(root/'validation.npz','test',identity='test')
            self.assertEqual(inspect_features(root/'train.npz')['hidden'],4)
            with self.assertRaisesRegex(ValueError,'test cannot'):
                train_artifact(root/'train.npz',root/'validation.npz',root/'artifact')
            write_cache(root/'bad.npz',label=3)
            with self.assertRaisesRegex(ValueError,'invalid feature'):
                inspect_features(root/'bad.npz')

    def test_artifact_integrity_and_model_relocation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for folder in ('a','b'):
                p=root/folder
                p.mkdir()
                (p/'config.json').write_text('{}')
                (p/'model.safetensors').write_bytes(b'weights')
            self.assertEqual(model_identity(root/'a')['fingerprint'],model_identity(root/'b')['fingerprint'])
            (root/'b/model.safetensors').write_bytes(b'different')
            self.assertNotEqual(model_identity(root/'a')['fingerprint'],model_identity(root/'b')['fingerprint'])
            (root/'head.safetensors').write_bytes(b'head')
            backbone = model_identity(root/'a')
            signature = dict(model_fingerprint=backbone['fingerprint'],recipe=RECIPE,hidden=4,limits={})
            (root/'head.json').write_text(json.dumps({'hidden':4,'features_meta':{'signature_sha256':digest(signature)}}))
            manifest = dict(schema_version=1,strategy='candidate_head',recipe=RECIPE,
                            head='attention_head_v1',backbone=backbone,feature_signature=signature,limits={},
                            files={n:file_digest(root/n) for n in ('head.safetensors','head.json')})
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(read_artifact(root),manifest)
            (root/'head.safetensors').write_bytes(b'corruption')
            with self.assertRaisesRegex(ValueError,'integrity'):
                read_artifact(root)

    def test_eval_rejects_leakage_before_loading_backbone(self):
        from openjev.cli import main
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=root/'test.jsonl'
            source.write_text(json.dumps(example())+'\n')
            task=ChoiceTask.parse(example())
            manifest={'training':{'train':provenance([task]),'validation':provenance([])}}
            with patch('openjev.decision_cli.read_artifact',return_value=manifest), patch('openjev.decision_mlx.DecisionScorer') as scorer:
                with self.assertRaisesRegex(ValueError,'overlap'):
                    main(['decision','eval',str(root/'artifact'),'--data',str(source),'--out',str(root/'out')])
                scorer.assert_not_called()

    def test_train_rejects_feature_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_cache(root/'train.npz')
            meta=write_cache(root/'valid.npz','validation',identity='valid')
            with np.load(root/'valid.npz') as z:
                arrays={key:z[key] for key in z.files}
            meta['signature']['runtime']={'test':'2'}
            meta['signature_sha256']=digest(meta['signature'])
            arrays['meta']=np.array(json.dumps(meta))
            np.savez(root/'valid.npz',**arrays)
            with self.assertRaisesRegex(ValueError,'contracts differ'):
                train_artifact(root/'train.npz',root/'valid.npz',root/'artifact')

    def test_cli_conversion_and_help(self):
        from openjev.cli import main
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            p=root/'legacy.jsonl'
            p.write_text(json.dumps({'context':'state','options':['one','two'],'label':1})+'\n')
            main(['decision','convert',str(p),'--out',str(root/'native.jsonl'),'--prefix','test'])
            task=load_tasks(root/'native.jsonl',labelled=True)[0]
            self.assertEqual(task.choice_id,'option-1')
            self.assertEqual(task.id,'test:1')
        with self.assertRaises(SystemExit) as ctx:
            main(['decision','score','--help'])
        self.assertEqual(ctx.exception.code,0)


if __name__ == '__main__':
    unittest.main()
