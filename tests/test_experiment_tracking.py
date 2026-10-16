"""Run lifecycle and sync boundary tests in disposable local git repositories."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'scripts').mkdir()
        for name in ['run.py', 'experiment_notes.py', 'fid_report.py']:
            shutil.copyfile(SOURCE / 'scripts' / name, self.root / 'scripts' / name)
        (self.root / 'docs').mkdir()
        (self.root / 'docs/RESULTS.md').write_text('| id | parent | 가설 | 핵심 지표 | 결론 |\n|---|---|---|---|---|\n')
        (self.root / '.gitignore').write_text('**/__pycache__/\nruns/*/logs/\nruns/*/ckpt/\n')
        self.command('git', 'init', '-b', 'main')
        self.command('git', 'config', 'user.name', 'Workflow Test')
        self.command('git', 'config', 'user.email', 'test@example.invalid')
        self.commit()

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, *args, ok=True):
        result = subprocess.run(args, cwd=self.root, text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def commit(self):
        self.command('git', 'add', '.')
        self.command('git', 'commit', '-m', 'test snapshot')

    def load_notes(self):
        sys.path.insert(0, str(self.root / 'scripts'))
        # run module may belong to another test root; ROOT of notes is independent.
        spec = importlib.util.spec_from_file_location('notes_test', self.root / 'scripts/experiment_notes.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = self.root
        return module

    def config(self, code, ident='exp-001'):
        (self.root / 'configs').mkdir(exist_ok=True)
        path = self.root / 'configs' / (ident + '.yaml')
        path.write_text(json.dumps({'id': ident, 'parent_id': None, 'hypothesis': 'lifecycle validation',
                                    'seed': 42, 'command': ['{python}', '-c', code]}))
        self.commit()
        return path

    def run_config(self, path, ok=True):
        return self.command(sys.executable, 'scripts/run.py', 'run', '--config', str(path), ok=ok)

    def test_success_metrics_and_duplicate_run(self):
        path = self.config("import os; from pathlib import Path; Path(os.environ['EXP_METRICS_PATH']).write_text(chr(123)+chr(34)+'loss'+chr(34)+': 0.5'+chr(125))")
        self.run_config(path)
        meta = json.loads((self.root / 'runs/exp-001/meta.json').read_text())
        self.assertEqual(meta['status'], 'done')
        self.assertFalse(meta['git_dirty'])
        self.assertTrue(meta['finished_at'])
        self.assertEqual(json.loads((self.root / 'runs/exp-001/metrics.json').read_text()), {'loss': .5, 'official_fid': None, 'valid_fid': None})
        notes = self.load_notes()
        self.assertEqual(notes.update_results(), ['exp-001'])
        self.assertEqual(notes.update_results(), [])
        self.commit()
        self.assertNotEqual(self.run_config(path, ok=False).returncode, 0)
        self.assertEqual(meta, json.loads((self.root / 'runs/exp-001/meta.json').read_text()))

    def test_failed_command_preserves_final_metadata(self):
        path = self.config('raise RuntimeError("intentional test failure")')
        self.assertNotEqual(self.run_config(path, ok=False).returncode, 0)
        meta = json.loads((self.root / 'runs/exp-001/meta.json').read_text())
        self.assertEqual(meta['status'], 'failed')
        self.assertTrue(meta['finished_at'])
        self.assertIn('exited 1', meta['error_summary'])
        self.assertTrue((self.root / 'runs/exp-001/logs/console.log').exists())
        self.assertTrue((self.root / 'runs/exp-001/metrics.json').exists())

    def test_success_without_metrics_is_failed(self):
        path = self.config('pass')
        self.assertNotEqual(self.run_config(path, ok=False).returncode, 0)
        meta = json.loads((self.root / 'runs/exp-001/meta.json').read_text())
        self.assertEqual(meta['status'], 'failed')
        self.assertIn('FileNotFoundError', meta['error_summary'])

    def test_dirty_source_refused_before_run(self):
        path = self.config('pass')
        (self.root / 'uncommitted.py').write_text('pass')
        self.assertNotEqual(self.run_config(path, ok=False).returncode, 0)
        self.assertFalse((self.root / 'runs/exp-001').exists())

    def test_stage_allowlist_and_size(self):
        notes = self.load_notes()
        (self.root / 'docs/new.md').write_text('note')
        (self.root / 'unrelated.py').write_text('not part of notes')
        self.assertEqual(notes.stage_notes(), ['docs/new.md'])
        self.command('git', 'commit', '-m', 'notes')
        (self.root / 'docs/large.png').write_bytes(b'x' * (1048576 + 1))
        with self.assertRaisesRegex(RuntimeError, 'exceeds'):
            notes.stage_notes()
        self.assertEqual(self.command('git', 'diff', '--cached', '--name-only').stdout, '')

    def test_review_fields_and_config_immutability(self):
        notes = self.load_notes()
        (self.root / 'reviews').mkdir()
        review = self.root / 'reviews/review.json'
        review.write_text(json.dumps({'title': 'keep', 'status': 'open', 'resolution': None}))
        self.commit()
        review.write_text(json.dumps({'title': 'keep', 'status': 'done', 'resolution': 'exp-001'}))
        notes.check_reviews(review)
        review.write_text(json.dumps({'title': 'changed', 'status': 'done'}))
        with self.assertRaisesRegex(RuntimeError, 'Only status/resolution'):
            notes.check_reviews(review)
        review.write_text(json.dumps({'title': 'keep', 'status': 'open', 'resolution': None}))
        path = self.config('pass')
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(RuntimeError, 'immutable'):
            notes.stage_notes()

    def test_explicit_fid_mapping_preserves_unknowns(self):
        spec = importlib.util.spec_from_file_location('fid_test', self.root / 'scripts/fid_report.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        paired = module.normalize_metrics({'official': {'fid': 25.0}, 'development': {'fid': 41.0}})
        self.assertEqual((paired['official_fid'], paired['valid_fid']), (25.0, 41.0))
        unknown = module.normalize_metrics({'fid': 99.0})
        self.assertIsNone(unknown['official_fid'])
        self.assertIsNone(unknown['valid_fid'])
        single = module.normalize_metrics({'reference_split': 'official', 'fid': 25.0})
        self.assertEqual(single['official_fid'], 25.0)
        self.assertIsNone(single['valid_fid'])
        protocol = {'sample_count':3020,'batch_size':16,'sampler':'ddim','steps':32,
                    'guidance_scale':2.0,'nfe_per_sample':64,'seed':42,'precision':'bf16','clip_denoised':True}
        module.validate_generation(protocol, protocol)
        with self.assertRaisesRegex(ValueError, 'seed'):
            module.validate_generation(protocol, dict(protocol, seed=1042))

    def test_hook_denies_origin_and_wrong_target(self):
        hook = str(SOURCE / '.githooks/pre-push')
        self.assertNotEqual(self.command(hook, 'origin', 'any', ok=False).returncode, 0)
        self.assertEqual(self.command(hook, 'festgen', 'git@github-FestGen:JoohoSon24/FestGen.git').returncode, 0)
        result = subprocess.run([hook, 'festgen', 'git@github-FestGen:JoohoSon24/FestGen.git'],
                                 input='refs/heads/main abc refs/heads/other def\n', text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

if __name__ == '__main__':
    unittest.main()
