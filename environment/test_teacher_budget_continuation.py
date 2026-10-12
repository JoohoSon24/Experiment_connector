"""Small CPU checks for continuation guards and external logging behavior."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("budget_runner", ROOT / "scripts/run_teacher_budget_continuation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Run:
    url = "https://example.invalid/unit-test"
    def __init__(self):
        self.summary = {}
        self.records = []
    def log(self, data):
        self.records.append(data)


class Guards(unittest.TestCase):
    def setUp(self):
        _, self.base = runner.load_config(runner.BASE)

    def test_allow_budget_and_output_bookkeeping(self):
        new = copy.deepcopy(self.base)
        new.update(num_train_steps=150000, experiment_name="continuation", output_dir="new-run")
        runner.check_recipe(self.base, new)

    def test_reject_recipe_changes_even_when_checkpoint_shapes_match(self):
        changes = {"seed": 43, "batch_size": 32, "ema_decay": .999, "class_dropout": .2,
                   "learning_rate": .00005, "horizontal_flip_probability": .25,
                   "beta_end": .019, "warmup_steps": 100, "gradient_accumulation": 2}
        for key, value in changes.items():
            with self.subTest(key=key):
                new = copy.deepcopy(self.base)
                new[key] = value
                with self.assertRaises(ValueError):
                    runner.check_recipe(self.base, new)

    def test_reject_inference_only_checkpoint_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference_only.pt"
            torch.save({"state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "Full teacher checkpoint required"):
                runner.checkpoint_step(path, (100000, 100000))

    def test_log_true_global_step_and_enforce_memory_ceiling(self):
        original = runner.RUN_ROOT
        try:
            with tempfile.TemporaryDirectory(dir=ROOT / "environment") as directory:
                runner.RUN_ROOT = Path(directory)
                run = Run()
                observer = runner.Observer(run, None)
                observer.training_line('{"step":100020,"loss":0.02,"lr":0.0002,"peak_vram_bytes":9790000000}',
                                       ROOT / "environment")
                self.assertEqual(run.records[0]["global_step"], 100020)
                self.assertEqual(run.records[0]["train/lr"], .0002)
                self.assertEqual(json.loads((runner.RUN_ROOT / "status.json").read_text())["phase"], "training")
                observer.training_line("not a training JSON line", ROOT / "environment")
                self.assertEqual(len(run.records), 1)
                with self.assertRaisesRegex(RuntimeError, "20GB"):
                    observer.training_line('{"step":100040,"loss":0.02,"lr":0.0002,"peak_vram_bytes":20000000000}',
                                           ROOT / "environment")
        finally:
            runner.RUN_ROOT = original


if __name__ == "__main__":
    unittest.main()
