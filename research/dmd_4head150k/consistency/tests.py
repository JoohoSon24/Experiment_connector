"""CPU numerical and exact-resume tests; no dataset, teacher file or GPU needed."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from teacher.config import Config
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint, make_model
from .config import Settings
from .core import denoise, loss, noise_levels, sample, sampling_times, teacher_step
from .train import run_training


class ConstantEpsilon(nn.Module):
    num_classes = 3

    def forward(self, x, t, labels):
        return torch.ones_like(x) * (labels[:, None, None, None] + 1) * 0.05


class TinyData(Dataset):
    def __init__(self):
        self.images = torch.randn(7, 3, 8, 8, generator=torch.Generator().manual_seed(19)).clamp(-1, 1)
        self.paths = [f"synthetic/{index}" for index in range(7)]
        self.classes = ["a", "b", "c"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.images[index], torch.tensor(index % 3)


def fixture():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "configs/teacher/teacher_b64_noaug_constant_100k.json").read_text())
    raw.update(image_size=8, model_channels=8, channel_mult=[1, 2], num_res_blocks=1,
               attention_resolutions=[4], num_classes=3)
    cfg = Config(**raw)
    torch.manual_seed(7)
    teacher = make_model(cfg, 3)
    nn.init.normal_(teacher.out_conv.weight, std=0.01)
    data = TinyData()
    source = {"config": raw, "ema": teacher.state_dict(), "class_names": data.classes,
              "train_paths_sha256": hashlib.sha256("\n".join(data.paths).encode()).hexdigest()}
    return cfg, teacher, data, source


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.alpha = Diffusion(1000, 0.0001, 0.02, torch.device("cpu")).alpha_bar

    def test_boundary_and_ddim_coordinate_equivalence(self):
        network = ConstantEpsilon()
        y = torch.randn(3, 3, 8, 8)
        labels = torch.arange(3)
        self.assertTrue(torch.equal(denoise(network, y, torch.zeros(3, dtype=torch.long), labels, self.alpha), y))
        t, next_t = torch.tensor([999, 700, 20]), torch.tensor([800, 300, 0])
        a = self.alpha[t, None, None, None].sqrt()
        s = (1 - self.alpha[t, None, None, None]).sqrt()
        an = self.alpha[next_t, None, None, None].sqrt()
        sn = (1 - self.alpha[next_t, None, None, None]).sqrt()
        x = a * y
        for guidance in (0.0, 1.0, 2.0):
            eps_u = network(x, t, torch.full_like(labels, 3))
            eps_c = network(x, t, labels)
            eps = eps_u + guidance * (eps_c - eps_u)
            for clip in (False, True):
                x0 = (x - s * eps) / a
                if clip:
                    x0 = x0.clamp(-1, 1)
                expected_vp = an * x0 + sn * (x - a * x0) / s
                actual = teacher_step(network, y, t, next_t, labels, self.alpha, guidance, clip)
                torch.testing.assert_close(an * actual, expected_vp, atol=2e-6, rtol=2e-5)

    def test_backward_target_frozen_and_sampler_nfe(self):
        cfg, teacher, data, source = fixture()
        student, target = make_model(cfg, 3), make_model(cfg, 3)
        teacher.requires_grad_(False)
        target.requires_grad_(False)
        value = loss(student, target, teacher, data.images[:3], torch.arange(3), self.alpha,
                     torch.linspace(0, 999, 32).long(), 2.0, 0.5, True)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(student.out_conv.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        student.eval()
        calls = []
        hook = student.register_forward_hook(lambda *args: calls.append(1))
        for nfe in (1, 2, 3, 4):
            times = sampling_times(1000, nfe)
            calls.clear()
            output = sample(student, self.alpha, times, torch.arange(3), 42, image_size=8)
            self.assertEqual(len(calls), nfe)
            repeat = sample(student, self.alpha, times, torch.arange(3), 42, image_size=8)
            self.assertTrue(torch.equal(output, repeat))
            self.assertEqual(output.shape, (3, 3, 8, 8))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(((output >= -1) & (output <= 1)).all())
        hook.remove()
        with self.assertRaises(ValueError):
            sample(student, self.alpha, torch.tensor([999, 0]), torch.arange(3), 42, image_size=8)

    def test_exact_resume_across_epoch_and_changed_config_rejected(self):
        cfg, teacher, data, source = fixture()
        settings = Settings(num_train_steps=4, batch_size=4, microbatch_size=2,
                            num_workers=0, amp_dtype="none", num_scales=16,
                            save_interval=1, log_interval=10, sample_interval=99, milestone_steps=())
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            full, split = Path(folder) / "full", Path(folder) / "split"
            run_training(settings, source, data, full, "test-teacher", torch.device("cpu"))
            first = run_training(settings, source, data, split, "test-teacher", torch.device("cpu"), 1)
            self.assertEqual(first["status"], "paused")
            self.assertFalse((split / "final.pt").exists())
            run_training(settings, source, data, split, "test-teacher", torch.device("cpu"))
            a, b = load_checkpoint(full / "final.pt"), load_checkpoint(split / "final.pt")
            def same(x, y):
                if isinstance(x, torch.Tensor):
                    self.assertTrue(torch.equal(x, y))
                elif isinstance(x, dict):
                    self.assertEqual(x.keys(), y.keys())
                    for key in x:
                        same(x[key], y[key])
                elif isinstance(x, (list, tuple)):
                    self.assertEqual(len(x), len(y))
                    for u, v in zip(x, y):
                        same(u, v)
                else:
                    self.assertEqual(x, y)
            for key in ("model", "target", "ema", "optimizer", "torch_rng", "python_rng",
                        "step", "epoch", "batch_in_epoch", "images_seen", "last_loss"):
                same(a[key], b[key])
            with self.assertRaisesRegex(ValueError, "settings changed"):
                run_training(replace(settings, guidance_scale=1), source, data, split,
                             "test-teacher", torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "teacher_sha256 changed"):
                run_training(settings, source, data, split, "different-teacher", torch.device("cpu"))

    def test_schedule_and_configuration_guards(self):
        with self.assertRaises(ValueError):
            noise_levels(torch.tensor([0.5, 0.6]))
        with self.assertRaises(ValueError):
            Settings(warmup_steps=10).validate(1000)
        with self.assertRaises(ValueError):
            Settings(num_scales=1001).validate(1000)


if __name__ == "__main__":
    unittest.main()
