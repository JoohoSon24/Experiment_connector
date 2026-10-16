"""Sequential n1 then n2 training-budget ablations with holdout selection."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import torch

from teacher.utils import load_checkpoint, write_json


ROOT = Path("experiments/low_nfe_budget_100k")
CONFIG = "configs/distillation/low_nfe_budget_100k.json"
BASE = Path("experiments/fixed_cfg2_progressive")
PLAN = (
    ("n1", BASE / "n2/final.pt", BASE / "n1/final.pt"),
    ("n2", BASE / "n4/final.pt", BASE / "n2/final.pt"),
)
STEPS = (50_000, 75_000, 100_000)


def command(args: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(args), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, args)


def checkpoint_for(stage: Path, step: int) -> Path:
    return stage / ("final.pt" if step == 100_000 else f"step_{step:06d}.pt")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for the long training run")
    if len(list(Path("data/development/seed42_holdout5/reference").glob("*.png"))) != 755:
        raise RuntimeError("Training-only development reference is incomplete")
    settings = json.loads(Path(CONFIG).read_text())
    assert settings["progressive_updates"] == 100_000
    assert settings["evaluation_steps"] == list(STEPS)
    assert settings["learning_rate_schedule"] == "constant" and settings["warmup_steps"] == 0
    ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"config": CONFIG, "order": ["n1", "n2"], "stages": {}}
    for name, teacher, original in PLAN:
        if not teacher.is_file() or not original.is_file():
            raise FileNotFoundError(f"Missing frozen baseline: {name}, {teacher}, {original}")
        run = ROOT / name
        run.mkdir(parents=True, exist_ok=True)
        final = checkpoint_for(run, 100_000)
        if not final.exists():
            command([sys.executable, "-u", "-m", "distillation.train", "--config", CONFIG,
                     "--teacher-checkpoint", str(teacher), "--stage", name,
                     "--output-dir", str(run), "--resume-checkpoint", str(original)],
                    run / "console.log")
        completed = load_checkpoint(final, "cpu")
        if completed["step"] != 100_000 or completed["teacher_checkpoint"] != str(teacher):
            raise ValueError(f"Unexpected completed checkpoint: {final}")
        del completed
        baseline_metrics = json.loads((BASE / name / "development_evaluation/metrics.json").read_text())
        reports = [{"step": 25_000, "fid": baseline_metrics["fid"],
                    "checkpoint": str(original), "reference_split": "development"}]
        for step in STEPS:
            checkpoint = checkpoint_for(run, step)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            output = run / f"development_step_{step:06d}"
            command([sys.executable, "-u", "-m", "distillation.evaluate", "--checkpoint", str(checkpoint),
                     "--output-dir", str(output), "--reference-split", "development", "--seed", "42"],
                    output / "console.log")
            measured = json.loads((output / "metrics.json").read_text())
            if measured["fid"] is None or measured["reference_count"] != 755:
                raise ValueError(f"Incomplete development evaluation: {output}")
            reports.append({"step": step, "fid": measured["fid"], "checkpoint": str(checkpoint),
                            "reference_split": "development"})
            print(f"{name} step={step} development FID={measured['fid']:.5f}", flush=True)
        best = min(reports, key=lambda item: item["fid"])
        stage_report: dict[str, object] = {"frozen_teacher": str(teacher), "development": reports,
                                           "selected_on_development": best,
                                           "selection_rule": "minimum FID on fixed 755-image training-only holdout"}
        summary["stages"][name] = stage_report
        write_json(ROOT / "summary.json", summary)
        if name == "n1":
            if best["step"] == 25_000:
                stage_report["official_selected"] = {"fid": 61.4540, "nfe": 1,
                                                      "checkpoint": "checkpoints/one_nfe.ckpt",
                                                      "source": str(BASE / "n1/official_evaluation/metrics.json")}
            else:
                exported = run / "selected_one_nfe.ckpt"
                command([sys.executable, "-u", "-m", "distillation.export", "--checkpoint", best["checkpoint"],
                         "--output", str(exported), "--mode", "one_nfe"], run / "export_console.log")
                official = run / "official_selected"
                if not (official / "metrics.json").exists():
                    command([sys.executable, "-u", "evaluate.py", "--model_checkpoint", str(exported),
                             "--evaluate_mode", "one_nfe", "--reference_dir", "results/reference",
                             "--output_dir", str(official), "--batch_size", "16"], official / "console.log")
                    matches = re.findall(r"one_nfe FID:\s*([0-9.eE+-]+)", (official / "console.log").read_text())
                    if not matches:
                        raise ValueError("Protected evaluator did not report FID")
                    write_json(official / "metrics.json", {"fid": float(matches[-1]), "nfe": 1,
                                                           "checkpoint": str(exported), "selected_step": best["step"],
                                                           "sample_count": 3020, "reference_count": 3020,
                                                           "selection_split": "development"})
                stage_report["official_selected"] = json.loads((official / "metrics.json").read_text())
        else:
            official = run / "official_selected"
            command([sys.executable, "-u", "-m", "distillation.evaluate", "--checkpoint", best["checkpoint"],
                     "--output-dir", str(official), "--reference-split", "official", "--seed", "42"],
                    official / "console.log")
            stage_report["official_selected"] = json.loads((official / "metrics.json").read_text())
        write_json(ROOT / "summary.json", summary)
        print(f"{name} selection complete: step={best['step']} holdout FID={best['fid']:.5f}", flush=True)
    print("Both low-NFE budget ablations completed", flush=True)


if __name__ == "__main__":
    main()
