"""Build a source overlay and optional inference checkpoint archive for teammates."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "handoff"
BASE = "e3f5120000cd91037017d86f23993492195101a5"
SOURCE_DIRS = ("configs", "constraints", "distillation", "docs", "scripts", "teacher")
SOURCE_FILES = (".gitignore", "CODEX.md", "PROJECT_SPEC.md", "model.py",
                "data/development/seed42_holdout5/train.txt",
                "data/development/seed42_holdout5/holdout.txt",
                "data/development/seed42_holdout5/metadata.json")
WEIGHTS = {
    "checkpoints/one_nfe.ckpt": ROOT / "experiments/low_nfe_budget_100k/n1/selected_one_nfe.ckpt",
    "checkpoints/few_nfe.ckpt": ROOT / "checkpoints/few_nfe.ckpt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != BASE:
        raise ValueError(f"Unexpected upstream base commit: {actual}")
    upstream = subprocess.check_output(["git", "show", f"{BASE}:model.py"], cwd=ROOT, text=True)
    current = (ROOT / "model.py").read_text()
    if upstream.split("class ModelOneNFE")[0] != current.split("class ModelOneNFE")[0]:
        raise ValueError("Protected base Model class was changed")
    OUT.mkdir(parents=True, exist_ok=True)
    files = [ROOT / p for p in SOURCE_FILES]
    for dirname in SOURCE_DIRS:
        files += [p for p in (ROOT / dirname).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files = sorted(set(files))
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("Missing source file")
    allowed = {".py", ".json", ".md", ".sh", ".txt"}
    if any(p.suffix not in allowed and p.name not in {".gitignore"} for p in files):
        raise ValueError("Unexpected source artifact extension")
    manifest = {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}
    code_zip = OUT / "fastgen_code_overlay.zip"
    with ZipFile(code_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files:
            archive.write(p, p.relative_to(ROOT))
        archive.write(OUT / "TEAM_HANDOFF.md", "TEAM_HANDOFF.md")
        archive.writestr("OVERLAY_MANIFEST.json", json.dumps({"upstream_commit": BASE, "files": manifest}, indent=2) + "\n")
    weights_zip = OUT / "fastgen_inference_checkpoints.zip"
    with ZipFile(weights_zip, "w", compression=ZIP_STORED, allowZip64=True) as archive:
        for target, source in WEIGHTS.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, target)
    for p in (code_zip, weights_zip):
        with ZipFile(p) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Corrupt ZIP entry: {bad}")
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in (code_zip, weights_zip)), encoding="utf-8"
    )
    print(json.dumps({"upstream_commit": BASE, "source_file_count": len(files),
                      "code_zip_bytes": code_zip.stat().st_size, "weights_zip_bytes": weights_zip.stat().st_size,
                      "archives": [str(code_zip), str(weights_zip)]}, indent=2))


if __name__ == "__main__":
    main()
