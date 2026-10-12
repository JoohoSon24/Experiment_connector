"""Report frozen DDPM CFG 1/2 settings using existing generation artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    reports = {}
    for scale in (1, 2):
        source = run / 'development_evaluation' / f'ema_ddpm_clip1_s{scale}'
        output = run / ('official_fid_ddpm50' if scale == 1 else 'official_fid_ddpm50_cfg2')
        previous = json.loads((source / 'metrics.json').read_text())
        assert previous['reference_split'] == 'development'
        assert previous['weights'] == 'ema' and previous['guidance_scale'] == scale
        target = output / 'generated'
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted((source / 'generated').glob('*.png')):
            destination = target / path.name
            if not destination.exists():
                os.link(path, destination)
            elif not os.path.samefile(path, destination):
                raise ValueError(f'Unexpected existing image: {destination}')
        metadata_path = output / 'metrics.json'
        if not metadata_path.exists():
            previous.update(reference_split='official', fid=None)
            metadata_path.write_text(json.dumps(previous, indent=2) + '\n')
        if json.loads(metadata_path.read_text()).get('fid') is None:
            command = [sys.executable, '-u', '-m', 'teacher.evaluate',
                       '--checkpoint', str(run / 'final.pt'), '--output-dir', str(output),
                       '--reference-split', 'official', '--weights', 'ema', '--sampler', 'ddpm',
                       '--clip-denoised', '--steps', '50', '--guidance-scale', str(scale),
                       '--seed', '42', '--batch-size', '16', '--precision', 'bf16', '--score-only']
            with (output / 'console.log').open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(metadata_path.read_text())
        report['generation_reused_from'] = str(source)
        report['selection_source'] = 'Frozen CFG1 baseline; CFG2 selected on training-only holdout, before official reporting'
        metadata_path.write_text(json.dumps(report, indent=2) + '\n')
        reports[f'cfg{scale}'] = report
        print(f'Official CFG {scale}: FID={report["fid"]:.6f}, NFE={report["nfe_per_sample"]}', flush=True)
    metrics_path = run / 'metrics.json'
    metrics = json.loads(metrics_path.read_text())
    metrics.update(fid=reports['cfg1']['fid'], fid_reports=reports)
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
