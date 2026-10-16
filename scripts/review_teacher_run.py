"""Ordered, training-holdout-only sampler/EMA comparisons after the fresh run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teacher.utils import load_checkpoint, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='experiments/teacher_b64_100k')
    args=parser.parse_args();run=Path(args.run_dir)
    checkpoint=run/'final.pt'
    state=load_checkpoint(checkpoint,'cpu')
    if state['step']!=100000:
        raise ValueError('Expected completed 100,000-update checkpoint')
    del state
    reports=[]
    settings=[(weights,method,True) for weights in ('model','ema','ema:0.999') for method in ('ddim','ddpm')]
    settings.append(('ema','ddim',False))
    # Full CFG sweep stays on the unseen subset of official training.
    settings.extend(('ema','ddpm',True,scale) for scale in (0,0.5,2,3,4,6))
    # Matched DDIM/DDPM comparisons at guided scales; same checkpoint and holdout.
    settings.extend(('ema','ddim',True,scale) for scale in (2,3))
    for setting in settings:
        weights,method,clip=setting[:3];scale=setting[3] if len(setting)==4 else 1
        name=f'{weights.replace(":","_")}_{method}_clip{int(clip)}_s{scale:g}'
        output=run/'development_evaluation'/name
        args_list=[sys.executable,'-u','-m','teacher.evaluate','--checkpoint',str(checkpoint),
                   '--output-dir',str(output),'--reference-split','development',
                   '--reference-dir',str(run/'development_reference'),'--weights',weights,
                   '--sampler',method,'--steps','50','--guidance-scale',str(scale),
                   '--seed','42','--batch-size','16','--precision','bf16']
        if clip: args_list.append('--clip-denoised')
        if (output/'metrics.json').exists() and json.loads((output/'metrics.json').read_text()).get('fid') is not None:
            reports.append(json.loads((output/'metrics.json').read_text()));continue
        if (output/'generated').exists() and len(list((output/'generated').glob('*.png')))==3020:
            args_list.append('--score-only')
        print('Running development comparison:',name,flush=True)
        subprocess.run(args_list,check=True)
        reports.append(json.loads((output/'metrics.json').read_text()))
        write_json(run/'development_comparison.json',{'reference':'755 unseen official-training images',
                   'note':'Small-reference FID is biased; only compare frozen methods within this identical development split. Not an official validation score.',
                   'reports':reports})
    print('Sampler and EMA comparisons complete. Inspect the development report before choosing subsequent ablations.',flush=True)


if __name__=='__main__': main()
