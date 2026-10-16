"""Frozen teacher step/seed comparison on the predeclared training holdout."""
from __future__ import annotations
import json
from pathlib import Path
import statistics
import subprocess
import sys

RUN=Path('experiments/teacher_b64_fliponly_constant_100k')
OUTPUT=Path('experiments/teacher_sampling_steps_seeds')

def main() -> None:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    reports=[]
    for steps,seed in [(s,k) for s in (20,50,100) for k in (42,1042,2042)]+[(32,42)]:
        # The 32-step result anchors the power-of-two progressive-distillation grid.
        output=RUN/'development_evaluation/ema_ddim_clip1_s2' if (steps,seed)==(50,42) else OUTPUT/f'ddim{steps}_cfg2_seed{seed}'
        metrics=output/'metrics.json'
        if not metrics.exists() or json.loads(metrics.read_text()).get('fid') is None:
            command=[sys.executable,'-u','-m','teacher.evaluate','--checkpoint',str(RUN/'final.pt'),
                     '--output-dir',str(output),'--reference-split','development','--reference-dir',str(RUN/'development_reference'),
                     '--weights','ema','--sampler','ddim','--clip-denoised','--steps',str(steps),'--guidance-scale','2',
                     '--seed',str(seed),'--batch-size','16','--precision','bf16']
            if (output/'generated').exists() and len(list((output/'generated').glob('*.png')))==3020:
                command.append('--score-only')
            subprocess.run(command,check=True)
        result=json.loads(metrics.read_text())
        assert result['steps']==steps and result['seed']==seed and result['reference_split']=='development'
        reports.append(result)
        summary={str(s):{'mean_fid':statistics.mean(r['fid'] for r in reports if r['steps']==s),
                        'std_fid':statistics.stdev([r['fid'] for r in reports if r['steps']==s]) if sum(r['steps']==s for r in reports)>1 else None,
                        'seeds':[r['seed'] for r in reports if r['steps']==s]} for s in sorted({r['steps'] for r in reports})}
        (OUTPUT/'comparison.json').write_text(json.dumps({'reference_count':755,'guidance_scale':2,'sampling_seeds':[42,1042,2042],'note':'Disjoint batch-seed ranges; earlier adjacent-seed report is archived separately.','reports':reports,'summary':summary},indent=2)+'\n')
        print('Completed:',steps,seed,result['fid'],flush=True)

if __name__=='__main__': main()
