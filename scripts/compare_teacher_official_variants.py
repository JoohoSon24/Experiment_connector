"""Frozen official reporting only; no selection or training changes."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'experiments/teacher_variant_sequence'
CASES={'attention4_100k':STUDY/'attention4/attention4_100k/final.pt',
'attention4_150k':STUDY/'attention4/attention4_150k/final.pt',
'attention4_200k':STUDY/'attention4/attention4_200k/final.pt',
'dropout005_210k':STUDY/'dropout005/dropout005_210k/final.pt',
'attention32_100k':STUDY/'attention32/attention32_100k/final.pt'}
# Reuse the exact generated PNGs: only reference set changes, not sampler or checkpoint.
reports={}
for name,ckpt in CASES.items():
 arm=name.rsplit('_',1)[0]
 if arm.startswith('attention4'):arm='attention4'
 if arm.startswith('attention32'):arm='attention32'
 if arm.startswith('dropout005'):arm='dropout005'
 step=int(name.rsplit('_',1)[1][:-1])*1000
 source=STUDY/arm/f'evaluation/step_{step:06d}/ddim32_cfg2'
 out=STUDY/'official_comparison'/name
 out.mkdir(parents=True,exist_ok=True)
 old=json.loads((source/'metrics.json').read_text())
 expect={'checkpoint_step':step,'weights':'ema','seed':42,'batch_size':16,'precision':'bf16','steps':32,'guidance_scale':2.0,'sample_count':3020,'sampler':'ddim','clip_denoised':True}
 if any(old.get(k)!=v for k,v in expect.items()) or old['checkpoint']!=str(ckpt):raise ValueError('Generation conditions mismatch')
 generated=out/'generated'
 if not generated.exists():generated.symlink_to(source/'generated',target_is_directory=True)
 if generated.resolve()!=(source/'generated').resolve():raise ValueError('Different existing generated folder')
 metric=out/'metrics.json'
 if not metric.exists():
  meta=dict(old,reference_split='official',reference_count=3020,fid=None)
  metric.write_text(json.dumps(meta,indent=2)+'\n')
 if json.loads(metric.read_text()).get('fid') is None:
  with (out/'console.log').open('a') as log:
   subprocess.run([sys.executable,'-u','-m','teacher_variants.evaluate','--checkpoint',str(ckpt),
    '--output-dir',str(out),'--reference-split','official','--reference-dir','results/reference',
    '--weights','ema','--steps','32','--guidance-scale','2','--seed','42','--batch-size','16',
    '--precision','bf16','--sampler','ddim','--clip-denoised','--score-only'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 reports[name]=json.loads(metric.read_text())
 print(name,reports[name]['fid'],flush=True)
 (STUDY/'official_comparison/results.json').write_text(json.dumps(reports,indent=2)+'\n')
