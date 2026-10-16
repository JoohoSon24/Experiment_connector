from pathlib import Path
import os,sys,json,subprocess
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch
from distillation.train import train
import run_dmd_selected_teacher as d
pf=ROOT/'experiments/preflight'
assert (ROOT/'experiments/preflight_pass.json').exists()
train('configs/budget.json',str(pf/'guidance/final.pt'),'n16',str(pf/'resume_n16'),steps_override=2,resume_checkpoint=str(pf/'n16/final.pt'))
d.INITIAL=pf/'n1.ckpt';d.ENDPOINT=pf/'n4.ckpt';models=d.setup()
rows=[d.update(models,1,.1),d.update(models,2,0.)]
assert all(torch.isfinite(torch.tensor(r['loss'])) for r in rows)
report=json.loads((ROOT/'experiments/preflight_pass.json').read_text());report['budget_resume']='full AdamW/RNG continuation 1->2 passed';report['DMD_updates']=rows
(ROOT/'experiments/preflight_pass.json').write_text(json.dumps(report,indent=2))
del models;torch.cuda.empty_cache()
# Dispose only generated smoke-test weights; metrics, logs, source snapshots and verification report remain.
for p in pf.rglob('*'):
 if p.is_file() and p.suffix in ('.pt','.ckpt'):p.unlink()
subprocess.run(['tmux','new-session','-d','-s','recipe_teacher25_distill_DMD4',sys.executable,'-u',str(ROOT/'scripts/session.py')],check=True)
print('LAUNCHED recipe_teacher25_distill_DMD4',flush=True)
