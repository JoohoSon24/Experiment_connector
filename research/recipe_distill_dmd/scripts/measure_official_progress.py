"""Official reporting on immutable existing evaluation PNGs; does not select weights."""
import os,sys,json,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,wandb
from evaluate import compute_fid
from teacher.utils import write_json
OUT=ROOT/'experiments/recipe';REPORT=OUT/'official_progress';REPORT.mkdir(exist_ok=True)
torch.cuda.set_per_process_memory_fraction(.25)
run=wandb.init(entity='fv2z6n929c-kaist-digital-humanities-and-social-sciences-',project='pokemon-diffusion',name='recipe4head150k_official_progress',group='recipe4head150k_progressive_DMD',dir=str(REPORT),mode='online',save_code=False,settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
write_json(REPORT/'wandb_identity.json',{'id':run.id,'url':run.url})
expected={f'{c:04d}_{i:02d}.png' for c in range(151) for i in range(20)}
reference=ROOT/'results/reference';assert len(list(reference.glob('*.png')))==3020
rows={}
def measure(name,step,nfe):
 source=OUT/name/f'eval_{step:06d}';generated=source/'generated';out=REPORT/f'{name}_{step:06d}.json'
 assert {p.name for p in generated.glob('*.png')}==expected
 development=json.loads((source/'metrics.json').read_text())
 if out.exists():result=json.loads(out.read_text())
 else:
  score=compute_fid(generated,reference,'cuda',16)
  result={'stage':name,'step':step,'nfe':nfe,'fid':score,'development_fid':development['fid'],'reference_split':'official','reference_count':3020,'sample_count':3020,'seed':42,'batch_size':16,'precision':'bf16','weights':'EMA','source_generated':str(generated),'checkpoint':str(OUT/name/'final.pt') if step==(50000 if name=='guidance' else 25000) else None,'note':'Reporting only; no checkpoint selected on official FID'}
  write_json(out,result)
 rows[name]=result;write_json(REPORT/'summary.json',{'results':rows,'wandb':run.url});run.log({f'eval/{name}_official_fid':result['fid'],f'eval/{name}_development_fid':result['development_fid']});print(json.dumps(result),flush=True);torch.cuda.empty_cache()
try:
 for name,step,nfe in [('guidance',50000,32),('n16',25000,16),('n8',25000,8)]:measure(name,step,nfe)
 measure('n4',20000,4)
 deadline=time.monotonic()+1200
 while not (OUT/'n4/completed.json').exists():
  status=json.loads((OUT/'status.json').read_text())
  if status['phase']=='failed':raise RuntimeError('Main training failed: '+str(status))
  if time.monotonic()>deadline:raise TimeoutError('n4 final did not complete within20minutes; intermediate official report preserved')
  time.sleep(5)
 measure('n4',25000,4);run.summary['phase']='completed';run.finish();write_json(REPORT/'completed.json',{'results':rows,'wandb':run.url})
except BaseException as e:run.finish(exit_code=1);write_json(REPORT/'failed.json',{'error':str(e)});raise
