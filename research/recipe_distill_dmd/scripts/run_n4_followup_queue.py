"""Append three paired n4 refinements after the existing distillation/DMD queue."""
import os,sys,json,time,fcntl,copy,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,wandb
import run_recipe_queue as q
from distillation.refine import train
from teacher.utils import write_json,load_checkpoint,save_checkpoint
OUT=ROOT/'experiments/n4_followup';MAIN=ROOT/'experiments/recipe'
ARMS=[('n4_lr2e4','n8'),('n4_lr5e5','n8'),('n4_direct32_lr5e5','guidance')]
def status(phase,**kw):write_json(OUT/'status.json',{'phase':phase,'existing_queue':str(MAIN/'status.json'),'queued':[a for a,_ in ARMS],**kw})
def arm(name,parent_name):
 folder=OUT/name;folder.mkdir(exist_ok=True)
 if (folder/'completed.json').exists():return json.loads((folder/'completed.json').read_text())
 settings=json.loads((ROOT/f'configs/{name}.json').read_text());parent=MAIN/parent_name/'final.pt';initial=Path(settings['initial_student_checkpoint'])
 run=q.wb(folder,'followup_'+name,{**settings,'parent':str(parent),'initialization':'same n4 EMA25K + fresh AdamW','group_note':'appended after existingDMD4'})
 history=json.loads((folder/'fid_history.json').read_text()) if (folder/'fid_history.json').exists() else [{'step':0,**json.loads((MAIN/'n4/eval_025000/metrics.json').read_text())}]
 best=min(history,key=lambda x:x['fid']);selected=initial if best['step']==0 else folder/'best_selected.pt'
 meta=load_checkpoint(initial,'cpu');meta={k:v for k,v in meta.items() if k not in {'model','optimizer','torch_rng','python_rng','cuda_rng','ema'}}
 run.log({'step':0,'eval/baseline_development_fid':history[0]['fid']})
 def hook(row,ema,alpha,times,clip):
  nonlocal best,selected
  step=row['step']
  if step%20==0 or step==1:
   run.log({'step':step,**{'train/'+k:v for k,v in row.items() if k!='step'}});status('training',arm=name,step=step,target=10000,wandb=run.url)
  if step%1000==0:
   p=folder/f'samples_step_{step:07d}.png';run.log({'step':step,'qual/fixed_seed42':wandb.Image(str(p))})
  if step in settings['evaluation_steps']:
   status('evaluating',arm=name,step=step,wandb=run.url)
   m=q.evaluate_live(ema,alpha,times,False,folder/f'eval_{step:06d}');history.append({'step':step,**m});run.log({'step':step,'eval/development_fid':m['fid']})
   if m['fid']<best['fid']:
    best={'step':step,**m};selected=folder/'best_selected.pt';save_checkpoint(selected,{**meta,'ema':ema.state_dict(),'step':step,'sampling_times':times.cpu(),'settings':settings,'teacher_checkpoint':str(parent),'initial_student_checkpoint':str(initial),'clip_denoised':False})
   write_json(folder/'fid_history.json',history)
 try:
  if not (folder/'final.pt').exists():train(str(ROOT/f'configs/{name}.json'),str(parent),'n4',str(folder),progress_hook=hook)
  # Saved EMA export remains available for each evaluated best step; full active checkpoint supports exact resume.
  state=load_checkpoint(folder/'final.pt','cpu')
  from teacher.config import Config
  from teacher.utils import make_model
  from teacher.diffusion import Diffusion
  net=make_model(Config(**state['config']),151).cuda().eval().requires_grad_(False);chosen=load_checkpoint(selected,'cpu');net.load_state_dict(chosen['ema']);times=chosen['sampling_times'].cuda();alpha=Diffusion(1000,.0001,.02,torch.device('cuda')).alpha_bar;del chosen,state
  generated=MAIN/'n4/eval_025000' if best['step']==0 else folder/f'eval_{best["step"]:06d}'
  official=q.evaluate_live(net,alpha,times,False,generated,official=True);run.log({'step':10000,'eval/selected_official_fid':official['fid']})
  result={'arm':name,'selected':best,'selected_checkpoint':str(selected),'official':official,'milestones':history,'wandb':run.url,'parent':str(parent),'initial':str(initial)};write_json(folder/'completed.json',result)
  run.finish();del net;torch.cuda.empty_cache();q.compact(folder/'final.pt');return result
 except BaseException:run.finish(exit_code=1);raise

def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write_json(OUT/'plan.json',{'order':[a for a,_ in ARMS],'start_after':'existing recipe + DMD4 completed','steps_each':10000,'LR_comparison':['n4_lr2e4','n4_lr5e5'],'target_comparison':['n4_lr5e5','n4_direct32_lr5e5'],'baseline':'same n4 EMA25K, fresh AdamW','quality_selection':'development755, official3020 reporting','eval_steps':[2500,5000,10000]})
 status('waiting_for_existing_queue')
 while True:
  state=json.loads((MAIN/'status.json').read_text())
  if state['phase']=='completed':break
  if state['phase']=='failed':status('waiting_for_existing_queue_recovery',existing_error=state.get('error'))
  time.sleep(15)
 if not (ROOT/'experiments/n4_followup_preflight.json').exists():raise RuntimeError('Followup GPU preflight required')
 results={}
 for name,parent in ARMS:
  results[name]=arm(name,parent);write_json(OUT/'results.json',results)
 status('completed',results=results)
if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=str(e),traceback=traceback.format_exc());raise
