from pathlib import Path
import sys,os,json,subprocess
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from distillation.refine import train
from teacher.utils import load_checkpoint,make_model,write_json
from teacher.config import Config
from teacher.diffusion import Diffusion
from distillation.core import coefficients,velocity_x0,ddim_step,merged_target
OUT=ROOT/'experiments/n4_followup_smoke';OUT.mkdir(exist_ok=True);torch.cuda.set_per_process_memory_fraction(.25)
source=load_checkpoint(ROOT/'experiments/recipe/n8/final.pt','cpu');parent=make_model(Config(**source['config']),151).cuda().eval().requires_grad_(False);parent.load_state_dict(source['ema']);pt=source['sampling_times'].cuda();del source
alpha=Diffusion(1000,.0001,.02,torch.device('cuda')).alpha_bar;labels=torch.tensor([0,13,50,133],device='cuda');index=torch.arange(4,device='cuda');x=torch.randn((4,3,64,64),device='cuda');t=pt[2*index];a,s=coefficients(alpha,t)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 midt=pt[2*index+1];endidx=2*index+2;endt=torch.where(endidx<len(pt),pt[endidx.clamp_max(len(pt)-1)],-1);am,sm=coefficients(alpha,midt);an,sn=coefficients(alpha,endt)
 first=velocity_x0(parent,x,t,labels,alpha);mid=ddim_step(x,first,a,s,am,sm);second=velocity_x0(parent,mid,midt,labels,alpha);end=ddim_step(mid,second,am,sm,an,sn);old=merged_target(x,end,a,s,an,sn)
 end=x
 for sub in range(2):
  ti=pt[2*index+sub];ni=2*index+sub+1;nt=torch.where(ni<len(pt),pt[ni.clamp_max(len(pt)-1)],-1);ai,si=coefficients(alpha,ti);an,sn=coefficients(alpha,nt);x0=velocity_x0(parent,end,ti,labels,alpha);end=ddim_step(end,x0,ai,si,an,sn)
 new=merged_target(x,end,a,s,an,sn)
 assert torch.equal(old,new)
del parent;torch.cuda.empty_cache();results=[]
for name,pname in [('n4_lr2e4','n8'),('n4_direct32_lr5e5','guidance')]:
 settings=json.loads((ROOT/f'configs/{name}.json').read_text());settings.update(batch_size=8,num_workers=0);cfgpath=OUT/f'{name}.json';write_json(cfgpath,settings)
 result=train(str(cfgpath),str(ROOT/f'experiments/recipe/{pname}/final.pt'),'n4',str(OUT/name),steps_override=2);assert torch.isfinite(torch.tensor(result['last_loss']));results.append(result)
write_json(ROOT/'experiments/n4_followup_preflight.json',{'two_step_target_exact_parity':True,'GPU_updates_per_target':2,'microbatch_for_smoke':8,'production_batch':64,'production_no_grad_target_evaluations_sequential':True,'results':results})
# Remove only generated smoke weights; retain metrics/settings/logs.
for p in OUT.rglob('*.pt'):p.unlink()
subprocess.run(['tmux','new-session','-d','-s','n4_followup_after_DMD4',sys.executable,'-u',str(ROOT/'scripts/n4_followup_session.py')],check=True)
print('PREFLIGHT PASSED; followup queue launched waiting for originalDMD4',flush=True)
