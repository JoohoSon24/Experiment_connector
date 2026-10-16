"""Frozen-model distribution/precision audit. Never mutates active training."""
from pathlib import Path
import os,sys,json,contextlib
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image
from teacher.config import Config
from teacher.utils import load_checkpoint,make_model,write_json
from teacher.data import TrainImages
from teacher.diffusion import Diffusion
from distillation.core import coefficients,velocity_x0,merged_target,ddim_step,target_velocity,velocity_loss
OUT=ROOT/'experiments/n4_cause_analysis';OUT.mkdir(exist_ok=True)
torch.cuda.set_per_process_memory_fraction(.25);torch.manual_seed(123456);torch.cuda.manual_seed_all(123456)
def load(n):
 state=load_checkpoint(ROOT/f'experiments/recipe/{n}/final.pt','cpu');cfg=Config(**state['config']);m=make_model(cfg,151).cuda().eval().requires_grad_(False);m.load_state_dict(state['ema']);return m,cfg,state['sampling_times'].cuda()
parent,cfg,pt=load('n8');student,_,times=load('n4');alpha=Diffusion(1000,cfg.beta_start,cfg.beta_end,torch.device('cuda')).alpha_bar
@torch.inference_mode()
def target(x,labels,i,bf16=True):
 t=times[i].expand(len(labels));mt=pt[2*i+1].expand(len(labels));nt=(pt[2*i+2] if 2*i+2<len(pt) else torch.tensor(-1,device='cuda')).expand(len(labels))
 a,s=coefficients(alpha,t);am,sm=coefficients(alpha,mt);an,sn=coefficients(alpha,nt)
 with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
  first=velocity_x0(parent,x,t,labels,alpha);mid=ddim_step(x,first,a,s,am,sm);second=velocity_x0(parent,mid,mt,labels,alpha);end=ddim_step(mid,second,am,sm,an,sn)
 return merged_target(x,end,a,s,an,sn),end,nt
@torch.inference_mode()
def audit(x,labels,i,kind):
 t=times[i].expand(len(labels));a,s=coefficients(alpha,t);gt,end,nt=target(x,labels,i);an,sn=coefficients(alpha,nt)
 with torch.autocast('cuda',dtype=torch.bfloat16):pred=student(x,t,labels).float()
 v=target_velocity(x,gt,a,s);x0=a*x-s*pred;step=ddim_step(x,x0,a,s,an,sn);g32,_,_=target(x,labels,i,False)
 mse=float((x0-gt).square().mean());precision=float((gt-g32).square().mean())
 return {'distribution':kind,'time':int(times[i]),'count':len(labels),'x0_mse':mse,'endpoint_mse':float((step-end).square().mean()),'weighted_v_loss':float(velocity_loss(pred,v,a,s)),'target_outside_unit_fraction':float((gt.abs()>1).float().mean()),'target_abs_p99':float(torch.quantile(gt.abs().flatten(),.99)),'bf16_vs_fp32_target_mse':precision,'precision_error_over_student_error':precision/max(mse,1e-12),'merge_denominator':float((an-sn/s*a).flatten()[0])}
rows=[]
# Same deterministic64 images per split, processed8 at a time to bound diagnostic memory.
for split,manifest in [('train',cfg.train_manifest),('holdout',cfg.development_reference_manifest)]:
 ds=TrainImages(cfg.data_root,cfg.split_dir,64,151,split_file=manifest);idx=torch.randperm(len(ds),generator=torch.Generator().manual_seed(123456))[:64].tolist()
 for start in range(0,64,8):
  batch=[ds[k] for k in idx[start:start+8]];images=torch.stack([x for x,_ in batch]).cuda();labels=torch.tensor([y for _,y in batch],device='cuda')
  for i in range(4):
   t=times[i].expand(8);a,s=coefficients(alpha,t);x=torch.randn_like(images) if i==0 else a*images+s*torch.randn_like(images)
   rows.append(audit(x,labels,i,split+'_noised_real'))
  if split=='holdout':
   noise=torch.randn_like(images)
   for kind in ['parent8_rollout','student4_rollout']:
    x=noise.clone()
    for i in range(4):
     rows.append(audit(x,labels,i,kind))
     if kind=='parent8_rollout':_,x,_=target(x,labels,i)
     else:
      t=times[i].expand(8);nt=(times[i+1] if i<3 else torch.tensor(-1,device='cuda')).expand(8);a,s=coefficients(alpha,t);an,sn=coefficients(alpha,nt)
      with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):x0=velocity_x0(student,x,t,labels,alpha);x=ddim_step(x,x0,a,s,an,sn)
summary=[]
for kind in sorted({r['distribution'] for r in rows}):
 for t in times.tolist():
  group=[r for r in rows if r['distribution']==kind and r['time']==t]
  summary.append({'distribution':kind,'time':t,'samples':sum(r['count'] for r in group),**{k:sum(r[k]*r['count'] for r in group)/sum(r['count'] for r in group) for k in group[0] if k not in ['distribution','time','count']}})
write_json(OUT/'target_audit.json',{'sampling_times_n4':times.tolist(),'sampling_times_n8':pt.tolist(),'seed':123456,'samples_per_distribution':64,'interpretation':'small frozen diagnostic; no FID or causal proof; target p99 is mean of microbatch p99','summary':summary,'raw':rows})
print(json.dumps(summary),flush=True)
# Descriptive image statistics, not perceptual precision/recall.
folders={'legacy':ROOT/'experiments/compare_old_new_n4/legacy_generated','new':ROOT/'experiments/recipe/n4/eval_025000/generated'};stats={}
for name,folder in folders.items():
 edge=saturation=var=0.
 for c in range(151):
  stack=np.stack([np.asarray(Image.open(folder/f'{c:04d}_{i:02d}.png'),dtype=np.float32)/255 for i in range(20)])
  edge+=.5*(np.abs(np.diff(stack,axis=1)).mean()+np.abs(np.diff(stack,axis=2)).mean());saturation+=np.mean((stack<=1/255)|(stack>=254/255));var+=np.var(stack,axis=0).mean()
 stats[name]={'mean_adjacent_pixel_difference':float(edge/151),'fraction_near0_or1':float(saturation/151),'within_class_pixel_variance':float(var/151)}
write_json(OUT/'image_statistics.json',stats);print(json.dumps(stats),flush=True)
