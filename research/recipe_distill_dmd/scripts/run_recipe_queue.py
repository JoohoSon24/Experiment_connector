"""Original handoff guidance/progressive/budget recipe followed by four DMD arms."""
import os,sys,json,copy,random,time,fcntl,subprocess,traceback,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch,wandb
from teacher.utils import write_json,load_checkpoint,save_checkpoint,save_image_grid
from distillation.train import train
from distillation.evaluate import Adapter
from distillation.export import export
from evaluate import generate_samples,compute_fid
from src.utils import count_parameters
import run_dmd_selected_teacher as d
OUT=ROOT/'experiments/recipe';TEACHER=d.REAL
ENTITY='fv2z6n929c-kaist-digital-humanities-and-social-sciences-';PROJECT='pokemon-diffusion'
ARMS={'A_reg01_lr2e5':(.1,2e-5),'B_reg00_lr2e5':(0.,2e-5),'C_reg03_lr2e5':(.3,2e-5),'D_reg01_lr1e5':(.1,1e-5)}
BASE='configs/base.json';BUDGET='configs/budget.json'
def status(phase,**kw):write_json(OUT/'status.json',{'phase':phase,**kw})
def wb(folder,name,config):
 folder.mkdir(parents=True,exist_ok=True);p=folder/'wandb_identity.json';identity=json.loads(p.read_text()) if p.exists() else {'id':uuid.uuid4().hex[:8]}
 run=wandb.init(entity=ENTITY,project=PROJECT,id=identity['id'],resume='allow',name=name,group='recipe4head150k_progressive_DMD',dir=str(folder),mode='online',config=config,save_code=False,settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
 write_json(p,{'id':run.id,'url':run.url});run.define_metric('step');run.define_metric('train/*',step_metric='step');run.define_metric('eval/*',step_metric='step');return run

def evaluate_live(ema,alpha,times,clip,folder,official=False):
 folder.mkdir(parents=True,exist_ok=True);mp=folder/('official_metrics.json' if official else 'metrics.json')
 if mp.exists():return json.loads(mp.read_text())
 rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),random.getstate())
 try:
  model=Adapter(ema,alpha,times,42,clip).cuda().eval();gen=folder/'generated'
  if not gen.exists():generate_samples(model,gen,torch.arange(151).repeat_interleave(20),16,'cuda')
  expected={f'{c:04d}_{i:02d}.png' for c in range(151) for i in range(20)}
  if {x.name for x in gen.glob('*.png')}!=expected:raise RuntimeError('Incomplete FID generation preserved')
  fid=compute_fid(gen,d.OFFICIAL if official else d.REF,'cuda',16)
  row={'fid':fid,'reference_split':'official' if official else 'development','reference_count':3020 if official else 755,'sample_count':3020,'seed':42,'batch_size':16,'precision':'bf16','nfe':len(times),'parameter_count':count_parameters(model,trainable_only=False)};write_json(mp,row);return row
 finally:
  torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);random.setstate(rng[2])

def compact(path):
 state=load_checkpoint(path,'cpu');state={k:v for k,v in state.items() if k not in {'model','optimizer','torch_rng','python_rng','cuda_rng'}};state['retention']='Completed parent EMA + full lineage; active/latest and final budget checkpoints retain exact optimizer/RNG resume'
 save_checkpoint(path,state)
 latest=path.with_name('latest.pt')
 if latest.exists():latest.unlink()

def stage(name,parent,config=BASE,resume=None):
 folder=OUT/name;final=folder/'final.pt'
 if (folder/'completed.json').exists():return json.loads((folder/'completed.json').read_text())
 settings=json.loads(Path(config).read_text());end=settings['guidance_updates'] if name=='guidance' else settings['progressive_updates']
 stage_name=name.split('_')[0];run=wb(folder,'recipe4head150k_'+name,{'teacher':str(parent),'root_teacher':str(TEACHER),'heads':4,'stage':stage_name,**settings})
 rows=json.loads((folder/'fid_history.json').read_text()) if (folder/'fid_history.json').exists() else []
 def hook(row,ema,alpha,times,clip):
  step=row['step']
  if step%20==0 or step==1:run.log({'step':step,**{'train/'+k:v for k,v in row.items() if k!='step'}});status('distillation_training',stage=name,step=step,target=end,wandb=run.url,dmd_queued=list(ARMS))
  if step%1000==0:
   p=folder/f'samples_step_{step:07d}.png'
   if p.exists():run.log({'step':step,'qual/fixed_seed42':wandb.Image(str(p))})
  points=(50000,75000,100000) if '_budget' in name else tuple(range(10000,end+1,10000))+(end,)
  if step in points:
   status('distillation_fid',stage=name,step=step,wandb=run.url,dmd_queued=list(ARMS))
   m=evaluate_live(ema,alpha,times,clip,folder/f'eval_{step:06d}');rows.append({'step':step,**m});write_json(folder/'fid_history.json',rows);run.log({'step':step,'eval/development_fid':m['fid']})
 try:
  if not final.exists():train(config,str(parent),stage_name,str(folder),resume_checkpoint=str(resume) if resume else None,progress_hook=hook)
  state=load_checkpoint(final,'cpu')
  from teacher.config import Config
  from teacher.utils import make_model
  from teacher.diffusion import Diffusion
  cfg=Config(**state['config']);ema=make_model(cfg,151).cuda().eval().requires_grad_(False);ema.load_state_dict(state['ema']);alpha=Diffusion(1000,cfg.beta_start,cfg.beta_end,torch.device('cuda')).alpha_bar;times=state['sampling_times'].cuda();clip=state['clip_denoised'];del state
  last=evaluate_live(ema,alpha,times,clip,folder/f'eval_{end:06d}')
  if not any(x['step']==end for x in rows):rows.append({'step':end,**last});write_json(folder/'fid_history.json',rows)
  result={'stage':name,'final':str(final),'development':last,'milestones':rows,'wandb':run.url};write_json(folder/'completed.json',result);run.summary['phase']='completed';run.finish();del ema;torch.cuda.empty_cache();return result
 except BaseException as e:run.finish(exit_code=1);raise

def run_dmd():
 d.INITIAL=ROOT/'checkpoints/selected_one_nfe.ckpt';d.ENDPOINT=ROOT/'checkpoints/new_teacher_four_nfe.ckpt'
 for arm,(reg,lr) in ARMS.items():
  folder=OUT/'dmd'/arm
  if (folder/'completed.json').exists():continue
  if (folder/'train.log').exists():raise RuntimeError('Interrupted DMD arm preserved; exact resume unavailable')
  run=wb(folder,'recipe4head150k_DMD_'+arm,{'method':'supplied DMD-inspired VP recipe','root_teacher':str(TEACHER),'initial':str(d.INITIAL),'endpoint':str(d.ENDPOINT),'heads':4,'G_lr':lr,'reg':reg,'fake_lr':2e-4,'batch':64,'microbatch':16,'steps':2500,'fake_warmup':200,'EMA':.999})
  models=d.setup();student,endpoint,ema,real,fake,opt,fopt=models
  for g in opt.param_groups:g['lr']=lr
  baseline=d.score(ema,OUT/'dmd/baseline');best={'step':0,**baseline};bestpath=d.INITIAL;rows=[best];run.log({'step':0,'eval/development_fid':baseline['fid']})
  torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  try:
   status('DMD_fake_warmup',arm=arm,wandb=run.url)
   for _ in range(200):
    labels=torch.randint(151,(16,),device='cuda');x=torch.randn((16,3,64,64),device='cuda')
    with torch.no_grad():im=d.image(student,x,labels)
    d.fit_fake(student,fake,fopt,im,labels)
   for step in range(1,2501):
    row=d.update(models,step,reg)
    with (folder/'train.log').open('a') as f:f.write(json.dumps(row)+'\n')
    if step%20==0 or step==1:run.log({'step':step,**{'train/'+k:v for k,v in row.items() if k!='step'}});status('DMD_training',arm=arm,step=step,wandb=run.url)
    if step%100==0:
     with torch.inference_mode():im=ema.sample((8,3,64,64),device='cuda',category=torch.arange(8,device='cuda'),seed=42)
     p=folder/f'qual_{step:06d}.png';save_image_grid(im,p);run.log({'step':step,'qual/fixed_seed42':wandb.Image(str(p))})
    if step in (500,1000,2500):
     m=d.score(ema,folder/f'eval_{step:06d}');rows.append({'step':step,**m});run.log({'step':step,'eval/development_fid':m['fid']})
     if m['fid']<best['fid']:
      best={'step':step,**m};bestpath=folder/'best_one_nfe.ckpt';save_checkpoint(bestpath,{'state_dict':ema.state_dict(),'step':step,'heads':4,'nfe':1,'root_teacher':str(TEACHER),'initial':str(d.INITIAL),'reg':reg,'lr':lr})
     write_json(folder/'results.json',rows)
   selected_dir=OUT/'dmd/baseline' if best['step']==0 else folder/f'eval_{best["step"]:06d}'
   official=d.score(ema,selected_dir,official=True);run.log({'step':2500,'eval/selected_official_fid':official['fid']});write_json(folder/'completed.json',{'selected':best,'checkpoint':str(bestpath),'official':official,'milestones':rows,'wandb':run.url});run.finish()
  except BaseException:run.finish(exit_code=1);raise
  del models,student,endpoint,ema,real,fake,opt,fopt;torch.cuda.empty_cache()

def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 plan={'root_teacher':str(TEACHER),'teacher_sha256':d.digest(TEACHER),'teacher_official_fid':25.49841493955,'heads':4,'guidance_updates':50000,'progressive_steps':[16,8,4,2,1],'updates_per_stage':25000,'budget_extensions':{'n1':100000,'n2':100000},'DMD_queue':ARMS,'DMD_steps_each':2500,'selection':'development755 only; official3020 reporting','quality_gate':'guidance development FID <= teacher32 41.73750985 + 15','retention':'completed intermediate stages EMA only; active/latest and final n1/n2 budget full optimizer/RNG; DMD exports inference-only'}
 write_json(OUT/'plan.json',plan)
 try:
  parent=TEACHER
  for name in ['guidance','n16','n8','n4','n2','n1']:
   result=stage(name,parent)
   if name=='guidance' and result['development']['fid']>56.73750985:raise RuntimeError('Guidance quality gate failed; no progressive/DMD launched')
   parent=OUT/name/'final.pt'
   if name not in ('n1','n2') and 'optimizer' in load_checkpoint(parent,'cpu'):compact(parent)
  export(str(OUT/'n4/final.pt'),str(ROOT/'checkpoints/new_teacher_four_nfe.ckpt'),'few_nfe')
  selected={}
  for name,teacher_name in [('n1','n2'),('n2','n4')]:
   result=stage(name+'_budget',OUT/teacher_name/'final.pt',BUDGET,OUT/name/'final.pt')
   original=json.loads((OUT/name/'completed.json').read_text());candidates=[{'step':25000,**original['development']}]+result['milestones'];best=min(candidates,key=lambda x:x['fid'])
   chosen=OUT/name/'final.pt' if best['step']==25000 else OUT/(name+'_budget')/('final.pt' if best['step']==100000 else f'step_{best["step"]:06d}.pt')
   selected[name]={'best':best,'checkpoint':str(chosen)};write_json(OUT/'selected_students.json',selected)
   if name=='n1':export(str(chosen),str(ROOT/'checkpoints/selected_one_nfe.ckpt'),'one_nfe')
   if 'optimizer' in load_checkpoint(OUT/name/'final.pt','cpu'):compact(OUT/name/'final.pt')
  # Record matched official FID for selected students and n4 before DMD.
  for name,path in [('n1',selected['n1']['checkpoint']),('n2',selected['n2']['checkpoint']),('n4',str(OUT/'n4/final.pt'))]:
   dest=OUT/'official_students'/name
   subprocess.run([sys.executable,'-u','-m','distillation.evaluate','--checkpoint',path,'--output-dir',str(dest),'--reference-split','official'],check=True)
   measured=json.loads((dest/'metrics.json').read_text());folder=OUT/(name+'_budget') if name in ('n1','n2') else OUT/name
   report_run=wb(folder,'recipe4head150k_'+folder.name,{});report_run.log({'step':100000 if name in ('n1','n2') else 25000,'eval/selected_official_fid':measured['fid']});report_run.finish()
  run_dmd();status('completed',selected_students=selected,dmd=list(ARMS))
 except BaseException as e:status('failed',error=str(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
