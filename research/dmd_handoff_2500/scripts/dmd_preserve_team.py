"""Preserve team weights; one endpoint-distillation candidate, train-only selection."""
from pathlib import Path
import os,sys,json,time,hashlib,copy,math,traceback,shutil
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch
from model import ModelOneNFE,ModelFewNFE
from evaluate import generate_samples,compute_fid
from distillation.core import coefficients,target_velocity,velocity_loss
OUT=ROOT/'experiments/night_run';RUN=OUT/'dmd_preserve_team';RUN.mkdir(exist_ok=False)
REF=OUT/'reference_development';results=[];status='supplied one-NFE training-reference baseline';best=None;started=time.monotonic()
protected=json.loads((OUT/'audit_results.json').read_text())['protected_sha256']
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def report(error=None):
 lines=['# Night report — preserve supplied team models',f'Updated UTC: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}',f'Status: {status}',
 '\nUser explicitly superseded the extra fresh-lineage/755-image exclusion requirements with the supplied official course rules. Existing checkpoints are accepted as course-compliant on user attestation. Their exact training manifests are not independently embedded/verified; These scores use the same755-image development reference as the documented team holdout. Team configs/docs specify exclusion from training, accepted on user confirmation; original full teacher/student training artifacts are absent, so exclusion is not independently proven from the inference files. Scores are not official validation FID. Reference755 original-training images, generated3020 class-balanced images, batch16, seed42+batch index, bf16, clean-fid default. No official validation images/statistics/evaluation used.',
 '\nTeam architecture, velocity output and one/four-NFE sampler preserved. Initialization is exactly checkpoints/one_nfe.ckpt; frozen distillation teacher exactly checkpoints/few_nfe.ckpt. All models individually48,883,107 parameters. No external generator or new packages; original files/checkpoints remain unchanged.',
 '\nNew user-authorized substantial objective change: transfer the existing four-NFE teacher endpoint into the existing one-NFE student using matched initial noise and classes. DMD adds an estimated distribution gradient from a frozen real diffusion score and a trainable fake score. Weak endpoint regression stabilizes the change. This is a VP/v-generator adaptation, not an exact reproduction of the paper. DMD normalized score difference plus0.1-weight endpoint MSE, generator AdamW LR2e-5 constant; fake score AdamW LR2e-4 constant, no warmup, batch64/microbatch16, EMA0.999, clip grad1, seed42. Optimizer freshly initialized because supplied inference exports do not include optimizer state; this is fine-tuning, not exact historical optimizer continuation. Synthetic training pairs derive only from the course-compliant team teacher; neither reference nor validation images enter the loss.',
 '\nCandidate screening2.5K updates; extend5K only if FID below supplied baseline; extend10K only if5K FID improves its2.5K score. No further extension. This follow-up was explicitly authorized after the prior three-candidate phase ended. Only one new DMD candidate is run; all failed artifacts remain preserved. No fresh recreation stages continue.',
 '\n| Model | Added updates | Training-reference FID | Parameters | NFE | Peak allocated VRAM bytes | Training seconds |\n|---|---:|---:|---:|---:|---:|---:|']
 for r in results:lines.append(f'| {r["name"]} | {r["step"]} | {r["fid"]:.6f} | {r["parameter_count"]} | 1 | {r.get("peak_vram_bytes","n/a")} | {r.get("training_seconds",0):.2f} |')
 lines.append('\nSelected model: '+(json.dumps(best) if best else 'Pending supplied-model baseline.'))
 lines.append('\nSelection: minimum fixed training-reference FID, with the supplied baseline retained if the candidate degrades. Interpret holdout cleanliness under the documented/user-attested lineage, not as independently checkpoint-verified provenance. Official evaluation can only report a frozen model and never choose hyperparameters/candidates.')
 lines.append('\nNext experiments at mosttwo: compare an alternative teacher endpoint target under the same budget; replicate a winning candidate with another training seed.')
 lines.append('\nHistorical failure: new VP consistency sweep collapsed near uniform gray, FID515–578. It was not an improvement over the supplied progressive models and is excluded from this selection. Fresh teacher/recreation and all failure logs preserved. See QUALITY_DIAGNOSIS.md, CONSISTENCY_REPORT_BEFORE_RECOVERY.md and COURSE_RULES_OVERRIDE.md.')
 if error:lines.append('\nStopped: '+str(error))
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 shutil.copyfile(RUN/'REPORT.md',OUT/'NIGHT_REPORT.md')
 write(OUT/'run_status.json',{'status':status,'pid':os.getpid(),'records':results,'best':best,'error':error,'updated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
def load(cls,path):
 s=torch.load(path,map_location='cpu',weights_only=True);m=cls(device='cuda');m.load_state_dict(s['state_dict']);assert m.count_parameters()==48883107;return m.eval()
def verify_reference():
 manifest=json.loads((OUT/'reference_manifest.json').read_text())['images'];train=set((ROOT/'data/pokemon-generation-one-22k/train_split.txt').read_text().splitlines());validation=set((ROOT/'data/pokemon-generation-one-22k/val_split.txt').read_text().splitlines())
 assert len(manifest)==755 and {p.name for p in REF.glob('*.png')}=={r['reference'] for r in manifest}
 assert all(r['source'] in train and r['source'] not in validation and digest(REF/r['reference'])==r['sha256'] for r in manifest)
def evaluate(name,model,step,peak=0,training_seconds=0):
 verify_reference();folder=RUN/name;folder.mkdir();calls=[0]
 def hook(*args):calls[0]+=1
 handle=model.backbone.register_forward_hook(hook);model._batch_index=0;model.sampling_seed.fill_(42)
 generate_samples(model,folder/'generated',torch.arange(151).repeat_interleave(20),16,'cuda');handle.remove()
 assert calls[0]==189, calls
 fid=compute_fid(folder/'generated',REF,'cuda',16);assert math.isfinite(fid)
 r={'name':name,'step':step,'fid':fid,'reference_kind':'development_reference_team_documented_holdout','reference_count':755,'sample_count':3020,'seed':42,'batch_size':16,'precision':'bf16','parameter_count':model.count_parameters(),'nfe':1,'measured_batch_backbone_calls':calls[0],'peak_vram_bytes':peak,'training_seconds':training_seconds}
 write(folder/'metrics.json',r);results.append(r);write(RUN/'results.json',results);return r
try:
 assert torch.cuda.is_available();verify_reference()
 old_hashes={name:digest(ROOT/f'checkpoints/{name}.ckpt') for name in ['one_nfe','few_nfe']};write(RUN/'input_checkpoint_hashes.json',old_hashes)
 config={'method':'DMD-inspired VP adaptation with endpoint regularization','fake_score_lr':0.0002,'fake_score_warmup':200,'regularization_weight':0.1,'noise_t_range':[50,950],'references':['https://tianweiy.github.io/dmd/','https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html'],'candidate':'DMD distribution matching plus weak four-step endpoint regression','initialization':'checkpoints/one_nfe.ckpt','teacher':'checkpoints/few_nfe.ckpt','architecture_changes':False,'batch_size':64,'microbatch_size':16,'learning_rate':.00002,'optimizer':'AdamW','weight_decay':0,'ema_decay':.999,'warmup_steps':0,'learning_rate_schedule':'constant','seed':42,'screening_steps':2500,'continuation_steps':[5000,10000],'loss':'DMD normalized x0 score difference plus0.1 endpoint regression','training_data':'synthetic noise/class/endpoint pairs from user-confirmed provided-train-only teacher','reference':'755 development images; team configs/docs specify training exclusion, accepted on user confirmation; complete original artifacts not recovered','provenance_basis':'USER_CHECKPOINT_ATTESTATION.json and COURSE_RULES_OVERRIDE.md','real_score_checkpoint':'experiments/night_run/fresh_teacher/final.pt'};write(RUN/'config.json',config)
 student=load(ModelOneNFE,ROOT/'checkpoints/one_nfe.ckpt');report();baseline=evaluate('supplied_baseline',student,0);best=baseline;report()
 teacher=load(ModelFewNFE,ROOT/'checkpoints/few_nfe.ckpt').requires_grad_(False)
 ema=copy.deepcopy(student).eval().requires_grad_(False);student.train().requires_grad_(True)
 optimizer=torch.optim.AdamW(student.backbone.parameters(),lr=.00002,weight_decay=0)
 from teacher.config import Config
 from teacher.utils import make_model
 from teacher.cfg import apply_cfg
 real_state=torch.load(OUT/'fresh_teacher/final.pt',map_location='cpu',weights_only=True)
 real_cfg=Config(**real_state['config'])
 train_paths=Path(real_cfg.train_manifest).read_text().splitlines()
 assert real_state['train_paths_sha256']==hashlib.sha256('\n'.join(train_paths).encode()).hexdigest()
 real=make_model(real_cfg,151).cuda().eval().requires_grad_(False);real.load_state_dict(real_state['ema']);del real_state
 fake=copy.deepcopy(real).train().requires_grad_(True)
 assert sum(p.numel() for p in real.parameters())==48883107 and sum(p.numel() for p in fake.parameters())==48883107
 fake_optimizer=torch.optim.AdamW(fake.parameters(),lr=.0002,weight_decay=0)
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats();train_start=time.monotonic();previous=None
 def generator_image(noise,labels):
  times=torch.full_like(labels,999);a,s=coefficients(student.alpha_bar,times)
  with torch.autocast('cuda',dtype=torch.bfloat16):velocity=student.backbone(noise,times,labels)
  return (a*noise-s*velocity.float()).clamp(-1,1)
 def fit_fake(images,labels):
  fake_optimizer.zero_grad(set_to_none=True)
  times=torch.randint(50,951,(len(labels),),device='cuda');a,s=coefficients(student.alpha_bar,times);noise=torch.randn_like(images);noisy=a*images.detach()+s*noise
  with torch.autocast('cuda',dtype=torch.bfloat16):prediction=fake(noisy,times,labels);loss=(prediction.float()-noise).square().mean()
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(fake.parameters(),1,error_if_nonfinite=True);fake_optimizer.step()
  assert torch.cuda.max_memory_allocated()<20_000_000_000
  return float(loss.detach())
 status='DMD fake-score warmup200';report()
 for warmup in range(200):
  labels=torch.randint(151,(16,),device='cuda');noise=torch.randn((16,3,64,64),device='cuda')
  with torch.no_grad():images=generator_image(noise,labels)
  value=fit_fake(images,labels)
  if (warmup+1)%20==0:print(json.dumps({'warmup':warmup+1,'fake_score_loss':value,'peak_vram_bytes':torch.cuda.max_memory_allocated()}),flush=True)
 write(RUN/'fake_score_warmup.json',{'updates':200,'last_loss':value,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'real_parameters':48883107,'fake_parameters':48883107,'trained_only_on_generated_images':True})
 for step in range(1,10001):
  status=f'DMD existing1-NFE generator training {step}';optimizer.zero_grad(set_to_none=True);loss_sum=0
  for micro in range(4):
   labels=torch.randint(151,(16,),device='cuda');seed=1000000+step*4+micro
   generator=torch.Generator(device='cuda').manual_seed(seed);x=torch.randn((16,3,64,64),device='cuda',generator=generator)
   with torch.no_grad():detached_images=generator_image(x,labels)
   fake_loss=fit_fake(detached_images,labels)
   times=torch.randint(50,951,(16,),device='cuda');a,s=coefficients(student.alpha_bar,times);noise=torch.randn_like(x);noisy=a*detached_images+s*noise
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    uncond=real(noisy,times,torch.full_like(labels,151));cond=real(noisy,times,labels);real_eps=apply_cfg(uncond,cond,2).float();fake_eps=fake(noisy,times,labels).float()
    real_x0=(noisy-s*real_eps)/a;fake_x0=(noisy-s*fake_eps)/a
    normalizer=(detached_images-real_x0).abs().mean(dim=(1,2,3),keepdim=True).clamp_min(.001)
    distribution_gradient=(fake_x0-real_x0)/normalizer
    assert torch.isfinite(distribution_gradient).all()
    desired=teacher.sample((16,3,64,64),device='cuda',category=labels,seed=seed).clone()
   images=generator_image(x,labels)
   dmd_target=(images.detach()-distribution_gradient).detach()
   dmd_loss=.5*(images-dmd_target).square().mean()
   regression=(images-desired).square().mean()
   value=(dmd_loss+.1*regression)/4
   assert torch.isfinite(value);value.backward();loss_sum+=float(value.detach())
  grad=torch.nn.utils.clip_grad_norm_(student.parameters(),1,error_if_nonfinite=True);optimizer.step()
  with torch.no_grad():
   for target_param,param in zip(ema.parameters(),student.parameters(),strict=True):target_param.lerp_(param.detach(),.001)
  peak=torch.cuda.max_memory_allocated();assert peak<20_000_000_000
  elapsed=time.monotonic()-train_start;line={'step':step,'loss':loss_sum,'grad_norm':float(grad),'dmd_loss':float(dmd_loss.detach()),'dmd_regression':float(regression.detach()),'fake_score_loss':fake_loss,'peak_vram_bytes':peak,'elapsed_seconds':elapsed}
  with (RUN/'train.log').open('a') as f:f.write(json.dumps(line)+'\n')
  if step%20==0:print(json.dumps(line),flush=True)
  if step in [1,2,2500,5000,10000] or step%500==0:
   saved={'state_dict':ema.state_dict(),'student':student.state_dict(),'optimizer':optimizer.state_dict(),'fake_score':fake.state_dict(),'fake_optimizer':fake_optimizer.state_dict(),'step':step,'config':config,'input_checkpoint_hashes':old_hashes,'peak_vram_bytes':peak,'training_seconds':elapsed,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'provenance_basis':'user attestation; no independent holdout exclusion claimed'}
   torch.save(saved,RUN/'latest.tmp');os.replace(RUN/'latest.tmp',RUN/'latest.pt')
  if step==2 or step%100==0:report()
  if step in [2500,5000,10000]:
   saved_path=RUN/f'dmd_step_{step:06d}.ckpt';torch.save({'state_dict':ema.state_dict(),'config':config,'step':step,'input_checkpoint_hashes':old_hashes,'parameter_count':ema.count_parameters(),'nfe':1},saved_path)
   status=f'preserved team candidate training-reference evaluation {step}';report();row=evaluate(f'dmd_{step:06d}',ema,step,peak,elapsed);best=min([best,row],key=lambda r:r['fid']);report()
   if step==2500 and row['fid']>=baseline['fid']:break
   if step==5000 and row['fid']>=previous:break
   previous=row['fid']
  if time.monotonic()-started>6*3600:raise TimeoutError('Bounded six-hour endpoint experiment deadline')
 selected=ROOT/'checkpoints/one_nfe.ckpt' if best['step']==0 else RUN/f'dmd_step_{best["step"]:06d}.ckpt'
 export=RUN/'selected_one_nfe.ckpt';assert not export.exists();shutil.copyfile(selected,export)
 model=load(ModelOneNFE,export);calls=[0]
 def hook(*args):calls[0]+=1
 handle=model.backbone.register_forward_hook(hook)
 with torch.inference_mode():image=model.sample((1,3,64,64),device='cuda',category=torch.tensor([0],device='cuda'),seed=42)
 handle.remove();assert calls[0]==1 and torch.isfinite(image).all()
 best={**best,'selected_source':str(selected),'export':str(export),'export_sha256':digest(export),'export_measured_nfe':1};write(RUN/'selected_model.json',best)
 assert all(digest(ROOT/f'checkpoints/{name}.ckpt')==h for name,h in old_hashes.items())
 assert all(digest(p)==h for p,h in protected.items());status='complete DMD existing-team experiment';report()
except BaseException as error:
 status='stopped DMD existing-team experiment';report(repr(error));(RUN/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
