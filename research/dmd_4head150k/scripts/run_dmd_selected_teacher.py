"""Two DMD-inspired ablations using the selected four-head score teacher."""
from pathlib import Path
import os,sys,json,time,copy,hashlib,fcntl,uuid,math
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch
from model import ModelOneNFE,ModelFewNFE
from evaluate import generate_samples,compute_fid
from distillation.core import coefficients
from teacher.cfg import apply_cfg
from teacher.utils import save_image_grid
from teacher_variants.config import Config
from teacher_variants.utils import make_model

WORK=Path('/root/Diffusion/workspace_distillation')
REAL=WORK/'experiments/teacher_variant_sequence/attention4/attention4_150k/final.pt'
INITIAL=WORK/'checkpoints/one_nfe.ckpt'
ENDPOINT=WORK/'checkpoints/few_nfe.ckpt'
REF=WORK/'data/development/seed42_holdout5/reference'
OFFICIAL=WORK/'results/reference'
STUDY=ROOT/'experiments/dmd_4head150k'
ARMS={'endpoint01':0.1,'endpoint00':0.0}

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(cls,path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    m=cls(device='cuda');m.load_state_dict(state['state_dict']);assert m.count_parameters()==48883107
    return m.eval()

def setup():
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    student=load(ModelOneNFE,INITIAL)
    endpoint=load(ModelFewNFE,ENDPOINT).requires_grad_(False)
    ema=copy.deepcopy(student).eval().requires_grad_(False)
    student.train().requires_grad_(True)
    state=torch.load(REAL,map_location='cpu',weights_only=True)
    cfg=Config(**state['config'])
    assert cfg.num_attention_heads==4 and state['step']==150000
    paths=[p.strip() for p in (WORK/cfg.train_manifest).read_text().splitlines() if p.strip()]
    assert state['train_paths_sha256']==hashlib.sha256('\n'.join(paths).encode()).hexdigest()
    mapping=json.loads((WORK/cfg.split_dir/'category_to_id.json').read_text())
    assert state['class_names']==[n for n,_ in sorted(mapping.items(),key=lambda p:p[1])]
    assert mapping==json.loads((ROOT/'data/pokemon-generation-one-22k/category_to_id.json').read_text())
    real=make_model(cfg,151).cuda().eval().requires_grad_(False);real.load_state_dict(state['ema']);del state
    fake=copy.deepcopy(real).train().requires_grad_(True)
    assert all(sum(p.numel() for p in m.parameters())==48883107 for m in (real,fake))
    from teacher.diffusion import Diffusion
    assert torch.equal(student.alpha_bar,Diffusion(1000,cfg.beta_start,cfg.beta_end,torch.device('cuda')).alpha_bar)
    opt=torch.optim.AdamW(student.backbone.parameters(),lr=2e-5,weight_decay=0)
    fopt=torch.optim.AdamW(fake.parameters(),lr=2e-4,weight_decay=0)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
    return student,endpoint,ema,real,fake,opt,fopt

def image(student,x,labels):
    t=torch.full_like(labels,999);a,s=coefficients(student.alpha_bar,t)
    with torch.autocast('cuda',dtype=torch.bfloat16):v=student.backbone(x,t,labels)
    return (a*x-s*v.float()).clamp(-1,1)

def fit_fake(student,fake,opt,images,labels):
    opt.zero_grad(set_to_none=True)
    t=torch.randint(50,951,(len(labels),),device='cuda');a,s=coefficients(student.alpha_bar,t)
    noise=torch.randn_like(images);noisy=a*images.detach()+s*noise
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=fake(noisy,t,labels);loss=(pred.float()-noise).square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('fake loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(fake.parameters(),1,error_if_nonfinite=True);opt.step()
    return float(loss.detach())

def update(models,step,reg,micros=4):
    student,endpoint,ema,real,fake,opt,fopt=models
    opt.zero_grad(set_to_none=True);loss_sum=0.;dl=rg=0.
    for micro in range(micros):
        labels=torch.randint(151,(16,),device='cuda');seed=1000000+step*4+micro
        x=torch.randn((16,3,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
        with torch.no_grad():detached=image(student,x,labels)
        fl=fit_fake(student,fake,fopt,detached,labels)
        t=torch.randint(50,951,(16,),device='cuda');a,s=coefficients(student.alpha_bar,t)
        noise=torch.randn_like(x);noisy=a*detached+s*noise
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            uncond=real(noisy,t,torch.full_like(labels,151));cond=real(noisy,t,labels)
            re=apply_cfg(uncond,cond,2).float();fe=fake(noisy,t,labels).float()
            rx=(noisy-s*re)/a;fx=(noisy-s*fe)/a
            norm=(detached-rx).abs().mean(dim=(1,2,3),keepdim=True).clamp_min(.001)
            gradient=(fx-rx)/norm
            if not torch.isfinite(gradient).all():raise FloatingPointError('DMD gradient')
            desired=endpoint.sample((16,3,64,64),device='cuda',category=labels,seed=seed).clone() if reg else None
        images=image(student,x,labels)
        dmd=.5*(images-(images.detach()-gradient).detach()).square().mean()
        regression=(images-desired).square().mean() if reg else torch.zeros((),device='cuda')
        value=(dmd+reg*regression)/micros
        if not torch.isfinite(value):raise FloatingPointError('generator loss')
        value.backward();loss_sum+=float(value.detach());dl+=float(dmd.detach())/micros;rg+=float(regression.detach())/micros
    gn=torch.nn.utils.clip_grad_norm_(student.parameters(),1,error_if_nonfinite=True);opt.step()
    with torch.no_grad():
        for q,p in zip(ema.parameters(),student.parameters(),strict=True):q.lerp_(p.detach(),.001)
    peak=torch.cuda.max_memory_allocated()
    if peak>=20_000_000_000:raise RuntimeError('Peak allocation >=20GB')
    return {'step':step,'loss':loss_sum,'dmd_loss':dl,'endpoint_mse':rg,'fake_score_loss':fl,'grad_norm':float(gn),'peak_vram_bytes':peak}

def score(model,out,official=False):
    generated=out/'generated';out.mkdir(parents=True,exist_ok=True)
    if not generated.exists():
        calls=[0]
        hook=model.backbone.register_forward_hook(lambda *a:calls.__setitem__(0,calls[0]+1))
        model._batch_index=0;model.sampling_seed.fill_(42)
        generate_samples(model,generated,torch.arange(151).repeat_interleave(20),16,'cuda');hook.remove()
        assert calls[0]==189
    assert len(list(generated.glob('*.png')))==3020
    ref=OFFICIAL if official else REF
    assert len(list(ref.glob('*.png')))==(3020 if official else 755)
    mf=out/('official_metrics.json' if official else 'metrics.json')
    if mf.exists():return json.loads(mf.read_text())
    val=compute_fid(generated,ref,'cuda',16)
    result={'fid':val,'reference_split':'official' if official else 'development','reference_count':3020 if official else 755,
            'sample_count':3020,'seed':42,'batch_size':16,'precision':'bf16','parameter_count':model.count_parameters(),'nfe':1}
    write(mf,result);return result

def smoke():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    models=setup()
    labels=torch.randint(151,(16,),device='cuda');x=torch.randn((16,3,64,64),device='cuda')
    with torch.no_grad():imgs=image(models[0],x,labels)
    fit_fake(models[0],models[4],models[6],imgs,labels)
    rows=[update(models,1,reg) for reg in (0.1,0.)]
    calls=[0];m=models[2];hook=m.backbone.register_forward_hook(lambda *a:calls.__setitem__(0,calls[0]+1))
    with torch.inference_mode():out=m.sample((2,3,64,64),device='cuda',category=torch.tensor([0,13],device='cuda'),seed=42)
    hook.remove();assert calls[0]==1 and torch.isfinite(out).all()
    write(STUDY/'preflight.json',{'updates':rows,'inference_backbone_calls':calls[0],'teacher_heads':4,'generator_heads':1,'finite_output':True})
    print(json.dumps(rows),flush=True)


def main():
    if '--smoke' in sys.argv:smoke();return
    STUDY.mkdir(parents=True,exist_ok=True)
    lock=(STUDY/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (STUDY/'preflight.json').exists():raise RuntimeError('GPU preflight required')
    plan={'method':'DMD-inspired VP adaptation from supplied handoff; synthetic-only training',
          'real_score':{'checkpoint':str(REAL),'sha256':digest(REAL),'heads':4,'ema':0.9999,'step':150000,'cfg':2},
          'generator_initialization':{'checkpoint':str(INITIAL),'sha256':digest(INITIAL),'heads':1},
          'endpoint':{'checkpoint':str(ENDPOINT),'sha256':digest(ENDPOINT),'note':'Legacy4-NFE endpoint, not a child of new4-head teacher'},
          'arms':ARMS,'generator_updates':2500,'fake_warmup_updates':200,'batch_size':64,'microbatch_size':16,
          'G_lr':2e-5,'fake_lr':2e-4,'G_ema':0.999,'seed':42,'noise_t_range':[50,950],
          'inference_exports':[500,1000,2500],'optimizer_resume':'not implemented; original archived trainer also has no resume CLI',
          'selection':'development reference755; official reference3020 reporting only',
          'invalid_archive_export':'Truncated21MB checkpoint; not used. Originalone_nfe follows handoff retraining guide',
          'lineage':'Newteacher holdout exclusion verified; full legacyG/endpoint ancestry not independently verified'}
    if (STUDY/'plan.json').exists() and json.loads((STUDY/'plan.json').read_text())!=plan:raise ValueError('Plan mismatch')
    write(STUDY/'plan.json',plan)
    import wandb
    urls={};all_results={}
    for arm,reg in ARMS.items():
        directory=STUDY/arm;directory.mkdir(exist_ok=True)
        if (directory/'completed.json').exists():
            all_results[arm]=json.loads((directory/'completed.json').read_text());continue
        if (directory/'train.log').exists():raise RuntimeError('Incomplete previous run preserved; no exact optimizer resume available')
        ip=directory/'wandb_identity.json';identity={'id':uuid.uuid4().hex[:8]};write(ip,identity)
        run=wandb.init(entity='fv2z6n929c-kaist-digital-humanities-and-social-sciences-',project='pokemon-diffusion',
          id=identity['id'],name=f'dmd_4head150k_{arm}',group='dmd_selected4head150k_endpoint_ablation',
          dir=str(directory),mode='online',config={**plan,'arm':arm,'regularization_weight':reg},save_code=False,
          settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
        identity['url']=run.url;write(ip,identity);urls[arm]=run.url
        run.define_metric('step');run.define_metric('train/*',step_metric='step');run.define_metric('eval/*',step_metric='step')
        def status(phase,**extra):
            write(STUDY/'status.json',{'phase':phase,'active_arm':arm,'runs':urls,**extra});run.summary['phase']=phase
        try:
            models=setup();student,endpoint,ema,real,fake,opt,fopt=models
            baseline=score(ema,STUDY/'baseline')
            # Evaluation/cache paths must not alter paired training RNG.
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            run.log({'step':0,'eval/development_fid':baseline['fid']})
            rows=[{'step':0,**baseline}]
            status('fake_warmup',generator_step=0)
            for n in range(200):
                labels=torch.randint(151,(16,),device='cuda');x=torch.randn((16,3,64,64),device='cuda')
                with torch.no_grad():imgs=image(student,x,labels)
                val=fit_fake(student,fake,fopt,imgs,labels)
                if (n+1)%20==0:run.log({'warmup/update':n+1,'warmup/fake_score_loss':val})
            started=time.monotonic()
            for step in range(1,2501):
                row=update(models,step,reg);row['elapsed_seconds']=time.monotonic()-started
                with (directory/'train.log').open('a') as f:f.write(json.dumps(row)+'\n')
                if step%20==0 or step==1:
                    print(arm,json.dumps(row),flush=True);run.log({'step':step,**{'train/'+k:v for k,v in row.items() if k!='step'}})
                    status('training',generator_step=step)
                if step%100==0:
                    labels=torch.tensor([0,13,50,75,103,133,150,25],device='cuda')
                    with torch.inference_mode():images=ema.sample((8,3,64,64),device='cuda',category=labels,seed=42)
                    path=directory/f'qual_{step:06d}.png';save_image_grid(images,path)
                    run.log({'step':step,'qual/fixed_seed42':wandb.Image(str(path))})
                if step in (500,1000,2500):
                    ckpt=directory/f'dmd_step_{step:06d}.ckpt'
                    torch.save({'state_dict':ema.state_dict(),'step':step,'config':{**plan,'arm':arm,'regularization_weight':reg},'nfe':1,'parameter_count':48883107},ckpt)
                    status('evaluating',generator_step=step)
                    metrics=score(ema,directory/f'eval_{step:06d}');rows.append({'step':step,**metrics})
                    run.log({'step':step,'eval/development_fid':metrics['fid']})
                    write(directory/'results.json',rows)
            chosen=min(rows,key=lambda x:x['fid'])
            selected=INITIAL if chosen['step']==0 else directory/f'dmd_step_{chosen["step"]:06d}.ckpt'
            status('official_reporting',selected_step=chosen['step'])
            source=STUDY/'baseline' if chosen['step']==0 else directory/f'eval_{chosen["step"]:06d}'
            official=score(ema,source,official=True)
            run.log({'step':2500,'eval/selected_official_fid':official['fid']})
            result={'development_results':rows,'selected_by_development':chosen,'selected_checkpoint':str(selected),'official_report':official,'wandb_url':run.url}
            write(directory/'completed.json',result);all_results[arm]=result;write(STUDY/'results.json',all_results)
            run.summary['selected_step']=chosen['step'];run.summary['phase']='completed';run.finish()
            del models,student,endpoint,ema,real,fake,opt,fopt;torch.cuda.empty_cache()
        except BaseException as e:
            status('failed',error=str(e)[:500]);run.finish(exit_code=1);raise
    write(STUDY/'status.json',{'phase':'completed','runs':urls})

if __name__=='__main__':main()
