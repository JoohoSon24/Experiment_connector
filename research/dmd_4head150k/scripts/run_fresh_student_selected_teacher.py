"""Fresh student from only selected teacher: endpoint bootstrap then DMD."""
import sys,os,json,time,copy,uuid,fcntl
from pathlib import Path
import torch
import run_dmd_selected_teacher as d
from teacher_variants.config import Config
from teacher_variants.utils import make_model
from teacher.diffusion import Diffusion
from distillation.core import coefficients
ROOT=d.ROOT;OUT=ROOT/'experiments/fresh_student_4head150k'

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'train.log').exists():raise RuntimeError('Existing run preserved; no exact resume implemented')
    import wandb
    run=wandb.init(entity='fv2z6n929c-kaist-digital-humanities-and-social-sciences-',project='pokemon-diffusion',
        name='fresh_student_only_4head150k_teacher',group='fresh_student_selected_teacher',dir=str(OUT),mode='online',
        config={'teacher':str(d.REAL),'teacher_sha256':d.digest(d.REAL),'teacher_weights':'EMA0.9999',
                'legacy_student_or_endpoint_used':False,'bootstrap_updates':5000,'DMD_updates':2500,
                'teacher_target':'clipped DDIM32 CFG2','synthetic_pool_size':4096,'seed':42,
                'note':'Endpoint bootstrap + DMD; different initialization from archived DMD'},
        settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
    d.write(OUT/'wandb_identity.json',{'url':run.url,'id':run.id})
    def status(phase,**kw):d.write(OUT/'status.json',{'phase':phase,'url':run.url,**kw})
    try:
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        student=d.ModelOneNFE(device='cuda').train()  # random initialization, never load legacy weights
        state=torch.load(d.REAL,map_location='cpu',weights_only=True);cfg=Config(**state['config'])
        assert cfg.num_attention_heads==4 and state['step']==150000
        real=make_model(cfg,151).cuda().eval().requires_grad_(False);real.load_state_dict(state['ema']);del state
        diffusion=Diffusion(1000,cfg.beta_start,cfg.beta_end,torch.device('cuda'))
        cache=OUT/'synthetic_teacher_pairs.pt'
        if not cache.exists():
            status('generating_teacher_pairs');xs=[];ys=[];cs=[]
            for i in range(256):
                labels=(torch.arange(16,device='cuda')+16*i)%151;seed=1000000+i
                x=torch.randn((16,3,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    y,_=diffusion.sample(real,labels,64,3,32,2,seed,sampler='ddim',clip_denoised=True)
                xs.append(x.cpu());ys.append(y.float().cpu());cs.append(labels.cpu())
                if i%16==0:print('teacher pairs',16*(i+1),flush=True);status('generating_teacher_pairs',samples=16*(i+1))
            torch.save({'noise':torch.cat(xs),'targets':torch.cat(ys),'labels':torch.cat(cs)},cache)
        pool=torch.load(cache,weights_only=True,map_location='cpu')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        opt=torch.optim.AdamW(student.backbone.parameters(),lr=2e-4,weight_decay=0)
        ema=copy.deepcopy(student).eval().requires_grad_(False)
        fake=copy.deepcopy(real).train().requires_grad_(True)
        fopt=torch.optim.AdamW(fake.parameters(),lr=2e-4,weight_decay=0)
        def pairs(step,micro):
            i=((step-1)*4+micro)%256;s=slice(16*i,16*(i+1))
            return tuple(pool[k][s].cuda() for k in ('noise','targets','labels'))
        def qual(step):
            with torch.inference_mode():im=ema.sample((8,3,64,64),device='cuda',category=torch.arange(8,device='cuda'),seed=42)
            p=OUT/f'qual_{step:06d}.png';d.save_image_grid(im,p);run.log({'step':step,'qual/fixed_seed':wandb.Image(str(p))})
        torch.cuda.reset_peak_memory_stats()
        for stage,budget in (('bootstrap',5000),('DMD',2500)):
            if stage=='DMD':
                for g in opt.param_groups:g['lr']=2e-5
                for w in range(200):
                    x,target,labels=pairs(w+1,0)
                    with torch.no_grad():im=d.image(student,x,labels)
                    d.fit_fake(student,fake,fopt,im,labels)
            for step in range(1,budget+1):
                opt.zero_grad(set_to_none=True);loss_sum=0.
                for micro in range(4):
                    x,target,labels=pairs(step,micro)
                    if stage=='bootstrap':
                        images=d.image(student,x,labels);loss=(images-target).square().mean()/4
                    else:
                        with torch.no_grad():detached=d.image(student,x,labels)
                        d.fit_fake(student,fake,fopt,detached,labels)
                        t=torch.randint(50,951,(16,),device='cuda');a,s=coefficients(student.alpha_bar,t)
                        noisy=a*detached+s*torch.randn_like(x)
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                            re=d.apply_cfg(real(noisy,t,torch.full_like(labels,151)),real(noisy,t,labels),2).float()
                            fe=fake(noisy,t,labels).float();rx=(noisy-s*re)/a;fx=(noisy-s*fe)/a
                            gradient=(fx-rx)/(detached-rx).abs().mean((1,2,3),keepdim=True).clamp_min(.001)
                        images=d.image(student,x,labels)
                        loss=(.5*(images-(images.detach()-gradient).detach()).square().mean()+.1*(images-target).square().mean())/4
                    if not torch.isfinite(loss):raise FloatingPointError('loss nonfinite')
                    loss.backward();loss_sum+=float(loss.detach())
                grad=torch.nn.utils.clip_grad_norm_(student.parameters(),1,error_if_nonfinite=True);opt.step()
                with torch.no_grad():
                    for q,p in zip(ema.parameters(),student.parameters(),strict=True):q.lerp_(p.detach(),.001)
                peak=torch.cuda.max_memory_allocated()
                if peak>=20_000_000_000:raise RuntimeError('Memory ceiling exceeded')
                row={'stage':stage,'step':step,'loss':loss_sum,'grad_norm':float(grad),'peak_vram_bytes':peak}
                with (OUT/'train.log').open('a') as f:f.write(json.dumps(row)+'\n')
                globalstep=step if stage=='bootstrap' else 5000+step
                if step%20==0 or step==1:
                    print(json.dumps(row),flush=True);run.log({'step':globalstep,'train/loss':loss_sum,'train/peak_vram_bytes':peak});status(stage,update=step,budget=budget)
                if step%100==0:qual(globalstep)
                if (stage=='bootstrap' and step==budget) or (stage=='DMD' and step in (500,1000,2500)):
                    ckpt=OUT/f'{stage}_{step:06d}.ckpt';torch.save({'state_dict':ema.state_dict(),'step':globalstep,'nfe':1,'teacher':str(d.REAL)},ckpt)
                    metrics=d.score(ema,OUT/f'eval_{stage}_{step:06d}')
                    run.log({'step':globalstep,'eval/development_fid':metrics['fid']})
        candidates=[]
        for p in OUT.glob('eval_*/metrics.json'):
            m=json.loads(p.read_text());candidates.append((m['fid'],p.parent))
        _,selected=min(candidates,key=lambda x:x[0])
        official=d.score(ema,selected,official=True);d.write(OUT/'selected.json',{'evaluation_dir':str(selected),'official':official})
        run.log({'eval/selected_official_fid':official['fid']});status('completed');run.finish()
    except BaseException as e:status('failed',error=str(e)[:500]);run.finish(exit_code=1);raise

if __name__=='__main__':main()
