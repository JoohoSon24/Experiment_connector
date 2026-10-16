"""Finish common fresh bootstrap, then four paired DMD ablations without idle gaps."""
import sys,os,json,time,copy,subprocess,fcntl,uuid
from pathlib import Path
import torch
import run_dmd_selected_teacher as d
from teacher_variants.config import Config
from teacher_variants.utils import make_model
from distillation.core import coefficients
ROOT=d.ROOT;BOOT=ROOT/'experiments/fresh_student_4head150k';OUT=ROOT/'experiments/fresh_student_dmd_queue'
ARMS={'A_reg01_lr2e5':(.1,2e-5),'B_reg00_lr2e5':(0.,2e-5),'C_reg03_lr2e5':(.3,2e-5),'D_reg01_lr1e5':(.1,1e-5)}

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan={'teacher':str(d.REAL),'teacher_sha256':d.digest(d.REAL),'student_initialization':'common fresh bootstrap EMA, no legacy models',
        'queue':ARMS,'steps_per_arm':2500,'evaluate_at':[250,500,1000,2500],'fake_lr':2e-4,'fake_warmup':200,
        'G_EMA':.999,'batch':64,'microbatch':16,'seed':42,'endpoint':'same selected teacher synthetic pairs',
        'retention':'Only best inference checkpoint per arm; all metrics/PNG preserved; no full optimizer resume'}
    d.write(OUT/'plan.json',plan)
    def status(phase,**kw):d.write(OUT/'status.json',{'phase':phase,**kw})
    status('waiting_for_teacher_pair_cache',queued=list(ARMS))
    # Preserve completed generation rather than discarding the running teacher-pair work.
    cache=BOOT/'synthetic_teacher_pairs.pt'
    while True:
        if cache.exists():
            try:
                pool=torch.load(cache,map_location='cpu',weights_only=True)
                if pool['noise'].shape==(4096,3,64,64):break
            except (RuntimeError,EOFError):pass
        time.sleep(2)
    initial=BOOT/'bootstrap_005000.ckpt'
    if not initial.exists():
        subprocess.run(['tmux','send-keys','-t','fresh_student_only_selected_teacher','C-c'],check=False)
        for _ in range(60):
            if subprocess.run(['tmux','has-session','-t','fresh_student_only_selected_teacher'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0:break
            time.sleep(1)
        else:raise RuntimeError('Original process did not stop; avoid overlapping GPU training')
        log=BOOT/'train.log'
        if log.exists():log.rename(BOOT/f'interrupted_bootstrap_{int(time.time())}.log')
        status('bootstrap_training',queued=list(ARMS))
        subprocess.run([sys.executable,'-u','scripts/run_fresh_student_bootstrap_only.py'],cwd=ROOT,check=True)
    import wandb
    results={};urls={}
    for arm,(reg,lr) in ARMS.items():
        folder=OUT/arm;folder.mkdir(exist_ok=True)
        if (folder/'completed.json').exists():results[arm]=json.loads((folder/'completed.json').read_text());continue
        if (folder/'train.log').exists():raise RuntimeError('Interrupted arm preserved, no exact resume implemented')
        run=wandb.init(entity='fv2z6n929c-kaist-digital-humanities-and-social-sciences-',project='pokemon-diffusion',
            name=f'fresh_teacher_only_DMD_{arm}',group='fresh_student_DMD_four_arms',mode='online',dir=str(folder),
            config={**plan,'arm':arm,'regularization_weight':reg,'generator_lr':lr},save_code=False,
            settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
        urls[arm]=run.url;d.write(folder/'wandb_identity.json',{'id':run.id,'url':run.url})
        run.define_metric('step');run.define_metric('train/*',step_metric='step');run.define_metric('eval/*',step_metric='step')
        try:
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            student=d.load(d.ModelOneNFE,initial).train().requires_grad_(True)
            ema=copy.deepcopy(student).eval().requires_grad_(False)
            state=torch.load(d.REAL,map_location='cpu',weights_only=True);cfg=Config(**state['config'])
            assert cfg.num_attention_heads==4 and state['step']==150000
            real=make_model(cfg,151).cuda().eval().requires_grad_(False);real.load_state_dict(state['ema']);del state
            fake=copy.deepcopy(real).train().requires_grad_(True)
            opt=torch.optim.AdamW(student.backbone.parameters(),lr=lr,weight_decay=0)
            fopt=torch.optim.AdamW(fake.parameters(),lr=2e-4,weight_decay=0)
            baseline=d.score(ema,BOOT/'eval_bootstrap_005000');best={'step':0,**baseline};bestpath=initial
            rows=[best];torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.reset_peak_memory_stats()
            def pairs(step,micro):
                i=((step-1)*4+micro)%256;s=slice(16*i,16*(i+1))
                return tuple(pool[k][s].cuda() for k in ('noise','targets','labels'))
            status('fake_warmup',active_arm=arm,queued=list(ARMS),runs=urls)
            for w in range(200):
                x,target,labels=pairs(w+1,0)
                with torch.no_grad():im=d.image(student,x,labels)
                d.fit_fake(student,fake,fopt,im,labels)
            started=time.monotonic()
            for step in range(1,2501):
                opt.zero_grad(set_to_none=True);ls=0.;dl=rg=0.
                for micro in range(4):
                    x,target,labels=pairs(step,micro)
                    with torch.no_grad():detached=d.image(student,x,labels)
                    fl=d.fit_fake(student,fake,fopt,detached,labels)
                    t=torch.randint(50,951,(16,),device='cuda');a,s=coefficients(student.alpha_bar,t)
                    noisy=a*detached+s*torch.randn_like(x)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        re=d.apply_cfg(real(noisy,t,torch.full_like(labels,151)),real(noisy,t,labels),2).float()
                        fe=fake(noisy,t,labels).float();rx=(noisy-s*re)/a;fx=(noisy-s*fe)/a
                        gradient=(fx-rx)/(detached-rx).abs().mean((1,2,3),keepdim=True).clamp_min(.001)
                    images=d.image(student,x,labels)
                    dmd=.5*(images-(images.detach()-gradient).detach()).square().mean()
                    ep=(images-target).square().mean();loss=(dmd+reg*ep)/4
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                    loss.backward();ls+=float(loss.detach());dl+=float(dmd.detach())/4;rg+=float(ep.detach())/4
                gn=torch.nn.utils.clip_grad_norm_(student.parameters(),1,error_if_nonfinite=True);opt.step()
                with torch.no_grad():
                    for q,p in zip(ema.parameters(),student.parameters(),strict=True):q.lerp_(p.detach(),.001)
                peak=torch.cuda.max_memory_allocated()
                if peak>=20_000_000_000:raise RuntimeError('20GB ceiling')
                row={'step':step,'loss':ls,'dmd_loss':dl,'endpoint_mse':rg,'fake_score_loss':fl,'grad_norm':float(gn),'peak_vram_bytes':peak,'elapsed_seconds':time.monotonic()-started}
                with (folder/'train.log').open('a') as f:f.write(json.dumps(row)+'\n')
                if step==1 or step%20==0:
                    run.log({'step':step,**{'train/'+k:v for k,v in row.items() if k!='step'}});print(arm,json.dumps(row),flush=True)
                    status('DMD_training',active_arm=arm,step=step,runs=urls)
                if step%100==0:
                    with torch.inference_mode():im=ema.sample((8,3,64,64),device='cuda',category=torch.arange(8,device='cuda'),seed=42)
                    p=folder/f'qual_{step:06d}.png';d.save_image_grid(im,p);run.log({'step':step,'qual/fixed_seed42':wandb.Image(str(p))})
                if step in (250,500,1000,2500):
                    m=d.score(ema,folder/f'eval_{step:06d}');rows.append({'step':step,**m});run.log({'step':step,'eval/development_fid':m['fid']})
                    if m['fid']<best['fid']:
                        best={'step':step,**m};temp=folder/'best.tmp'
                        torch.save({'state_dict':ema.state_dict(),'step':step,'config':{**plan,'arm':arm,'regularization_weight':reg,'generator_lr':lr},'nfe':1},temp)
                        bestpath=folder/'best_one_nfe.ckpt';temp.replace(bestpath)
                    d.write(folder/'results.json',rows)
            source=BOOT/'eval_bootstrap_005000' if best['step']==0 else folder/f'eval_{best["step"]:06d}'
            official=d.score(ema,source,official=True);run.log({'eval/selected_official_fid':official['fid']})
            result={'selected_by_development':best,'selected_checkpoint':str(bestpath),'official':official,'milestones':rows,'url':run.url}
            d.write(folder/'completed.json',result);results[arm]=result;d.write(OUT/'results.json',results)
            run.finish();del student,ema,real,fake,opt,fopt;torch.cuda.empty_cache()
        except BaseException as e:status('failed',active_arm=arm,error=str(e)[:500]);run.finish(exit_code=1);raise
    status('completed',runs=urls)

if __name__=='__main__':main()
