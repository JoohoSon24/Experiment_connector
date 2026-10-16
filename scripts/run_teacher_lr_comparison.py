"""Paired noaug teacher LR study; original trainer is unchanged."""
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid
import run_teacher_budget_continuation as h

ROOT=h.ROOT
STUDY=ROOT/'experiments/teacher_lr_comparison_200k'
BASELINE=ROOT/'experiments/teacher_budget_noaug_fresh_200k/teacher_b64_noaug_fresh_200k/final.pt'
ORIGINAL_BASE=h.BASE
ARMS={'control_lr2e4':0.0002,'finetune_lr5e5':0.00005}


def make_config(arm,target):
    _,base=h.load_config(ORIGINAL_BASE)
    cfg=dict(base,learning_rate=ARMS[arm],num_train_steps=target,
             experiment_name=f'{arm}_{target//1000}k',output_dir=str((STUDY/arm).relative_to(ROOT)))
    # Explicitly allow only LR and output/budget bookkeeping changes.
    comparable=dict(cfg,learning_rate=base['learning_rate'])
    h.check_recipe(base,comparable)
    path=ROOT/f'configs/teacher/lr_compare_{arm}_{target//1000}k.json'
    if path.exists() and h.effective_config(json.loads(path.read_text()))!=h.effective_config(cfg):raise ValueError('Config mismatch')
    h.write_json(path,cfg)
    return path,cfg


def main():
    os.chdir(ROOT)
    STUDY.mkdir(parents=True,exist_ok=True)
    lock=(STUDY/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    h.assert_original_teacher_source()
    _,sha=h.checkpoint_step(BASELINE,(200000,200000))
    baseline=json.loads((BASELINE.parent.parent/'milestone_200000.json').read_text())
    plan={'baseline_checkpoint':str(BASELINE),'baseline_sha256':sha,'arms':ARMS,
          'paired_initial_state':'Same raw model, AdamW moments, EMA, scaler, RNG, epoch and cursor',
          'scheduled_additional_updates':[5000,10000],
          'extension_rule':'If best 210K DDIM32 development FID improves baseline by >=0.30, extend BOTH arms to 225K',
          'extension_evaluation_additional_updates':[20000,25000],
          'evaluation':'primary EMA; clipped DDIM32/50 CFG2; seed42; batch16; bf16; ref755; generated3020',
          'qual':'Original CFG1 DDPM50 preview every1000 updates; matched DDIM32 CFG2 preview at FID milestones',
          'execution':'Sequential on one GPU; comparison pairs use identical checkpoint starts',
          'selection':'development only; no official validation tuning'}
    pp=STUDY/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=json.loads(json.dumps(plan)):raise ValueError('Plan differs')
    h.write_json(pp,plan)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    import wandb
    runs={};observers={};results={};previous={arm:BASELINE for arm in ARMS}
    for arm in ARMS:
        directory=STUDY/arm;directory.mkdir(exist_ok=True)
        ip=directory/'wandb_identity.json'
        identity=json.loads(ip.read_text()) if ip.exists() else {'id':uuid.uuid4().hex[:8]}
        h.write_json(ip,identity)
        run=wandb.init(entity='fv2z6n929c-kaist-digital-humanities-and-social-sciences-',project='pokemon-diffusion',
                       id=identity['id'],resume='allow',name=f'teacher_200k_{arm}',group='teacher_200k_lr_comparison',
                       job_type='paired-lr-finetune',dir=str(directory),config={**plan,'arm':arm,'learning_rate':ARMS[arm]},
                       mode='online',reinit='create_new',save_code=False,
                       settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
        identity['url']=run.url;h.write_json(ip,identity)
        run.define_metric('global_step')
        run.define_metric('train/*',step_metric='global_step')
        run.define_metric('eval/*',step_metric='global_step',summary='min')
        run.define_metric('qual/*',step_metric='global_step')
        run.log({'global_step':200000,'eval/ddim32_cfg2_fid':baseline['ddim32_cfg2']['fid'],
                 'eval/ddim50_cfg2_fid':baseline['ddim50_cfg2']['fid']})
        obs=h.Observer(run,wandb);obs.global_step=200000
        runs[arm]=run;observers[arm]=obs
        results[arm]={'200000':baseline}
    def persist(phase,**extra):
        h.write_json(STUDY/'status.json',{'phase':phase,'runs':{a:r.url for a,r in runs.items()},**extra})
    def charts():
        for steps in (32,50):
            xs=[];ys=[]
            for arm in ARMS:
                keys=sorted(results[arm],key=int)
                xs.append([int(k)-200000 for k in keys]);ys.append([results[arm][k][f'ddim{steps}_cfg2']['fid'] for k in keys])
            for run in runs.values():
                run.log({f'comparison/ddim{steps}_cfg2_fid':wandb.plot.line_series(xs=xs,ys=ys,keys=list(ARMS),
                         title=f'DDIM{steps} CFG2 holdout FID by additional updates',xname='additional_updates')})
    try:
        targets=[205000,210000]
        index=0
        while index<len(targets):
            target=targets[index]
            for arm in ARMS:
                h.RUN_ROOT=STUDY/arm
                obs=observers[arm]
                cfgpath,cfg=make_config(arm,target)
                stage=ROOT/cfg['output_dir']/cfg['experiment_name']
                final=stage/'final.pt'
                latest=stage/'latest.pt'
                resume=latest if latest.exists() else previous[arm]
                h.BASE=ORIGINAL_BASE if resume==BASELINE else cfgpath
                step,_=h.checkpoint_step(resume,(200000,target))
                if not final.exists():
                    obs.global_step=step
                    persist('training',active_arm=arm,target_step=target)
                    obs.status('training',target_step=target,resumed_from=str(resume),learning_rate=ARMS[arm])
                    h.assert_original_teacher_source()
                    if step==target:h.shutil.copy2(resume,final)
                    else:h.command([sys.executable,'-u','-m','teacher.train','--config',str(cfgpath),'--resume',str(resume)],stage/'console.log',obs,stage)
                h.BASE=cfgpath
                h.checkpoint_step(final,(target,target))
                persist('evaluating',active_arm=arm,target_step=target)
                results[arm][str(target)]=h.evaluate(final,target,obs)
                # Representative class IDs spread across class list; same seed/images across arms.
                generated=h.RUN_ROOT/f'evaluation/step_{target:06d}/ddim32_cfg2/generated'
                images=[wandb.Image(str(generated/f'{label:04d}_00.png'),caption=f'class {label}, DDIM32 CFG2') for label in (0,13,50,75,103,133,150)]
                runs[arm].log({'global_step':target,'qual/ddim32_cfg2':images})
                previous[arm]=final
                obs.status('waiting_for_paired_arm',milestone=target)
                h.write_json(STUDY/'results.json',results)
                charts()
            if target==210000:
                best=min(results[a]['210000']['ddim32_cfg2']['fid'] for a in ARMS)
                gain=baseline['ddim32_cfg2']['fid']-best
                extend=gain>=0.30
                h.write_json(STUDY/'extension_decision.json',{'baseline_fid':baseline['ddim32_cfg2']['fid'],
                    'best_210k_fid':best,'gain':gain,'threshold':0.30,'extend_both_arms':extend,
                    'note':'Operational threshold, not a statistical significance claim'})
                if extend:targets.extend([220000,225000])
            index+=1
        for arm,run in runs.items():
            run.summary['phase']='completed'
            run.summary['final_step']=targets[-1]
            run.finish()
        persist('completed',final_step=targets[-1])
    except BaseException as error:
        persist('failed',error=str(error)[:500])
        for run in runs.values():run.finish(exit_code=1)
        raise

if __name__=='__main__':main()
