"""Paired noaug teacher LR study; original trainer is unchanged."""
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid
import run_teacher_budget_continuation as h
from teacher_variants.config import Config
from teacher_variants.utils import make_model
import torch
import dataclasses
COMMAND=h.command
def variant_command(args,*pos,**kw):
    return COMMAND([a.replace('teacher.train','teacher_variants.train').replace('teacher.evaluate','teacher_variants.evaluate') for a in args],*pos,**kw)
h.command=variant_command

ROOT=h.ROOT
STUDY=ROOT/'experiments/teacher_4head150k_followup/lr_comparison'
BASELINE=ROOT/'experiments/teacher_variant_sequence/attention4/attention4_150k/final.pt'
ORIGINAL_BASE=ROOT/'configs/teacher/variant_attention4_150k.json'
ARMS={'control_lr2e4':0.0002,'finetune_lr1e4':0.0001}



def validate(path,expected_range):
    state=torch.load(path,map_location='cpu',weights_only=False)
    required={'model','ema','ema_tracks','optimizer','scaler','config','step','epoch','batch_in_epoch','class_names','train_paths_sha256','torch_rng','python_rng','cuda_rng'}
    if not required<=state.keys():raise ValueError('Incomplete full checkpoint')
    base=json.loads(ORIGINAL_BASE.read_text());old=state['config']
    norm=lambda d:json.loads(json.dumps(dataclasses.asdict(Config(**d))))
    left,right=norm(old),norm(base)
    for k in left:
        if k not in {*h.BOOKKEEPING,'learning_rate'} and left[k]!=right[k]:raise ValueError(f'Recipe changed: {k}')
    if left['learning_rate'] not in (0.0002,0.0001):raise ValueError('Unexpected LR')
    if not expected_range[0]<=state['step']<=expected_range[1]:raise ValueError('Step mismatch')
    g=state['optimizer']['param_groups']
    if len(g)!=1 or g[0]['lr']!=left['learning_rate'] or g[0]['weight_decay']!=0 or not state['optimizer']['state']:raise ValueError('Optimizer mismatch')
    if set(state['ema_tracks'])!={'0.999'} or not state['cuda_rng']:raise ValueError('EMA/RNG absent')
    import hashlib
    paths=[x.strip() for x in (ROOT/base['train_manifest']).read_text().splitlines() if x.strip()]
    mapping=json.loads((ROOT/base['split_dir']/'category_to_id.json').read_text())
    names=[n for n,_ in sorted(mapping.items(),key=lambda p:p[1])]
    if state['class_names']!=names or state['train_paths_sha256']!=hashlib.sha256('\n'.join(paths).encode()).hexdigest():raise ValueError('Mapping/split mismatch')
    with torch.device('meta'):expected=make_model(Config(**base),151).state_dict()
    for weights in [state['model'],state['ema'],*state['ema_tracks'].values()]:
        if weights.keys()!=expected.keys() or any(weights[k].shape!=expected[k].shape for k in expected):raise ValueError('Model mismatch')
    step=state['step'];del state,expected
    return step,h.digest(path)

def make_config(arm,target):
    base=json.loads(ORIGINAL_BASE.read_text())
    cfg=dict(base,learning_rate=ARMS[arm],num_train_steps=target,
             experiment_name=f'{arm}_{target//1000}k',output_dir=str((STUDY/arm).relative_to(ROOT)))
    # Explicitly allow only LR and output/budget bookkeeping changes.
    comparable=dict(cfg,learning_rate=base['learning_rate'])
    assert {k:v for k,v in comparable.items() if k not in h.BOOKKEEPING} == {k:v for k,v in base.items() if k not in h.BOOKKEEPING}
    path=ROOT/f'configs/teacher/lr_compare_4head150k_{arm}_{target//1000}k.json'
    if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('Config mismatch')
    h.write_json(path,cfg)
    return path,cfg


def main():
    os.chdir(ROOT)
    STUDY.mkdir(parents=True,exist_ok=True)
    lock=(STUDY/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    h.assert_original_teacher_source()
    _,sha=validate(BASELINE,(150000,150000))
    baseline={f'ddim{n}_cfg2':json.loads((BASELINE.parent.parent/f'evaluation/step_150000/ddim{n}_cfg2/metrics.json').read_text()) for n in (32,50)}
    plan={'baseline_checkpoint':str(BASELINE),'baseline_sha256':sha,'arms':ARMS,
          'paired_initial_state':'Same raw model, AdamW moments, EMA, scaler, RNG, epoch and cursor',
          'scheduled_additional_updates':[5000,10000],
          'extension_rule':'Stop at additional10K; no automatic extension',
          'extension_evaluation_additional_updates':[],
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
                       id=identity['id'],resume='allow',name=f'teacher_4head150k_{arm}',group='teacher_4head150k_lr_comparison',
                       job_type='paired-lr-finetune',dir=str(directory),config={**plan,'arm':arm,'learning_rate':ARMS[arm]},
                       mode='online',reinit='create_new',save_code=False,
                       settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
        identity['url']=run.url;h.write_json(ip,identity)
        run.define_metric('global_step')
        run.define_metric('train/*',step_metric='global_step')
        run.define_metric('eval/*',step_metric='global_step',summary='min')
        run.define_metric('qual/*',step_metric='global_step')
        run.log({'global_step':150000,'eval/ddim32_cfg2_fid':baseline['ddim32_cfg2']['fid'],
                 'eval/ddim50_cfg2_fid':baseline['ddim50_cfg2']['fid']})
        obs=h.Observer(run,wandb);obs.global_step=150000
        runs[arm]=run;observers[arm]=obs
        results[arm]={'150000':baseline}
    def persist(phase,**extra):
        h.write_json(STUDY/'status.json',{'phase':phase,'runs':{a:r.url for a,r in runs.items()},**extra})
    def charts():
        for steps in (32,50):
            xs=[];ys=[]
            for arm in ARMS:
                keys=sorted(results[arm],key=int)
                xs.append([int(k)-150000 for k in keys]);ys.append([results[arm][k][f'ddim{steps}_cfg2']['fid'] for k in keys])
            for run in runs.values():
                run.log({f'comparison/ddim{steps}_cfg2_fid':wandb.plot.line_series(xs=xs,ys=ys,keys=list(ARMS),
                         title=f'DDIM{steps} CFG2 holdout FID by additional updates',xname='additional_updates')})
    try:
        targets=[155000,160000]
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
                step,_=validate(resume,(150000,target))
                if not final.exists():
                    obs.global_step=step
                    persist('training',active_arm=arm,target_step=target)
                    obs.status('training',target_step=target,resumed_from=str(resume),learning_rate=ARMS[arm])
                    h.assert_original_teacher_source()
                    if step==target:h.shutil.copy2(resume,final)
                    else:h.command([sys.executable,'-u','-m','teacher.train','--config',str(cfgpath),'--resume',str(resume)],stage/'console.log',obs,stage)
                h.BASE=cfgpath
                validate(final,(target,target))
                persist('evaluating',active_arm=arm,target_step=target)
                # Preserve final and alias completed latest to it, saving one full-state copy.
                latest=stage/'latest.pt'
                if latest.exists() and latest.stat().st_ino != final.stat().st_ino:
                    validate(latest,(target,target))
                    latest.unlink();os.link(final,latest)
                results[arm][str(target)]=h.evaluate(final,target,obs)
                # Representative class IDs spread across class list; same seed/images across arms.
                generated=h.RUN_ROOT/f'evaluation/step_{target:06d}/ddim32_cfg2/generated'
                images=[wandb.Image(str(generated/f'{label:04d}_00.png'),caption=f'class {label}, DDIM32 CFG2') for label in (0,13,50,75,103,133,150)]
                runs[arm].log({'global_step':target,'qual/ddim32_cfg2':images})
                previous[arm]=final
                obs.status('waiting_for_paired_arm',milestone=target)
                h.write_json(STUDY/'results.json',results)
                charts()
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
