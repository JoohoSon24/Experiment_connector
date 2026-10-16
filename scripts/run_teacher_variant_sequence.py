"""Sequential isolated teacher architecture/regularization experiments."""
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid
import torch
import run_teacher_budget_continuation as h
from teacher_variants.config import Config, load_config
from teacher_variants.utils import make_model

ROOT=h.ROOT
STUDY=ROOT/'experiments/teacher_variant_sequence'
ORIGINAL_BASE=h.BASE
FULL200=ROOT/'experiments/teacher_budget_noaug_fresh_200k/teacher_b64_noaug_fresh_200k/final.pt'
ARMS={'attention4':{'num_attention_heads':4},'dropout005':{'dropout':0.05},'attention32':{'attention_resolutions':[16,32]}}
COMMAND=h.command

def variant_command(args,*pos,**kwargs):
    args=list(args)
    args=[a.replace('teacher.train','teacher_variants.train').replace('teacher.evaluate','teacher_variants.evaluate') for a in args]
    return COMMAND(args,*pos,**kwargs)
h.command=variant_command

def config(arm,target):
    base=json.loads(ORIGINAL_BASE.read_text())
    result=dict(base,num_attention_heads=1,**{})
    result.update(ARMS[arm])
    result.update(num_train_steps=target,experiment_name=f'{arm}_{target//1000}k',output_dir=str((STUDY/arm).relative_to(ROOT)))
    for key in base:
        if key not in {'num_train_steps','experiment_name','output_dir',*ARMS[arm]} and result[key]!=base[key]:
            raise ValueError(f'Unexpected recipe change: {key}')
    path=ROOT/f'configs/teacher/variant_{arm}_{target//1000}k.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Config changed')
    h.write_json(path,result)
    return path,result


def validate(path,cfg,min_step,max_step,initial_dropout=False):
    s=torch.load(path,map_location='cpu',weights_only=False)
    required={'model','ema','ema_tracks','optimizer','scaler','config','class_names','train_paths_sha256','torch_rng','python_rng','cuda_rng','step','epoch','batch_in_epoch'}
    if not required<=s.keys():raise ValueError('Not a full checkpoint')
    old=s['config'];current=dict(cfg)
    normalized=lambda d:json.loads(json.dumps(__import__('dataclasses').asdict(Config(**d))))
    oldn,newn=normalized(old),normalized(current)
    for key in oldn:
        if key in {'num_train_steps','experiment_name','output_dir'}:continue
        if initial_dropout and key=='dropout' and oldn[key]==0 and newn[key]==0.05:continue
        if oldn[key]!=newn[key]:raise ValueError(f'Checkpoint semantics mismatch {key}')
    if not min_step<=s['step']<=max_step:raise ValueError('Unexpected step')
    baseline=json.loads(ORIGINAL_BASE.read_text())
    mapping=json.loads((ROOT/baseline['split_dir']/'category_to_id.json').read_text())
    names=[n for n,_ in sorted(mapping.items(),key=lambda p:p[1])]
    manifest=[l.strip() for l in (ROOT/baseline['train_manifest']).read_text().splitlines() if l.strip()]
    import hashlib
    if s['class_names']!=names or s['train_paths_sha256']!=hashlib.sha256('\n'.join(manifest).encode()).hexdigest():raise ValueError('Data mapping/split mismatch')
    with torch.device('meta'):expected=make_model(Config(**current),151).state_dict()
    for w in [s['model'],s['ema'],*s['ema_tracks'].values()]:
        if w.keys()!=expected.keys() or any(w[k].shape!=expected[k].shape for k in expected):raise ValueError('Weight shape mismatch')
    groups=s['optimizer']['param_groups']
    if len(groups)!=1 or not s['optimizer']['state'] or groups[0]['lr']!=0.0002 or groups[0]['weight_decay']!=0:raise ValueError('AdamW state mismatch')
    if set(s['ema_tracks'])!={'0.999'} or not s['cuda_rng']:raise ValueError('EMA/RNG missing')
    return s['step']


def main():
    os.chdir(ROOT);STUDY.mkdir(parents=True,exist_ok=True)
    lock=(STUDY/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    h.assert_original_teacher_source()
    _,baseline_sha=h.checkpoint_step(FULL200,(200000,200000))
    source={str(p.relative_to(ROOT)):h.digest(p) for p in sorted((ROOT/'teacher_variants').glob('*.py'))}
    preflight=json.loads((STUDY/'preflight/attention32_batch64_preflight/metrics.json').read_text())
    if preflight['completed_steps']!=1 or preflight['peak_vram_bytes']>=20_000_000_000:raise RuntimeError('Attention32 batch64 memory preflight failed')
    plan={'order':list(ARMS),'changes':ARMS,'baseline_200k_sha256':baseline_sha,
          'architecture_initialization':'fresh seed42 for attention4 and attention32',
          'dropout_initialization':'same baseline200K full state; dropout0->0.05; LR2e-4; extra10K',
          'architecture_budget':'100K first; extend through150K to200K if DDIM32 holdout FID beats matched100K baseline by >=0.30',
          'fixed_recipe':'noaug batch64 constantLR2e-4 warmup0 noise1000 bf16 seed42 EMA0.9999+0.999',
          'evaluation':'clipped DDIM32/50 CFG2; seed42 batch16; generated3020; holdout755; no official tuning',
          'variant_source_sha256':source,'attention32_preflight_peak_bytes':preflight['peak_vram_bytes']}
    pp=STUDY/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise ValueError('Existing plan differs')
    h.write_json(pp,plan)
    import wandb
    baseline100=json.loads((FULL200.parent.parent/'milestone_100000.json').read_text())
    baseline200=json.loads((FULL200.parent.parent/'milestone_200000.json').read_text())
    control210=json.loads((ROOT/'experiments/teacher_lr_comparison_200k/results.json').read_text())['control_lr2e4']['210000']
    urls={};reports={}
    def status(phase,**extra):h.write_json(STUDY/'status.json',{'phase':phase,'runs':urls,**extra})
    try:
        for arm in ARMS:
            h.RUN_ROOT=STUDY/arm;h.RUN_ROOT.mkdir(exist_ok=True)
            ip=h.RUN_ROOT/'wandb_identity.json'
            identity=json.loads(ip.read_text()) if ip.exists() else {'id':uuid.uuid4().hex[:8]}
            h.write_json(ip,identity)
            run=wandb.init(entity='fv2z6n929c-kaist-digital-humanities-and-social-sciences-',project='pokemon-diffusion',
                name=f'teacher_variant_{arm}_seed42',id=identity['id'],resume='allow',group='teacher_variants_single_change',
                job_type='teacher-architecture' if arm!='dropout005' else 'teacher-regularization',
                mode='online',dir=str(h.RUN_ROOT),config={**plan,'active_arm':arm},save_code=False,
                settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
            urls[arm]=run.url;identity['url']=run.url;h.write_json(ip,identity)
            run.define_metric('global_step');run.define_metric('train/*',step_metric='global_step')
            run.define_metric('eval/*',step_metric='global_step',summary='min')
            run.define_metric('qual/*',step_metric='global_step')
            obs=h.Observer(run,wandb)
            obs.global_step=200000 if arm=='dropout005' else 0
            targets=[210000] if arm=='dropout005' else [100000]
            previous=FULL200 if arm=='dropout005' else None
            reports[arm]={}
            run.summary['phase']='training'
            run.summary['baseline_ddim32_fid']=control210['ddim32_cfg2']['fid'] if arm=='dropout005' else baseline100['ddim32_cfg2']['fid']
            i=0
            while i<len(targets):
                target=targets[i]
                cfgpath,cfg=config(arm,target)
                stage=ROOT/cfg['output_dir']/cfg['experiment_name']
                final=stage/'final.pt';latest=stage/'latest.pt'
                resume=latest if latest.exists() else previous
                start=validate(resume,cfg,200000 if arm=='dropout005' else 0,target,initial_dropout=resume==FULL200) if resume else 0
                if not final.exists():
                    obs.global_step=start;status('training',active_arm=arm,target_step=target)
                    obs.status('training',target_step=target,initialization='baseline200K dropout fork' if arm=='dropout005' else 'fresh seed42')
                    args=[sys.executable,'-u','-m','teacher_variants.train','--config',str(cfgpath)]
                    if resume:
                        args+=['--resume',str(resume)]
                        if resume==FULL200:args+=['--allow-dropout-change']
                    h.assert_original_teacher_source()
                    if start==target:h.shutil.copy2(resume,final)
                    else:h.command(args,stage/'console.log',obs,stage)
                validate(final,cfg,target,target)
                status('evaluating',active_arm=arm,target_step=target)
                metrics=h.evaluate(final,target,obs)
                reports[arm][str(target)]=metrics
                generated=h.RUN_ROOT/f'evaluation/step_{target:06d}/ddim32_cfg2/generated'
                run.log({'global_step':target,'qual/ddim32_cfg2':[wandb.Image(str(generated/f'{c:04d}_00.png'),caption=f'class{c}') for c in (0,13,50,75,103,133,150)]})
                h.write_json(STUDY/'results.json',reports)
                comparison=control210 if arm=='dropout005' else json.loads((FULL200.parent.parent/f'milestone_{target:06d}.json').read_text())
                for n in (32,50):
                    run.log({f'comparison/ddim{n}_fid':wandb.plot.bar(wandb.Table(columns=['model','FID'],data=[['baseline',comparison[f'ddim{n}_cfg2']['fid']],[arm,metrics[f'ddim{n}_cfg2']['fid']]]),'model','FID',title=f'Matched budget DDIM{n} CFG2')})
                if arm!='dropout005' and target==100000:
                    gain=baseline100['ddim32_cfg2']['fid']-metrics['ddim32_cfg2']['fid']
                    extend=gain>=0.30
                    h.write_json(h.RUN_ROOT/'extension_decision.json',{'gain':gain,'threshold':0.30,'extend_to200k':extend,'note':'Operational threshold, not statistical significance'})
                    if extend:targets.extend([150000,200000])
                previous=final;i+=1
            obs.status('completed',final_step=targets[-1]);run.finish()
        status('completed')
    except BaseException as error:
        status('failed',error=str(error)[:500])
        if 'run' in locals():run.finish(exit_code=1)
        raise

if __name__=='__main__':main()
