"""EMA and sampling repeats, followed by the authorized paired LR comparison."""
import fcntl
import json
from pathlib import Path
import os
import subprocess
import sys
import uuid
import run_teacher_budget_continuation as h

ROOT=h.ROOT
STUDY=ROOT/'experiments/teacher_4head150k_followup'
MODELS={'attention4_150k':ROOT/'experiments/teacher_variant_sequence/attention4/attention4_150k/final.pt',
        'singlehead_200k':ROOT/'experiments/teacher_budget_noaug_fresh_200k/teacher_b64_noaug_fresh_200k/final.pt'}
SEEDS=[42,1042,2042]


def main():
    os.chdir(ROOT);STUDY.mkdir(parents=True,exist_ok=True)
    lock=(STUDY/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan={'experiments':['EMA comparison','sampling-seed repeats','paired LR fine-tuning'],
          'models':{name:{'path':str(p),'sha256':h.digest(p)} for name,p in MODELS.items()},
          'weights':['ema','ema:0.999'],'sampling_seeds':SEEDS,
          'evaluation':'development755 reference; generated3020; clipped DDIM32 CFG2 seed-specific; batch16 bf16',
          'ddim50_secondary':'seed42 only, both EMA weights',
          'lr_study':'same4-head150K full state; constant2e-4 vs1e-4; +5K and+10K; no other recipe change',
          'selection':'development only; no official evaluation or tuning in this study'}
    pp=STUDY/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise ValueError('Plan differs')
    h.write_json(pp,plan)
    import wandb
    ip=STUDY/'wandb_identity.json'
    identity=json.loads(ip.read_text()) if ip.exists() else {'id':uuid.uuid4().hex[:8]}
    h.write_json(ip,identity)
    run=wandb.init(entity='fv2z6n929c-kaist-digital-humanities-and-social-sciences-',project='pokemon-diffusion',
                  name='teacher_4head150k_EMA_sampling_repeats',id=identity['id'],resume='allow',
                  group='teacher_4head150k_followup',job_type='EMA-seed-evaluation',mode='online',
                  dir=str(STUDY),config=plan,save_code=False,
                  settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
    identity['url']=run.url;h.write_json(ip,identity)
    def status(phase,**kw):
        h.write_json(STUDY/'status.json',{'phase':phase,'evaluation_run':run.url,**kw})
        run.summary['phase']=phase
    records=[]
    eval_finished=False
    try:
        for name,ckpt in MODELS.items():
            for weight in ('ema','ema:0.999'):
                for seed,steps in [(s,32) for s in SEEDS]+[(42,50)]:
                    label='ema09999' if weight=='ema' else 'ema0999'
                    out=STUDY/'evaluation'/name/label/f'seed{seed}_ddim{steps}'
                    out.mkdir(parents=True,exist_ok=True)
                    source=(ROOT/'experiments/teacher_variant_sequence/attention4/evaluation/step_150000' if name=='attention4_150k' else ROOT/'experiments/teacher_budget_noaug_fresh_200k/evaluation/step_200000')/f'ddim{steps}_cfg2'
                    mf=out/'metrics.json'
                    if weight=='ema' and seed==42 and not mf.exists():
                        existing=json.loads((source/'metrics.json').read_text())
                        if existing['checkpoint']!=str(ckpt) or existing['weights']!='ema':raise ValueError('Baseline provenance differs')
                        (out/'generated').symlink_to(source/'generated',target_is_directory=True)
                        h.write_json(mf,existing)
                        h.write_json(out/'reuse.json',{'source':str(source),'note':'Previously measured exact same configuration; not new measurement'})
                    if not mf.exists() or json.loads(mf.read_text()).get('fid') is None:
                        status('evaluating',model=name,weights=weight,seed=seed,steps=steps)
                        args=[sys.executable,'-u','-m','teacher_variants.evaluate','--checkpoint',str(ckpt),
                              '--output-dir',str(out),'--reference-split','development','--reference-dir','data/development/seed42_holdout5/reference',
                              '--weights',weight,'--steps',str(steps),'--guidance-scale','2','--seed',str(seed),
                              '--batch-size','16','--precision','bf16','--sampler','ddim','--clip-denoised']
                        if mf.exists() and len(list((out/'generated').glob('*.png')))==3020:args.append('--score-only')
                        h.command(args,out/'console.log')
                    m=json.loads(mf.read_text())
                    expected={'checkpoint':str(ckpt),'weights':weight,'seed':seed,'steps':steps,'guidance_scale':2.0,
                              'reference_split':'development','reference_count':755,'sample_count':3020,'batch_size':16,
                              'precision':'bf16','sampler':'ddim','clip_denoised':True}
                    if any(m.get(k)!=v for k,v in expected.items()) or m.get('fid') is None:raise ValueError('Evaluation metadata mismatch')
                    row={'model':name,'weights':weight,'seed':seed,'steps':steps,'fid':m['fid']}
                    records.append(row);h.write_json(STUDY/'evaluation_results.json',records)
                    run.log({f'fid/{name}/{label}/ddim{steps}':m['fid'],'sampling_seed':seed,
                             f'qual/{name}/{label}/ddim{steps}':[wandb.Image(str(out/'generated'/f'{c:04d}_00.png'),caption=f'class{c} seed{seed}') for c in (0,13,50,103,150)]})
                    table=wandb.Table(columns=['model','weights','seed','steps','FID'],data=[[x[k] for k in ('model','weights','seed','steps','fid')] for x in records])
                    run.log({'comparison/EMA_seed_FID':table})
        import statistics
        summaries=[]
        for name in MODELS:
            for weight in ('ema','ema:0.999'):
                vals=[r['fid'] for r in records if r['model']==name and r['weights']==weight and r['steps']==32]
                summaries.append({'model':name,'weights':weight,'sampling_seeds':SEEDS,'mean_fid':statistics.mean(vals),'sample_std_fid':statistics.stdev(vals),
                                  'note':'Sampling-seed variability; not training repeats or reference uncertainty'})
        h.write_json(STUDY/'evaluation_summary.json',summaries)
        table=wandb.Table(columns=['candidate','mean_FID'],data=[[x['model']+'/'+x['weights'],x['mean_fid']] for x in summaries])
        run.log({'comparison/mean_ddim32_fid':wandb.plot.bar(table,'candidate','mean_FID',title='DDIM32 CFG2 mean across3 sampling seeds')})
        status('evaluation_completed_starting_lr_comparison');run.finish();eval_finished=True
        subprocess.run([sys.executable,'-u','scripts/run_teacher_4head150k_lr_comparison.py'],cwd=ROOT,check=True)
        h.write_json(STUDY/'status.json',{'phase':'completed','evaluation_run':identity['url'],
            'lr_status':json.loads((STUDY/'lr_comparison/status.json').read_text())})
    except BaseException as error:
        h.write_json(STUDY/'status.json',{'phase':'failed','evaluation_run':identity['url'],'error':str(error)[:500]})
        if not eval_finished:run.finish(exit_code=1)
        raise

if __name__=='__main__':main()
