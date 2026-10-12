"""Wait for the authorized study, then report fixed official FID without tuning."""
import json
from pathlib import Path
import sys
import time
import uuid
import fcntl
import run_teacher_budget_continuation as h
ROOT=h.ROOT
STUDY=ROOT/'experiments/teacher_4head150k_followup'

def main():
    import os
    os.chdir(ROOT)
    lock=(STUDY/'official_report.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        status=json.loads((STUDY/'status.json').read_text()) if (STUDY/'status.json').exists() else {}
        if status.get('phase')=='failed':raise RuntimeError('Main study failed; official reporting not started')
        if status.get('phase')=='completed':break
        time.sleep(20)
    import wandb
    outroot=STUDY/'official_report';outroot.mkdir(exist_ok=True)
    ip=outroot/'wandb_identity.json'
    identity=json.loads(ip.read_text()) if ip.exists() else {'id':uuid.uuid4().hex[:8]}
    h.write_json(ip,identity)
    run=wandb.init(entity='fv2z6n929c-kaist-digital-humanities-and-social-sciences-',project='pokemon-diffusion',
        name='teacher_4head150k_followup_official_FID_report',id=identity['id'],resume='allow',
        group='teacher_4head150k_followup',job_type='frozen-official-report',mode='online',dir=str(outroot),
        config={'note':'Official report only. No training/selection/gate uses these scores','sampler':'DDIM32 CFG2 seed-specific batch16 bf16'},
        settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
    identity['url']=run.url;h.write_json(ip,identity)
    cases=[]
    for m in json.loads((STUDY/'evaluation_results.json').read_text()):
        if m['steps']!=32:continue
        label='ema09999' if m['weights']=='ema' else 'ema0999'
        cases.append((f"{m['model']}_{label}_seed{m['seed']}",STUDY/'evaluation'/m['model']/label/f"seed{m['seed']}_ddim32"))
    for arm,stages in json.loads((STUDY/'lr_comparison/results.json').read_text()).items():
        for step,m in stages.items():
            if int(step)==150000:continue
            cases.append((f'{arm}_{step}',STUDY/'lr_comparison'/arm/f'evaluation/step_{step}/ddim32_cfg2'))
    reports={}
    try:
        for name,source in cases:
            old=json.loads((source/'metrics.json').read_text());out=outroot/name;out.mkdir(exist_ok=True)
            if not (out/'generated').exists():(out/'generated').symlink_to(source/'generated',target_is_directory=True)
            mf=out/'metrics.json'
            if not mf.exists():h.write_json(mf,dict(old,reference_split='official',reference_count=3020,fid=None))
            if json.loads(mf.read_text()).get('fid') is None:
                h.command([sys.executable,'-u','-m','teacher_variants.evaluate','--checkpoint',old['checkpoint'],
                    '--output-dir',str(out),'--reference-split','official','--reference-dir','results/reference',
                    '--weights',old['weights'],'--steps','32','--guidance-scale','2','--seed',str(old['seed']),
                    '--batch-size','16','--precision','bf16','--sampler','ddim','--clip-denoised','--score-only'],out/'console.log')
            m=json.loads(mf.read_text());reports[name]=m;h.write_json(outroot/'results.json',reports)
            run.log({f'official_fid/{name}':m['fid']})
            table=wandb.Table(columns=['condition','Official_FID'],data=[[k,v['fid']] for k,v in reports.items()])
            run.log({'comparison/official_FID':wandb.plot.bar(table,'condition','Official_FID',title='Official DDIM32 CFG2 FID (report only)')})
        h.write_json(outroot/'status.json',{'phase':'completed','url':run.url});run.finish()
    except BaseException as error:
        h.write_json(outroot/'status.json',{'phase':'failed','error':str(error)[:500]});run.finish(exit_code=1);raise

if __name__=='__main__':main()
