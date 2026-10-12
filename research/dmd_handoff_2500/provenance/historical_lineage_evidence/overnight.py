"""Bounded development-only sequential fresh teacher / consistency experiments."""
from pathlib import Path
import os, sys, json, hashlib, subprocess, time, traceback, math
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
OUT=ROOT/'experiments/night_run';PY=sys.executable
SOURCE=Path('/root/FastGen/AIC313-Project-FastGen/data/pokemon-generation-one-22k')
DATA=OUT/'development_images';REF=OUT/'reference_development'
DEADLINE=time.time()+11*3600
records=[];best=None;teacher_metrics=[];status='preflight'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def command(module,args,log):
    report()
    with Path(log).open('a') as f:
        subprocess.run([PY,str(OUT/'safe_launch.py'),module,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,DEADLINE-time.time()),env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'})
def report(error=None):
    lines=['# Overnight development report',f'UTC update: {time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}',f'Status: {status}',
    '\nDevelopment split: 16,324 train / 755 holdout; all four requested relationships PASS. 151 classes, five holdout images per class. See AUDIT.md and audit_results.json.',
    '\nExisting inference exports excluded: their complete teacher/student lineage cannot be independently verified. All new learned models initialized through a fresh development-only teacher; no external pretrained generation weights.',
    '\nArchitecture: canonical class-conditional 64x64 U-Net, channels 96, multipliers [1,2,3,4], 2 residual blocks, attention 16; each model 48,883,107 parameters. Teacher fresh seed 42; epsilon linear VP1000, AdamW LR 2e-4 constant, batch64, EMA0.9999 with extra0.999; canonical teacher budget100K. Teacher is not a final one/few-NFE model.',
    '\nStudent: VP consistency L2, fresh-teacher hidden weights with zeroed output head, CFG2, AdamW LR2e-4 constant, batch64/microbatch16, target and sampling EMA0.999, scales128, sigma_data0.5. Baseline budget2.5K. Independent candidate runs have the same initialization seed and teacher.',
    '\nCandidates (one change each): lower LR1e-4 for stable consistency optimization; target EMA0.99 for less stale consistency targets; scales64 for larger adjacent noise transitions. Maximum three candidates; 2.5K screening, 5K only if below baseline FID, 10K only if 5K FID improves on its own2.5K score. No extension beyond10K.',
    '\nEvaluation: solely 755 holdout reference PNGs, seed42 plus generation batch index, 151 classes x20 =3020 generated images, batch16, bf16, clean-fid default clean mode. All student comparisons use identical reference hash. Exactly1 backbone call; no teacher at inference.',
    '\n| Run | Steps | Development FID | Parameters | NFE | Peak VRAM bytes | Training seconds |\n|---|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f'| {r["name"]} | {r["step"]} | {r["fid"]:.6f} | {r["parameter_count"]} | 1 | {r["peak_vram_bytes"]} | {r["elapsed_seconds"]:.2f} |')
    if not records:lines.append('| Pending | — | Not evaluated | 48,883,107 | 1 planned | Not measured | Not measured |')
    lines.append('\nDevelopment-best: '+(json.dumps(best,ensure_ascii=False) if best else 'Not yet selected; no inference export claimed.'))
    lines.append('\nSelection reason: lowest measured development FID under the fixed evaluation policy. No official evaluation was run or used.')
    lines.append('\nNext experiments (after this run): (1) replicate the winning change with another training seed; (2) evaluate endpoint-focused distillation against this clean consistency baseline.')
    if error:lines.append('\nStopped safely: '+str(error))
    (OUT/'NIGHT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'run_status.json',{'status':status,'pid':os.getpid(),'records':records,'best':best,'error':error,'updated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})

def verify_teacher(checkpoint):
    s=torch.load(checkpoint,map_location='cpu',weights_only=True)
    expected=hashlib.sha256('\n'.join(train).encode()).hexdigest()
    assert s['train_paths_sha256']==expected
    assert s['config']['train_manifest']==str(ROOT/'data/development/seed42_holdout5/train.txt')
    assert sum(v.numel() for v in s['model'].values())<100_000_000
    return digest(checkpoint)
def evaluate_student(name,run,step):
    checkpoint=run/f'step_{step:06d}.pt'
    s=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert s['teacher_sha256']==teacher_hash
    assert s['train_paths_sha256']==hashlib.sha256('\n'.join(train).encode()).hexdigest()
    assert s['settings']['num_train_steps']==10000
    elapsed=s['elapsed_seconds'];peak=s['peak_vram_bytes'];del s
    verify_reference()
    target=run/f'development_{step:06d}'
    command('consistency.evaluate',['--checkpoint',checkpoint,'--output-dir',target,'--reference-dir',REF,'--reference-split','development','--nfe',1,'--seed',42],OUT/f'{name}_eval_{step}.log')
    metric=read(target/'metrics.json');assert metric['reference_count']==755 and metric['sample_count']==3020
    assert metric['reference_split']=='development' and math.isfinite(metric['fid'])
    row={**metric,'name':name,'step':step,'elapsed_seconds':elapsed,'peak_vram_bytes':peak,'checkpoint':str(checkpoint)}
    records.append(row);write(OUT/'development_results.json',records);return row

def verify_reference():
    files=sorted(REF.glob('*.png'))
    assert len(files)==755 and {p.name for p in files}=={f'{i:04d}.png' for i in range(755)}
    assert all(digest(REF/name)==value for name,value in reference_hashes.items())

def train_student(name,settings,updates):
    command('consistency.train',['--config',OUT/f'{name}_settings.json','--teacher-checkpoint',teacher_checkpoint,'--output-dir',OUT/name,'--max-updates-this-run',updates],OUT/f'{name}_console.log')

try:
    audit=read(OUT/'audit_results.json');assert all(audit['checks'].values())
    assert all(digest(p)==v for p,v in audit['protected_sha256'].items())
    assert torch.cuda.is_available()
    train=(ROOT/'data/development/seed42_holdout5/train.txt').read_text().splitlines()
    holdout=(ROOT/'data/development/seed42_holdout5/holdout.txt').read_text().splitlines()
    assert not set(train)&set(holdout)
    DATA.mkdir(exist_ok=True)
    # Materialize only original-training-derived images; official images are absent.
    for path in train+holdout:
        dst=DATA/path;dst.parent.mkdir(parents=True,exist_ok=True);os.link(SOURCE/path,dst) if not dst.exists() else None
        assert dst.samefile(SOURCE/path)
    (DATA/'.transparency_composited_on_white').touch()
    REF.mkdir(exist_ok=True)
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    transform=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((.5,)*3,(.5,)*3)])
    reference_hashes={}
    for i,path in enumerate(holdout):
        with Image.open(DATA/path) as im:x=transform(im.convert('RGB'))
        dst=REF/f'{i:04d}.png'
        expected=to_pil_image(((x+1)/2).clamp(0,1)).convert('RGB')
        if dst.exists():
            with Image.open(dst) as actual: assert actual.convert('RGB').tobytes()==expected.tobytes()
        else: expected.save(dst)
        reference_hashes[dst.name]=digest(dst)
    write(OUT/'reference_manifest.json',{'source_manifest':str(ROOT/'data/development/seed42_holdout5/holdout.txt'),'source_manifest_sha256':digest(ROOT/'data/development/seed42_holdout5/holdout.txt'),'images':[{'reference':f'{i:04d}.png','source':p,'sha256':reference_hashes[f'{i:04d}.png']} for i,p in enumerate(holdout)]})
    status='development evaluator preflight';report()
    from evaluate import compute_fid
    fid=compute_fid(REF,REF,'cuda',16)
    assert math.isfinite(fid) and abs(fid)<.01
    write(OUT/'evaluation_preflight.json',{'holdout_self_fid':fid,'official_access':False,'reference_count':755,'metric_asset':'/tmp/inception-2015-12-05.pt','metric_asset_sha256':digest('/tmp/inception-2015-12-05.pt')})
    config=read(ROOT/'configs/teacher/teacher_b64_noaug_constant_100k.json')
    config.update(data_root=str(DATA),split_dir=str(ROOT/'data/pokemon-generation-one-22k'),train_manifest=str(ROOT/'data/development/seed42_holdout5/train.txt'),development_reference_manifest=str(ROOT/'data/development/seed42_holdout5/holdout.txt'),output_dir=str(OUT),experiment_name='teacher_preflight',num_train_steps=2,diagnostic_interval=0,sample_interval=0)
    write(OUT/'teacher_preflight_config.json',config)
    status='fresh teacher VRAM preflight';command('teacher.train',['--config',OUT/'teacher_preflight_config.json'],OUT/'teacher_preflight_console.log')
    pre=read(OUT/'teacher_preflight/metrics.json');assert pre['peak_vram_bytes']<20_000_000_000 and pre['parameter_count']<100_000_000
    config.update(experiment_name='fresh_teacher',num_train_steps=100000,save_interval=5000)
    write(OUT/'fresh_teacher_config.json',config)
    audit['status']='PASS for fresh development-only lineage; existing checkpoints excluded';write(OUT/'fresh_lineage_audit.json',audit)
    with (OUT/'AUDIT.md').open('a') as f:f.write('\nFresh-lineage gate PASS: train and holdout images materialized without any official images, holdout-only FID self-test passed, fresh teacher2-step peak VRAM '+str(pre['peak_vram_bytes'])+' bytes. Existing exported models remain ineligible. Proceeding with fresh lineage.\n')
    status='fresh development-only teacher training100K';command('teacher.train',['--config',OUT/'fresh_teacher_config.json','--resume',OUT/'teacher_preflight/final.pt'],OUT/'fresh_teacher_console.log')
    teacher_checkpoint=OUT/'fresh_teacher/final.pt';teacher_hash=verify_teacher(teacher_checkpoint)
    for scale in [1,2]:
        status=f'fresh teacher development validation CFG{scale}'
        command('teacher.evaluate',['--checkpoint',teacher_checkpoint,'--output-dir',OUT/f'teacher_development_cfg{scale}','--reference-dir',REF,'--reference-split','development','--steps',50,'--guidance-scale',scale,'--seed',42,'--batch-size',16,'--precision','bf16','--sampler','ddim','--clip-denoised'],OUT/f'teacher_eval_cfg{scale}.log')
        tm=read(OUT/f'teacher_development_cfg{scale}/metrics.json');assert math.isfinite(tm['fid']);teacher_metrics.append(tm)
    write(OUT/'teacher_development_results.json',teacher_metrics)
    settings=read(ROOT/'configs/consistency/fixed_cfg2.json');settings.update(num_train_steps=10000,milestone_steps=[2500,5000,10000],sample_interval=2500)
    variants=[('baseline',{}),('candidate_lr',{'learning_rate':.0001}),('candidate_target_ema',{'target_ema_decay':.99}),('candidate_scales',{'num_scales':64})]
    baseline=None
    for name,change in variants:
        status=f'{name} screening2500';settings_i={**settings,**change};write(OUT/f'{name}_settings.json',settings_i)
        write(OUT/f'{name}_rationale.json',{'change':change,'teacher':str(teacher_checkpoint),'teacher_sha256':teacher_hash,'fresh_student_initialization':True,'budget_cap':10000})
        train_student(name,settings_i,2)
        smoke=read(OUT/name/'metrics.json');assert smoke['peak_vram_bytes']<20_000_000_000
        train_student(name,settings_i,2498)
        row=evaluate_student(name,OUT/name,2500)
        if baseline is None:baseline=row
        elif row['fid']<baseline['fid']:
            status=f'{name} continuation5000';train_student(name,settings_i,2500);r5=evaluate_student(name,OUT/name,5000)
            if r5['fid']<row['fid']:
                status=f'{name} continuation10000';train_student(name,settings_i,5000);evaluate_student(name,OUT/name,10000)
        best=min(records,key=lambda r:r['fid']);report()
    best=min(records,key=lambda r:r['fid']);status='exporting development-best model';report()
    exported=OUT/'development_best_one_nfe.ckpt'
    command('consistency.export',['--checkpoint',best['checkpoint'],'--output',exported,'--mode','one_nfe'],OUT/'export.log')
    from model import ModelOneNFE
    s=torch.load(exported,map_location='cpu',weights_only=True);model=ModelOneNFE(device='cuda');model.load_state_dict(s['state_dict']);calls=[0]
    def hook(*args):calls[0]+=1
    h=model.backbone.register_forward_hook(hook)
    with torch.inference_mode():x=model.sample((1,3,64,64),device='cuda',category=torch.tensor([0],device='cuda'),seed=42)
    h.remove();assert calls[0]==1 and model.count_parameters()<100_000_000 and torch.isfinite(x).all()
    best={**best,'export':str(exported),'export_sha256':digest(exported),'measured_export_nfe':calls[0]};write(OUT/'development_best.json',best)
    assert all(digest(p)==v for p,v in audit['protected_sha256'].items())
    status='complete';report()
except BaseException as error:
    status='stopped';report(repr(error));(OUT/'failure_traceback.txt').write_text(traceback.format_exc());raise
