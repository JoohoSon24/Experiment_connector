"""Fresh noaug teacher, explicitly authorized when original full checkpoint is absent."""
import fcntl
import json
import os
import sys
import uuid
from pathlib import Path
import run_teacher_budget_continuation as helper


def main():
    os.chdir(helper.ROOT)
    helper.RUN_ROOT = helper.ROOT / 'experiments/teacher_budget_noaug_fresh_200k'
    root = helper.RUN_ROOT
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _, base = helper.load_config(helper.BASE)
    configs = {}
    for target in (100000, 150000, 200000):
        raw = dict(base, num_train_steps=target,
                   experiment_name=f'teacher_b64_noaug_fresh_{target//1000}k',
                   output_dir=str(root.relative_to(helper.ROOT)))
        helper.check_recipe(base, raw)
        path = helper.ROOT / f'configs/teacher/teacher_b64_noaug_fresh_{target//1000}k.json'
        if path.exists() and json.loads(path.read_text()) != raw:
            raise ValueError(f'Existing config differs: {path}')
        helper.write_json(path, raw)
        configs[target] = path
    plan = {'initialization': 'fresh random initialization, seed42; user authorized',
            'original_100k_checkpoint_available': False, 'milestone_steps': [100000,150000,200000],
            'recipe': base, 'teacher_source_sha256': helper.assert_original_teacher_source(),
            'evaluation': {'steps':[32,50], 'guidance_scale':2,'seed':42,'batch_size':16,
                           'reference_count':755,'sample_count':3020,'weights':'primary EMA',
                           'sampler':'clipped DDIM','split':'development'},
            'wandb': {'entity':'fv2z6n929c-kaist-digital-humanities-and-social-sciences-',
                      'project':'pokemon-diffusion','mode':'online'}}
    plan_path = root / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing fresh-run plan differs')
    helper.write_json(plan_path, plan)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    import wandb
    identity_path = root / 'wandb_identity.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {'id':uuid.uuid4().hex[:8]}
    helper.write_json(identity_path, identity)
    run = wandb.init(**plan['wandb'],id=identity['id'],resume='allow',
                     name='teacher_noaug_fresh_0_to_100k_150k_200k_seed42',dir=str(root),config=plan,
                     save_code=False,tags=['teacher','noaug','fresh','seed42'],
                     settings=wandb.Settings(init_timeout=60,x_disable_meta=True))
    identity['url'] = run.url
    helper.write_json(identity_path, identity)
    run.define_metric('global_step')
    run.define_metric('train/*',step_metric='global_step')
    run.define_metric('eval/*',step_metric='global_step')
    observer = helper.Observer(run,wandb)
    observer.global_step = 0
    run.summary['initialization'] = 'fresh'
    milestones = {}
    previous = None
    try:
        for target in (100000,150000,200000):
            _, cfg = helper.load_config(configs[target])
            stage = helper.ROOT / cfg['output_dir'] / cfg['experiment_name']
            final = stage / 'final.pt'
            if not final.exists():
                latest = stage / 'latest.pt'
                resume = latest if latest.exists() else previous
                args = [sys.executable,'-u','-m','teacher.train','--config',str(configs[target])]
                current = 0
                if resume:
                    current, _ = helper.checkpoint_step(resume,(0 if target==100000 else target-50000,target))
                    args += ['--resume',str(resume)]
                observer.global_step = current
                observer.status('training',initialization='fresh',training_started=True,target_step=target,
                                resumed_from=str(resume) if resume else None)
                helper.assert_original_teacher_source()
                if current == target:
                    helper.shutil.copy2(resume,final)
                else:
                    helper.command(args,stage/'console.log',observer,stage)
            helper.checkpoint_step(final,(target,target))
            milestones[str(target)] = helper.evaluate(final,target,observer)
            previous = final
        selected = min(milestones,key=lambda key:milestones[key]['ddim32_cfg2']['fid'])
        helper.write_json(root/'summary.json',{'status':'completed','initialization':'fresh',
            'milestones':milestones,'selected_step':int(selected),
            'selection_policy':'Lowest DDIM32 CFG2 development FID; DDIM50 secondary'})
        run.summary['selected_step'] = int(selected)
        observer.status('completed',selected_step=int(selected))
        run.finish()
    except BaseException as error:
        observer.status('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error)[:500])
        run.finish(exit_code=1)
        raise

if __name__ == '__main__':
    main()
