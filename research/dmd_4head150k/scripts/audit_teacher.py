"""Training-only denoising and independent sampler checks for a frozen teacher.

The clipped DDIM diagnostic follows OpenAI improved-diffusion's pred_xstart
clipping and epsilon reconstruction. It does not alter production sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teacher.cfg import apply_cfg
from teacher.config import Config
from teacher.data import TrainImages
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint, make_model, save_image_grid, set_seed, write_json


@torch.inference_mode()
def reference_sample(model, diffusion, labels, size, steps, scale, seed, clip_x0, ancestral=False):
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    x = torch.randn((len(labels), 3, size, size), generator=generator, device=labels.device)
    times = torch.linspace(diffusion.steps - 1, 0, steps, device=labels.device).round().long().tolist()
    trace = []
    for index, t_value in enumerate(times):
        t = torch.full_like(labels, t_value)
        if scale == 1:
            eps = model(x, t, labels).float()
        elif scale == 0:
            eps = model(x, t, torch.full_like(labels, model.num_classes)).float()
        else:
            eps = apply_cfg(model(x, t, torch.full_like(labels, model.num_classes)).float(),
                            model(x, t, labels).float(), scale)
        alpha = diffusion.alpha_bar[t_value]
        prev = diffusion.alpha_bar[times[index + 1]] if index + 1 < len(times) else torch.ones_like(alpha)
        x0 = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
        if index in (0, len(times) // 2, len(times) - 1):
            trace.append({'t': t_value, 'x0_outside_fraction': float((x0.abs() > 1).float().mean()),
                          'x0_abs_max': float(x0.abs().max())})
        if clip_x0:
            x0 = x0.clamp(-1, 1)
        if ancestral:
            # Fixed-small DDPM posterior for the effective (possibly respaced) betas.
            beta = 1 - alpha / prev
            mean = beta * prev.sqrt() / (1 - alpha) * x0 + (1 - prev) * (1 - beta).sqrt() / (1 - alpha) * x
            noise = torch.randn(x.shape, generator=generator, device=x.device)
            x = mean + ((1 - prev) / (1 - alpha) * beta).clamp_min(0).sqrt() * noise
        else:
            if clip_x0:
                eps = (x - alpha.sqrt() * x0) / (1 - alpha).sqrt()
            x = prev.sqrt() * x0 + (1 - prev).sqrt() * eps
    return x.clamp(-1, 1), trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--sampling-only', action='store_true', help='Extend report with paired DDIM/DDPM and raw/EMA checks')
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(args.checkpoint, 'cpu')
    raw = state['config'].copy()
    raw['channel_mult'] = tuple(raw['channel_mult'])
    raw['attention_resolutions'] = tuple(raw['attention_resolutions'])
    cfg = Config(**raw)
    set_seed(42)
    device = torch.device('cuda')
    model = make_model(cfg, 151).to(device).eval()
    model.load_state_dict(state['ema'])
    diffusion = Diffusion(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, device)
    if args.sampling_only:
        report=json.loads((out/'results.json').read_text())
        chosen=torch.tensor([0,13,103,133],device=device).repeat_interleave(4)
        with torch.inference_mode():
            for weights,ancestral,clip,steps in [('ema',False,False,1000),('ema',False,True,1000),
                                                 ('ema',True,False,1000),('ema',True,True,1000),
                                                 ('model',True,True,1000),('model',False,True,50),
                                                 ('ema',True,True,50),('ema',True,True,250)]:
                model.load_state_dict(state[weights])
                # Exit autocast before changing weights: its cast cache can retain old tensors.
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    sample,trace=reference_sample(model,diffusion,chosen,64,steps,1,42,clip,ancestral)
                name=f'{weights}_{"ddpm" if ancestral else "ddim"}{steps}_clip{int(clip)}_s1_paired'
                save_image_grid(sample,out/f'{name}.png')
                report['sampler']=[row for row in report['sampler'] if row['name']!=name]
                report['sampler'].append({'name':name,'weights':weights,'nfe':steps,'trace':trace})
                write_json(out/'results.json',report)
                print(name,flush=True)
        return
    dataset = TrainImages(cfg.data_root, cfg.split_dir, cfg.image_size, cfg.num_classes, split_file=cfg.train_manifest)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:64].tolist()
    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = torch.stack([dataset[i][1] for i in indices]).to(device)
    save_image_grid(images[:16], out/'training_examples.png')
    result = {'checkpoint': args.checkpoint, 'step': state['step'], 'seed': 42,
              'initial_EMA_weight_retention': cfg.ema_decay ** state['step'],
              'denoising_dataset': '64 seen training images, no augmentation; not a generalization test',
              'denoising': [], 'sampler': []}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        noise = torch.randn_like(images)
        for value in (0, 10, 50, 100, 250, 500, 750, 999):
            alpha = diffusion.alpha_bar[value]
            noisy = alpha.sqrt()*images + (1-alpha).sqrt()*noise
            t = torch.full_like(labels, value)
            row = {'t': value, 'alpha_bar': float(alpha)}
            for name, condition in [('conditional', labels), ('wrong_class', (labels+1)%151), ('null', torch.full_like(labels,151))]:
                prediction = model(noisy, t, condition).float()
                row[name+'_epsilon_mse'] = float(F.mse_loss(prediction,noise))
                x0 = (noisy - (1-alpha).sqrt()*prediction)/alpha.sqrt()
                row[name+'_x0_mse'] = float(F.mse_loss(x0,images))
                if name == 'conditional' and value in (100,250,500):
                    save_image_grid(x0[:16], out/f'reconstruction_t{value}.png')
            model.load_state_dict(state['model'])
            online=model(noisy,t,labels).float()
            row['online_epsilon_mse']=float(F.mse_loss(online,noise))
            row['online_x0_mse']=row['online_epsilon_mse']*float((1-alpha)/alpha)
            model.load_state_dict(state['ema'])
            result['denoising'].append(row)
        chosen = torch.tensor([0,13,103,133],device=device).repeat_interleave(4)
        result['sample_classes'] = [state['class_names'][i] for i in (0,13,103,133)]
        for clip in (False,True):
            for scale in (0,0.5,1,2,3,4,6):
                sample, trace = reference_sample(model,diffusion,chosen,64,50,scale,42,clip)
                name=f'ddim50_clip{int(clip)}_s{scale:g}'
                save_image_grid(sample,out/f'{name}.png')
                result['sampler'].append({'name':name,'nfe':50 if scale in (0,1) else 100,'trace':trace})
            print('CFG sweep clip=',clip,flush=True)
        sample,trace = reference_sample(model,diffusion,chosen[:4],64,1000,1,42,True,ancestral=True)
        save_image_grid(sample,out/'ddpm1000_clipped_s1.png')
        result['sampler'].append({'name':'ddpm1000_clipped_s1','nfe':1000,'trace':trace})
    # Oracle with nonconstant target detects timestep/end-point algebra errors.
    target=images[:2]
    class Oracle:
        num_classes=151
        def __call__(self,x,t,c):
            alpha=diffusion.alpha_bar[t,None,None,None]
            return (x-alpha.sqrt()*target)/(1-alpha).sqrt()
    actual,nfe=diffusion.sample(Oracle(),labels[:2],64,3,50,1,42)
    result['oracle_max_abs_error']=float((actual-target).abs().max())
    assert result['oracle_max_abs_error']<1e-5 and nfe==50
    reference,_=reference_sample(Oracle(),diffusion,labels[:2],64,50,1,42,False)
    assert torch.allclose(actual,reference,atol=1e-5)
    result['oracle_reference_match']=True
    write_json(out/'results.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
