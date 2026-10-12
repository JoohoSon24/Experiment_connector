"""Short isolated training throughput/VRAM probe; never saves trained weights."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teacher.config import Config
from teacher.diffusion import Diffusion
from teacher.train import drop_classes, update_ema
from teacher.utils import load_checkpoint, make_model, set_seed, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    checkpoint=load_checkpoint(args.checkpoint,'cpu')
    raw=checkpoint['config'].copy()
    raw['channel_mult']=tuple(raw['channel_mult'])
    raw['attention_resolutions']=tuple(raw['attention_resolutions'])
    cfg=Config(**raw)
    set_seed(42)
    device=torch.device('cuda')
    model=make_model(cfg,151).to(device).train()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    diffusion=Diffusion(cfg.diffusion_steps,cfg.beta_start,cfg.beta_end,device)
    results={'gpu':torch.cuda.get_device_name(),'parameter_count':sum(p.numel() for p in model.parameters()),
             'notes':'Synthetic resident images; includes AdamW, EMA, clipping, bf16. Three warmup and three timed updates; no data I/O/checkpoint time. Weights not saved.',
             'cases':[]}
    previous=None
    for batch,accumulation in ((8,2),(16,1),(32,1),(64,1),(96,1),(128,1)):
        if previous and batch>previous['microbatch']:
            estimated=1.2*(previous['peak_allocated_GiB']*batch/previous['microbatch'])
            if estimated>17:
                results['stopped_before_batch']=batch
                results['estimated_peak_GiB']=estimated
                break
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0)
        images=torch.rand((batch,3,64,64),device=device)*2-1
        labels=torch.arange(batch,device=device)%151
        def update():
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                conditions=drop_classes(labels,151,cfg.class_dropout)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=diffusion.loss(model,images,conditions)/accumulation
                loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite benchmark gradients')
            optimizer.step()
            update_ema(ema,model,cfg.ema_decay)
        for _ in range(3): update()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(3): update()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        row={'microbatch':batch,'accumulation':accumulation,'effective_batch':batch*accumulation,
             'seconds_per_update':elapsed/3,'images_per_second':batch*accumulation*3/elapsed,
             'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
             'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30}
        results['cases'].append(row)
        previous=row
        print(json.dumps(row),flush=True)
        del optimizer,images,labels
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,results)


if __name__=='__main__': main()
