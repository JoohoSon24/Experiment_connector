"""Add previously omitted guided DDIM grids without replacing existing grids."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teacher.config import Config
from teacher.diffusion import Diffusion
from teacher.utils import load_checkpoint, make_model, save_image_grid, write_json


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    run=Path(args.run_dir)
    state=load_checkpoint(run/'final.pt','cpu')
    cfg=Config(**state['config'])
    step=state['step']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=make_model(cfg,len(state['class_names'])).to(device).eval()
    model.load_state_dict(state['ema'])
    del state
    directory=run/'diagnostics'/f'step_{step:07d}'
    metadata_path=directory/'metadata.json'
    metadata=json.loads(metadata_path.read_text())
    labels=torch.tensor(metadata['labels'],device=device)
    diffusion=Diffusion(cfg.diffusion_steps,cfg.beta_start,cfg.beta_end,device)
    with torch.inference_mode():
        for scale in (2,3):
            filename=f'ema_{cfg.ema_decay:g}_ddim_clip1_s{scale}.png'
            if (directory/filename).exists():
                if not any(record['file']==filename for record in metadata['variants']):
                    raise ValueError('Existing grid without matching metadata')
                continue
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' and cfg.amp_dtype=='bf16' else contextlib.nullcontext()
            with amp:
                images,nfe=diffusion.sample(model,labels,cfg.image_size,3,cfg.sample_steps,scale,cfg.seed,
                                          sampler='ddim',clip_denoised=True)
            assert images.shape==(len(labels),3,64,64) and torch.isfinite(images).all()
            assert nfe==2*cfg.sample_steps
            save_image_grid(images,directory/filename)
            metadata['variants'].append({'file':filename,'weights':f'ema_{cfg.ema_decay:g}',
                                        'sampler':'ddim','clip_denoised':True,'guidance_scale':scale,
                                        'nfe':nfe,'supplemental_post_training':True})
            write_json(metadata_path,metadata)
            print(f'PASS shape/finite/NFE={nfe}: {directory/filename}',flush=True)


if __name__=='__main__':
    main()
