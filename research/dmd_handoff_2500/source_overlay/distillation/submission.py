"""Inference helpers for the unchanged FastGen base Model interface."""
from __future__ import annotations
import contextlib
import torch
from teacher.model import ClassConditionalUNet
from teacher.diffusion import Diffusion
from .core import sample_velocity


def submission_backbone():
    return ClassConditionalUNet(image_size=64,in_channels=3,num_classes=151,model_channels=96,
                                channel_mult=(1,2,3,4),num_res_blocks=2,attention_resolutions=(16,),dropout=0)


def initialize_buffers(model,device,steps):
    if steps not in (1,4):raise ValueError('Submission supports exactly 1 or 4 NFE')
    schedule=Diffusion(1000,0.0001,0.02,torch.device(device))
    grid=torch.linspace(999,0,32,device=device).round().long()[::(32//steps)]
    model.register_buffer('alpha_bar',schedule.alpha_bar)
    model.register_buffer('sampling_times',grid)
    model.register_buffer('sampling_seed',torch.tensor(42,device=device))
    model._batch_index=0
    model.nfe_per_sample=steps
    model.to(device)


@torch.inference_mode()
def submission_sample(model,shape,device,category,**kwargs):
    device=torch.device(device)
    if tuple(shape[1:])!=(3,64,64) or category is None or len(category)!=shape[0]:
        raise ValueError('Expected categories and shape [B,3,64,64]')
    category=category.to(device=device,dtype=torch.long)
    if torch.any((category<0)|(category>=151)):raise ValueError('Invalid class')
    model.to(device).eval()
    seed=int(kwargs.get('seed',int(model.sampling_seed)+model._batch_index))
    if 'seed' not in kwargs:model._batch_index+=1
    amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
    with amp:
        if hasattr(model, 'consistency_sigma_data'):
            from consistency.core import sample
            return sample(model.backbone,model.alpha_bar,model.sampling_times,category,seed,
                          float(model.consistency_sigma_data))
        return sample_velocity(model.backbone,model.alpha_bar,model.sampling_times,category,seed,False)
