from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import torch
from distillation.train import train
from teacher.utils import load_checkpoint
from distillation.export import export
from model import ModelOneNFE,ModelFewNFE
T='/root/Diffusion/workspace_distillation/experiments/teacher_variant_sequence/attention4/attention4_150k/final.pt'
assert torch.cuda.is_available()
folder=ROOT/'experiments/preflight'
train('configs/base.json',T,'guidance',str(folder/'guidance'),steps_override=1)
state=load_checkpoint(folder/'guidance/final.pt','cpu');assert state['config']['num_attention_heads']==4 and state['step']==1 and state['prediction_type']=='v'
train('configs/base.json',str(folder/'guidance/final.pt'),'n16',str(folder/'n16'),steps_override=1)
state=load_checkpoint(folder/'n16/final.pt','cpu');assert len(state['sampling_times'])==16 and torch.isfinite(torch.tensor(state['last_loss']))
# Export/load/sample agreement with the four-head architecture for both submission wrappers.
from teacher.config import Config
from teacher.utils import make_model,save_checkpoint
from teacher.diffusion import Diffusion
from distillation.core import sample_velocity
cfg=Config(**state['config']);net=make_model(cfg,151).cuda().eval();net.load_state_dict(state['ema']);alpha=Diffusion(1000,cfg.beta_start,cfg.beta_end,torch.device('cuda')).alpha_bar
labels=torch.tensor([0,13],device='cuda')
for n,cls,mode in [(1,ModelOneNFE,'one_nfe'),(4,ModelFewNFE,'few_nfe')]:
 grid=torch.linspace(999,0,32).round().long()[::(32//n)];state['sampling_times']=grid;state['clip_denoised']=False;save_checkpoint(folder/f'synthetic_n{n}.pt',state)
 export(str(folder/f'synthetic_n{n}.pt'),str(folder/f'n{n}.ckpt'),mode)
 wrapper=cls(device='cuda');wrapper.load_state_dict(torch.load(folder/f'n{n}.ckpt',weights_only=True)['state_dict']);wrapper.eval()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  expected=sample_velocity(net,alpha,grid.cuda(),labels,42,False);actual=wrapper.sample((2,3,64,64),device='cuda',category=labels,seed=42)
 assert torch.equal(expected,actual),(n,(expected-actual).abs().max().item())
print('PREFLIGHT PASS: guidance and progressive real-batch64 update; 4-head export/sample exact parity',flush=True)
(ROOT/'experiments/preflight_pass.json').write_text(json.dumps({'guidance_updates':1,'progressive_updates':1,'batch':64,'heads':4,'export_sample_exact_parity':[1,4]}))
