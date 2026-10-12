"""Reevaluate legacy4NFE through original unchanged workspace and compare PNGs."""
from pathlib import Path
import sys,os,json,hashlib,math
W=Path('/root/Diffusion/workspace_distillation');R=Path('/root/Diffusion/recipe_distill_dmd');os.chdir(W);sys.path.insert(0,str(W))
import torch
from model import ModelFewNFE
from evaluate import generate_samples,compute_fid
from PIL import Image,ImageDraw,ImageFont
OUT=R/'experiments/compare_old_new_n4';OUT.mkdir(parents=True,exist_ok=True)
torch.cuda.set_per_process_memory_fraction(.25)
source=W/'checkpoints/few_nfe.ckpt';gen=OUT/'legacy_generated';mp=OUT/'legacy_official_metrics.json'
if not gen.exists():
 m=ModelFewNFE(device='cuda');state=torch.load(source,map_location='cpu',weights_only=True);m.load_state_dict(state['state_dict']);m.eval();del state
 assert m.count_parameters()==48883107
 generate_samples(m,gen,torch.arange(151).repeat_interleave(20),16,'cuda');del m;torch.cuda.empty_cache()
assert len(list(gen.glob('*.png')))==3020
if mp.exists():legacy=json.loads(mp.read_text())
else:
 fid=compute_fid(gen,W/'results/reference','cuda',16);legacy={'fid':fid,'checkpoint':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'nfe':4,'seed':42,'batch_size':16,'sample_count':3020,'reference_count':3020,'weights':'EMA','precision':'bf16'};mp.write_text(json.dumps(legacy,indent=2))
new=json.loads((R/'experiments/recipe/official_progress/n4_025000.json').read_text());newgen=Path(new['source_generated'])
names={v:k for k,v in json.loads((W/'data/pokemon-generation-one-22k/category_to_id.json').read_text()).items()}
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
except OSError:font=ImageFont.load_default()
cols=8;tw=160;th=295;h=70
selected=[0,3,6,13,24,38,49,50,63,74,93,103,129,132,133,150]
def grid(labels,path):
 canvas=Image.new('RGB',(cols*tw,h+math.ceil(len(labels)/cols)*th),'white');draw=ImageDraw.Draw(canvas)
 draw.text((10,10),f'4 NFE | TOP: legacy Official FID {legacy["fid"]:.4f} | BOTTOM: new {new["fid"]:.4f}',font=font,fill='black')
 draw.text((10,34),'Matched class and sample index 0; seed42 + batch index, batch16; nearest-neighbor 2x',font=font,fill='black')
 for i,c in enumerate(labels):
  x=(i%cols)*tw;y=h+(i//cols)*th;draw.text((x+4,y),f'{c:03d} {names[c]}',font=font,fill='black')
  for j,folder in enumerate([gen,newgen]):
   with Image.open(folder/f'{c:04d}_00.png') as im:canvas.paste(im.convert('RGB').resize((128,128),Image.Resampling.NEAREST),(x+8,y+23+j*134))
 canvas.save(path)
grid(selected,OUT/'comparison_16_classes.png');grid(list(range(151)),OUT/'comparison_all151.png')
result={'legacy':legacy,'new':new,'official_delta_new_minus_legacy':new['fid']-legacy['fid'],'historical_legacy_fid':33.0517,'legacy_reproduction_delta':legacy['fid']-33.0517,'pairing':'same category and sample index0, seed42+batch index, batch16','head_counts':{'legacy':1,'new':4},'teacher_budgets':{'legacy':100000,'new':150000}}
(OUT/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
