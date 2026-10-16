"""Make complete, aligned one-image-per-class quality sheets from official PNGs."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='experiments/fixed_cfg2_progressive')
    parser.add_argument('--sample-index',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.sample_index<20:raise ValueError('Official evaluation provides indices 0..19')
    root=Path(args.root);out=root/'quality_review';out.mkdir(parents=True,exist_ok=True)
    sources={'one_nfe':root/'n1/official_evaluation/one_nfe',
             'few_nfe':root/'n4/official_evaluation/few_nfe'}
    names_by_id={number:name for name,number in json.loads(Path('data/pokemon-generation-one-22k/category_to_id.json').read_text()).items()}
    assert len(names_by_id)==151 and set(names_by_id)==set(range(151))
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    cols=8;rows=math.ceil(151/cols);tile_w=144;tile_h=154;label_h=22
    cached={mode:[] for mode in sources};hashes={mode:[] for mode in sources}
    for label in range(151):
        filename=f'{label:04d}_{args.sample_index:02d}.png'
        for mode,folder in sources.items():
            path=folder/filename
            with Image.open(path) as im:
                assert im.size==(64,64) and im.mode=='RGB',path
                cached[mode].append(im.copy())
            hashes[mode].append(hashlib.sha256(path.read_bytes()).hexdigest())
    suffix='' if args.sample_index==0 else f'_sample{args.sample_index:02d}'
    for mode in sources:
        canvas=Image.new('RGB',(cols*tile_w,rows*tile_h),'white');draw=ImageDraw.Draw(canvas)
        for label,image in enumerate(cached[mode]):
            x=(label%cols)*tile_w;y=(label//cols)*tile_h
            draw.text((x+3,y+3),f'{label:03d} {names_by_id[label]}',font=font,fill='black')
            canvas.paste(image.resize((128,128),Image.Resampling.NEAREST),(x+8,y+label_h))
        canvas.save(out/f'{mode}_all_151_classes{suffix}.png',optimize=True)
    paired=Image.new('RGB',(cols*tile_w,rows*(label_h+2*136+4)),'white');draw=ImageDraw.Draw(paired)
    for label in range(151):
        x=(label%cols)*tile_w;y=(label//cols)*(label_h+2*136+4)
        draw.text((x+3,y+3),f'{label:03d} {names_by_id[label]}',font=font,fill='black')
        for j,mode in enumerate(('few_nfe','one_nfe')):
            paired.paste(cached[mode][label].resize((128,128),Image.Resampling.NEAREST),
                         (x+8,y+label_h+j*136))
    paired.save(out/f'one_vs_four_nfe_all_151_classes{suffix}.png',optimize=True)
    metadata={'checkpoint_one':'checkpoints/one_nfe.ckpt','checkpoint_few':'checkpoints/few_nfe.ckpt',
              'source_images':{k:str(v) for k,v in sources.items()},'sample_index':args.sample_index,
              'labels':names_by_id,'sha256_per_label':hashes,
              'note':'Same class and official sample index per cell; top=4 NFE, bottom=1 NFE in paired sheet. PNGs are nearest-neighbor 2x enlargements for visual inspection; original samples are 64x64.'}
    (out/('metadata.json' if not suffix else f'metadata{suffix}.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    print('Wrote:',out/f'one_vs_four_nfe_all_151_classes{suffix}.png',flush=True)
    for mode in sources:print('Wrote:',out/f'{mode}_all_151_classes{suffix}.png',flush=True)

if __name__=='__main__':main()
