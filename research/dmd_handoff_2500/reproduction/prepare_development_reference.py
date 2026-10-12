"""Reconstruct recorded development PNG bytes; never build official references."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]


def prepare(workspace, check_only=False):
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    workspace = Path(workspace).resolve()
    out = workspace / 'experiments/night_run/reference_development'
    if not check_only:
        assert workspace != Path('/root/DMD'), 'Refusing to write the inspected source workspace'
        assert not out.exists(), f'Refusing existing output: {out}'
    manifest = json.loads((PACKAGE / 'provenance/reference_manifest.json').read_text())
    train = set((workspace / 'data/pokemon-generation-one-22k/train_split.txt').read_text().splitlines())
    validation = set((workspace / 'data/pokemon-generation-one-22k/val_split.txt').read_text().splitlines())
    held = (workspace / 'data/development/seed42_holdout5/holdout.txt').read_text().splitlines()
    records = manifest['images']
    assert len(records) == 755 and held == [r['source'] for r in records]
    assert set(held) <= train and not set(held) & validation
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                    transforms.Normalize((.5,) * 3, (.5,) * 3)])
    payloads = []
    for item in records:
        name = item['reference']
        assert Path(name).name == name and name.endswith('.png')
        with Image.open(workspace / 'data/pokemon-generation-one-22k' / item['source']) as im:
            x = transform(im.convert('RGB'))
        buf = io.BytesIO()
        to_pil_image(((x + 1) / 2).clamp(0, 1)).convert('RGB').save(buf, format='PNG')
        payload = buf.getvalue()
        assert hashlib.sha256(payload).hexdigest() == item['sha256'], f'Reference mismatch: {name}'
        payloads.append((name, payload))
    if not check_only:
        out.mkdir(parents=True, exist_ok=False)
        for name, payload in payloads:
            with (out / name).open('xb') as f:
                f.write(payload)
    return {'status': 'PASS', 'reference_count': len(payloads), 'exact_hash_matches': len(payloads),
            'check_only': check_only, 'output': str(out), 'bytes': sum(len(b) for _, b in payloads),
            'evidence_kind': 'new reference byte reconstruction check'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.check_only), indent=2))
