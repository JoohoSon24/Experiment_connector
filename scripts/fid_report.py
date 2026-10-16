#!/usr/bin/env python3
"""Score the same PNGs against fixed official/valid references using cached stats.

No generation/training. Cache files and images are never committed.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_digest(directory, expected):
    files = sorted(Path(directory).glob('*.png'))
    if len(files) != expected:
        raise ValueError(f'{directory}: expected {expected} PNGs, got {len(files)}')
    hasher = hashlib.sha256()
    for path in files:
        hasher.update(path.name.encode())
        hasher.update(bytes.fromhex(digest(path)))
    return hasher.hexdigest()


def normalize_metrics(value):
    """Keep source fields, add explicit nullable scores, never guess an unlabeled FID."""
    result = dict(value)
    result.setdefault('official_fid', None)
    result.setdefault('valid_fid', None)
    for key, name in [('official', 'official_fid'), ('development', 'valid_fid'), ('valid', 'valid_fid')]:
        nested = result.get(key)
        if isinstance(nested, dict) and isinstance(nested.get('fid'), (int, float)):
            result[name] = nested['fid']
    reference = result.get('reference_split')
    if isinstance(result.get('fid'), (int, float)):
        if reference == 'official':
            result['official_fid'] = result['fid']
        elif reference in ('development', 'valid'):
            result['valid_fid'] = result['fid']
    return result


def validate_generation(protocol, metrics):
    for key in ('sample_count', 'batch_size', 'sampler', 'steps', 'guidance_scale',
                'nfe_per_sample', 'seed', 'precision', 'clip_denoised'):
        if metrics.get(key) != protocol[key]:
            raise ValueError(f'Generation differs from locked protocol: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-dir', required=True)
    parser.add_argument('--generation-metrics', required=True)
    parser.add_argument('--protocol', default='configs/fid_protocol_v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--splits', nargs='+', choices=['official', 'valid'], default=['official', 'valid'])
    args = parser.parse_args()
    os.chdir(ROOT)
    import numpy as np
    import torch
    from cleanfid import fid
    protocol = json.loads(Path(args.protocol).read_text())
    if version('clean-fid') != protocol['implementation_version']:
        raise ValueError('clean-fid version differs from locked protocol')
    source = json.loads(Path(args.generation_metrics).read_text())
    validate_generation(protocol, source)
    generated = Path(args.generated_dir)
    expected = {f'{label:04d}_{index:02d}.png' for label in range(protocol['num_classes'])
                for index in range(protocol['samples_per_class'])}
    if {p.name for p in generated.glob('*.png')} != expected:
        raise ValueError('Need exactly 20 PNGs for each of 151 classes')
    generation_hash = image_digest(generated, protocol['sample_count'])
    device = torch.device(args.device)
    model = fid.build_feature_extractor(protocol['mode'], device, use_dataparallel=False)
    def stats(directory):
        features = fid.get_folder_features(str(directory), model, num_workers=protocol['num_workers'],
                                           batch_size=protocol['batch_size'], device=device,
                                           mode=protocol['mode'])
        return np.mean(features, axis=0), np.cov(features, rowvar=False)
    mu, sigma = stats(generated)
    result = normalize_metrics(source)
    result.update(fid_protocol=protocol, fid_protocol_sha256=digest(args.protocol),
                  generated_images_sha256=generation_hash, reference_stats={})
    for split in args.splits:
        reference = protocol['reference'][split]
        if digest(reference['manifest_path']) != reference['manifest_sha256']:
            raise ValueError(f'Reference manifest changed: {split}')
        images_hash = image_digest(reference['reference_dir'], reference['reference_count'])
        cache = Path(reference['stats_path'])
        provenance = {'images_sha256': images_hash, 'manifest_sha256': reference['manifest_sha256'],
                      'mode': protocol['mode'], 'clean_fid': version('clean-fid'),
                      'feature_extractor': protocol['feature_extractor']}
        if cache.exists():
            with np.load(cache, allow_pickle=False) as value:
                if json.loads(str(value['provenance'])) != provenance:
                    raise ValueError(f'Reference stats mismatch; do not overwrite: {cache}')
                ref_mu, ref_sigma = value['mu'], value['sigma']
        else:
            ref_mu, ref_sigma = stats(reference['reference_dir'])
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation; never overwrite existing reference stats.
            with cache.open('xb') as stream:
                np.savez(stream, mu=ref_mu, sigma=ref_sigma, provenance=json.dumps(provenance, sort_keys=True))
        result[split + '_fid'] = float(fid.frechet_distance(mu, sigma, ref_mu, ref_sigma))
        result['reference_stats'][split] = dict(provenance, path=str(cache), stats_sha256=digest(cache))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + '.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, output)
    print(json.dumps({key: result[key] for key in ('official_fid', 'valid_fid')}))

if __name__ == '__main__':
    main()
