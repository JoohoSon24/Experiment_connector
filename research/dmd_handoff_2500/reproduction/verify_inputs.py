"""Read-only preflight for the archived DMD training script."""
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--allow-existing-run', action='store_true', help='Read-only audit mode; not a launch preflight')
    args = parser.parse_args()
    root = args.workspace.resolve()
    if not args.allow_existing_run:
        assert not (root / 'experiments/night_run/dmd_preserve_team').exists(), 'Run directory already exists'
    hashes = json.loads((PACKAGE / 'verification/source_hashes.json').read_text())
    identities = hashes['retraining_required_files']
    for rel, record in identities.items():
        assert (root / rel).stat().st_size == record['bytes'], rel
        assert digest(root / rel) == record['sha256'], rel
    overlay = json.loads((PACKAGE / 'source_overlay/OVERLAY_MANIFEST.json').read_text())
    for rel, record in overlay['files'].items():
        assert digest(root / rel) == record['sha256'], rel
    data = root / 'data/pokemon-generation-one-22k'
    dev = root / 'data/development/seed42_holdout5'
    lists = [(data / 'train_split.txt').read_text().splitlines(), (dev / 'train.txt').read_text().splitlines(),
             (dev / 'holdout.txt').read_text().splitlines(), (data / 'val_split.txt').read_text().splitlines()]
    a, b, c, d = map(set, lists)
    checks = {'dev_train_subset_original_train': b <= a, 'dev_holdout_subset_original_train': c <= a,
              'dev_train_holdout_disjoint': not b & c, 'dev_holdout_validation_disjoint': not c & d,
              'original_train_validation_disjoint': not a & d, 'dev_union_equals_original_train': b | c == a,
              'no_duplicate_original_train': len(a) == len(lists[0]),
              'no_duplicate_dev_train': len(b) == len(lists[1]), 'no_duplicate_holdout': len(c) == len(lists[2])}
    assert all(checks.values()) and list(map(len, lists)) == [17079, 16324, 755, 3020]
    audit = json.loads((root / 'experiments/night_run/audit_results.json').read_text())
    assert checks == audit['checks']
    for rel, expected in audit['protected_sha256'].items():
        assert digest(root / rel) == expected, rel
    manifest = json.loads((root / 'experiments/night_run/reference_manifest.json').read_text())['images']
    ref = root / 'experiments/night_run/reference_development'
    assert lists[2] == [r['source'] for r in manifest]
    assert {p.name for p in ref.glob('*.png')} == {r['reference'] for r in manifest}
    assert all(digest(ref / r['reference']) == r['sha256'] for r in manifest)
    sys.path.insert(0, str(root))
    import torch
    from model import ModelOneNFE, ModelFewNFE
    from teacher.config import Config
    from teacher.utils import make_model
    torch.set_num_threads(4)
    counts = {}
    for name, cls in [('one_nfe', ModelOneNFE), ('few_nfe', ModelFewNFE)]:
        state = torch.load(root / f'checkpoints/{name}.ckpt', map_location='cpu', weights_only=True)
        model = cls(device='cpu'); model.load_state_dict(state['state_dict'])
        counts[name] = model.count_parameters()
        assert counts[name] == 48883107
        assert model.sampling_times.tolist() == ([999] if name == 'one_nfe' else [999, 741, 483, 226])
        del state, model
        gc.collect()
    state = torch.load(root / 'experiments/night_run/fresh_teacher/final.pt', map_location='cpu', weights_only=True)
    expected_train = hashlib.sha256('\n'.join(lists[1]).encode()).hexdigest()
    assert state['train_paths_sha256'] == expected_train
    model = make_model(Config(**state['config']), 151)
    model.load_state_dict(state['ema'])
    counts['real_score'] = sum(p.numel() for p in model.parameters())
    assert counts['real_score'] == 48883107
    cuda = {'torch_cuda': torch.version.cuda, 'available': torch.cuda.is_available()}
    if args.device == 'cuda':
        assert cuda['available'] and torch.cuda.is_bf16_supported()
        cuda.update(name=torch.cuda.get_device_name(), total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                    bf16_supported=torch.cuda.is_bf16_supported())
    print(json.dumps({'status': 'PASS', 'workspace': str(root), 'read_only_audit_mode': args.allow_existing_run,
                      'verified_required_files': len(identities), 'verified_overlay_files': len(overlay['files']),
                      'split_checks': checks, 'split_counts': list(map(len, lists)),
                      'reference_png_hash_matches': len(manifest), 'parameter_counts': counts,
                      'train_paths_sha256': expected_train, 'cuda': cuda,
                      'training_peak_measured_by_this_check': False,
                      'evidence_kind': 'new read-only handoff preflight'}, indent=2))


if __name__ == '__main__':
    main()
