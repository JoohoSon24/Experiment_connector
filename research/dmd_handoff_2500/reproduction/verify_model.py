"""Read-only inference verification using only packaged code and one checkpoint."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'source_overlay'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=PACKAGE / 'checkpoints/dmd_step_002500.ckpt')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    import torch
    import model as model_module
    from model import ModelOneNFE
    assert Path(model_module.__file__).resolve().is_relative_to(PACKAGE / 'source_overlay')
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA unavailable; CPU verification may be selected explicitly'
        assert torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    opened_checkpoints = set()
    def audit_open(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            p = Path(values[0].decode() if isinstance(values[0], bytes) else values[0])
            if p.suffix in ('.pt', '.ckpt', '.pth'):
                opened_checkpoints.add(str(p.resolve()))
    sys.addaudithook(audit_open)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ModelOneNFE(device=args.device)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    count = model.count_parameters()
    assert count == 48883107
    assert model.sampling_times.tolist() == [999]
    assert not hasattr(model, 'consistency_sigma_data')
    records = []
    for batch in (1, 2, 16):
        calls = []
        handle = model.backbone.register_forward_hook(lambda *a: calls.append(1))
        with torch.inference_mode():
            images = model.sample((batch, 3, 64, 64), device=args.device,
                                  category=torch.arange(batch, device=args.device), seed=42)
        handle.remove()
        finite = bool(torch.isfinite(images).all())
        assert len(calls) == 1 and finite and list(images.shape) == [batch, 3, 64, 64]
        records.append({'batch': batch, 'backbone_forwards': len(calls), 'finite': finite,
                        'shape': list(images.shape), 'min': float(images.min()), 'max': float(images.max())})
    h = hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    assert opened_checkpoints == {str(args.checkpoint.resolve())}, opened_checkpoints
    print(json.dumps({'status': 'PASS', 'checkpoint': str(args.checkpoint.resolve()),
                      'sha256': h.hexdigest(), 'parameter_count': count,
                      'sampling_times': [999], 'measured_batches': records,
                      'device': args.device, 'torch': torch.__version__,
                      'inference_cfg': False, 'loaded_checkpoint_count': 1,
                      'auxiliary_checkpoint_dependencies': [], 'opened_checkpoint_files': sorted(opened_checkpoints),
                      'module_path': str(Path(model_module.__file__).resolve()),
                      'evidence_kind': 'new read-only handoff inference verification'}, indent=2))


if __name__ == '__main__':
    main()
