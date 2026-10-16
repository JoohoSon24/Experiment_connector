"""One CPU optimizer step on the repository U-Net and synthetic inputs; no FID."""
import json
import os
from pathlib import Path
import torch
from teacher.model import ClassConditionalUNet

def main():
    torch.set_num_threads(1)
    torch.manual_seed(int(os.environ.get('EXP_SEED', 42)))
    model = ClassConditionalUNet(image_size=8, in_channels=3, num_classes=2,
                                model_channels=8, channel_mult=(1, 2), num_res_blocks=1,
                                attention_resolutions=(), dropout=0.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    images = torch.randn(2, 3, 8, 8)
    noise = torch.randn_like(images)
    prediction = model(images, torch.tensor([1, 2]), torch.tensor([0, 1]))
    loss = torch.nn.functional.mse_loss(prediction, noise)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    metrics = {'loss': float(loss.detach()), 'steps': 1, 'device': 'cpu',
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'data': 'synthetic; infrastructure smoke only', 'peak_vram_bytes': None}
    Path(os.environ['EXP_METRICS_PATH']).write_text(json.dumps(metrics) + '\n')
    print(json.dumps(metrics))

if __name__ == '__main__':
    main()
