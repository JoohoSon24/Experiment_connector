"""Reject new/modified large artifacts while retaining approved historical files."""
import subprocess
import sys
from pathlib import PurePosixPath


def main():
    raw = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM', '-z'])
    errors = []
    for path in filter(None, raw.decode().split('\0')):
        parts = PurePosixPath(path).parts
        size = int(subprocess.check_output(['git', 'cat-file', '-s', ':' + path]))
        forbidden = (PurePosixPath(path).suffix in {'.pt', '.pth', '.ckpt', '.safetensors', '.npy', '.npz'}
                     or 'wandb' in parts or ('runs' in parts and ('logs' in parts or 'ckpt' in parts)))
        # Preserve small split manifests; no new raw data.
        if parts and parts[0] == 'data':
            forbidden |= not (PurePosixPath(path).suffix in {'.txt', '.json'} and size <= 1048576)
        if forbidden or size > 1048576:
            errors.append(f'{path}: forbidden artifact or exceeds 1MiB ({size} bytes)')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
