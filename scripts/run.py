#!/usr/bin/env python3
"""Immutable experiment manifests (JSON is valid YAML), subprocess tracking."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import traceback
from fid_report import normalize_metrics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('id parent_id hypothesis config_path git_commit git_dirty seed gpu host command '
          'started_at finished_at status').split()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def next_id():
    numbers = [int(p.stem[4:]) for base in ('runs', 'configs')
               for p in (ROOT / base).glob('exp-*')
               if re.fullmatch(r'exp-\d+', p.stem)]
    return f'exp-{max(numbers, default=0) + 1:03d}'


def lock():
    path = Path(git('rev-parse', '--git-path', 'experiment-notes.lock'))
    if not path.is_absolute():
        path = ROOT / path
    handle = path.open('a')
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def prepare(args):
    spec = json.loads(Path(args.spec).read_text())
    if 'parent_id' not in spec or not spec.get('hypothesis'):
        raise ValueError('parent_id (null for root) and hypothesis are required')
    if not isinstance(spec.get('command'), list) or not spec['command']:
        raise ValueError('command must be a nonempty argv list')
    with lock():
        ident = next_id()
        spec['id'] = ident
        # Snapshot dependencies, not a mutable path to the original config.
        sources = spec.pop('snapshot_files', [])
        spec['snapshots'] = {}
        for source in sources:
            path = ROOT / source
            if path.suffix not in ('.json', '.yaml', '.yml') or path.stat().st_size > 1048576:
                raise ValueError(f'Invalid config snapshot: {source}')
            spec['snapshots'][source] = path.read_text()
        target = ROOT / 'configs' / f'{ident}.yaml'
        target.parent.mkdir(exist_ok=True)
        with target.open('x') as stream:
            stream.write(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
    print(target.relative_to(ROOT))


def run(args):
    config = Path(args.config).resolve()
    spec = json.loads(config.read_text())
    ident = spec['id']
    if not re.fullmatch(r'exp-\d{3,}', ident) or config != ROOT / 'configs' / f'{ident}.yaml':
        raise ValueError('Use configs/exp-XXX.yaml created with prepare')
    if 'parent_id' not in spec or not spec.get('hypothesis'):
        raise ValueError('parent_id and hypothesis required')
    if spec['parent_id'] is not None and not (ROOT / 'runs' / spec['parent_id'] / 'meta.json').is_file():
        raise ValueError('Unknown parent_id')
    # Refuse source/config edits, including untracked source files.
    if git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError('Commit all changes (and sync previous results) before starting; dirty run refused')
    relative = str(config.relative_to(ROOT))
    git('ls-files', '--error-unmatch', relative)
    with lock():
        run_dir = ROOT / 'runs' / ident
        run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ('ckpt', 'logs', 'figures'):
        (run_dir / sub).mkdir()
    mapping = {'python': sys.executable, 'run_dir': str(run_dir), 'config_path': str(config)}
    command = [part.format_map(mapping) for part in spec['command']]
    meta = dict.fromkeys(FIELDS)
    meta.update(id=ident, parent_id=spec['parent_id'], hypothesis=spec['hypothesis'],
                config_path=relative, git_commit=git('rev-parse', 'HEAD'), git_dirty=False,
                seed=spec.get('seed'), gpu=None, host=socket.gethostname(), command=command,
                started_at=now(), status='running', pid=os.getpid(),
                process_start=Path('/proc/self/stat').read_text().split()[21],
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    child = None
    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    write_json(run_dir / 'meta.json', meta)
    try:
        try:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=10)
            meta['gpu'] = probe.stdout.strip() if probe.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
        for source, contents in spec.get('snapshots', {}).items():
            destination = run_dir / 'ckpt' / 'configs' / source
            if not destination.resolve().is_relative_to((run_dir / 'ckpt').resolve()):
                raise ValueError('Snapshot path escapes run')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(contents.replace('{run_dir}', str(run_dir)).replace('{id}', ident))
        environment = dict(os.environ, EXP_RUN_DIR=str(run_dir),
                           EXP_METRICS_PATH=str(run_dir / 'logs' / 'final_metrics.json'),
                           EXP_SEED=str(spec.get('seed', 42)))
        with (run_dir / 'logs' / 'console.log').open('w') as output:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            meta['child_pid'] = child.pid
            write_json(run_dir / 'meta.json', meta)
            code = child.wait()
        meta['exit_code'] = code
        if code:
            tail = (run_dir / 'logs' / 'console.log').read_text(errors='replace').splitlines()[-5:]
            raise RuntimeError(f'Command exited {code}: ' + ' | '.join(tail)[-1500:])
        source = Path(spec.get('metrics_source', str(run_dir / 'logs' / 'final_metrics.json')).format_map(mapping))
        if not source.is_absolute():
            source = ROOT / source
        metrics = json.loads(source.read_text())
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError('Final metrics must be a nonempty JSON object')
        write_json(run_dir / 'metrics.json', normalize_metrics(metrics))
        meta['status'] = 'done'
    except BaseException as exc:
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        meta.update(status='failed', error_summary=f'{type(exc).__name__}: {exc}'[:2000])
        write_json(run_dir / 'metrics.json', {'official_fid': None, 'valid_fid': None, 'error_summary': meta['error_summary']})
        (run_dir / 'logs' / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        meta['finished_at'] = now()
        write_json(run_dir / 'meta.json', meta)
        print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--spec', required=True)
    p = sub.add_parser('run'); p.add_argument('--config', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    (prepare if args.action == 'prepare' else run)(args)
