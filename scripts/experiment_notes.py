#!/usr/bin/env python3
"""Explicit Codex start/sync workflow. Only festgen/main may receive pushes."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from run import ROOT, now, write_json
from fid_report import normalize_metrics

LIMIT = 1048576


def git(*args, capture=True):
    result = subprocess.run(['git', *args], cwd=ROOT, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ''


def allowed(path):
    return (path.startswith(('docs/', 'configs/', 'reviews/')) or
            re.fullmatch(r'runs/exp-\d+/(meta\.json|metrics\.json|figures/[^/]+\.png)', path) is not None)


def preflight():
    if git('branch', '--show-current') != 'main':
        raise RuntimeError('Switch to main before syncing')
    url = git('remote', 'get-url', '--push', 'festgen')
    if url not in ('git@github.com:JoohoSon24/FestGen.git',
                   'git@github-FestGen:JoohoSon24/FestGen.git'):
        raise RuntimeError(f'Unexpected festgen push URL: {url}')
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Index must be empty before sync')
    # Pull/rebase must not hide local edits with stash or overwrite review content.
    if git('diff', '--name-only'):
        raise RuntimeError('Commit tracked edits before pull --rebase; no automatic stash')


def pull():
    preflight()
    git('pull', '--rebase', 'festgen', 'main', capture=False)


def records():
    return [(p, json.loads(p.read_text())) for p in sorted((ROOT / 'runs').glob('exp-*/meta.json'))]


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def metric_summary(metrics, prefix=''):
    result = []
    for key, value in metrics.items():
        name = key if key in ('official_fid', 'valid_fid') else f'{prefix}{key}'
        if isinstance(value, dict):
            result.extend(metric_summary(value, name + '.'))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result.append(f'{name}={value:.6g}')
    # Keep scalar measurements; never conflate development and official FID names.
    selected = [s for s in result if any(k in s.lower().split('=')[0] for k in
                ('fid', 'loss', 'step', 'nfe', 'parameter_count'))]
    return selected or result[:8]


def update_results():
    path = ROOT / 'docs' / 'RESULTS.md'
    text = path.read_text() if path.exists() else '# 실험 결과\n\n| id | parent | 가설 | 핵심 지표 | 결론 |\n|---|---|---|---|---|\n'
    added = []
    for meta_path, meta in records():
        if meta['status'] != 'done' or re.search(r'^\| ' + re.escape(meta['id']) + r' \|', text, re.M):
            continue
        metrics = json.loads(meta_path.with_name('metrics.json').read_text())
        prefix = metrics.get('reference_split', '')
        summary = '; '.join(metric_summary(metrics, prefix + '.' if prefix else '')) or '핵심 수치 미확인'
        settings = [f'{key}={metrics[key]}' for key in ('sampler', 'weights', 'guidance_scale', 'seed', 'reference_count') if key in metrics]
        if settings:
            summary += '; ' + '; '.join(settings)
        conclusion = meta.get('conclusion') or ('소급 기록; 원본 측정값 보존, 인과 결론 미확인' if meta.get('backfilled')
                                             else '실행 완료; 가설의 성립 여부는 리뷰 필요')
        text += '| ' + ' | '.join(map(cell, [meta['id'], meta.get('parent_id'), meta.get('hypothesis'), summary, conclusion])) + ' |\n'
        added.append(meta['id'])
    if added:
        path.write_text(text)
    return added


def check_reviews(path):
    relative = str(path.relative_to(ROOT))
    original = subprocess.run(['git', 'show', 'HEAD:' + relative], cwd=ROOT,
                              text=True, capture_output=True)
    if original.returncode:
        raise RuntimeError(f'Server cannot create reviewer files: {relative}')
    old = json.loads(original.stdout); new = json.loads(path.read_text())
    for document in (old, new):
        document.pop('status', None); document.pop('resolution', None)
    if old != new:
        raise RuntimeError(f'Only status/resolution may change: {relative}')


def stage_notes():
    # Honor existing index and stage nothing outside the explicit allowlist.
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Index is not empty')
    names = set(git('ls-files', '--modified', '--others', '--exclude-standard', '-z').split('\0')) - {''}
    selected = []
    for name in sorted(names):
        if not allowed(name):
            continue
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f'No deletions/symlinks allowed: {name}')
        if path.stat().st_size > LIMIT:
            raise RuntimeError(f'New/changed note exceeds 1MiB: {name}')
        if path.suffix in ('.pt', '.pth', '.ckpt', '.safetensors', '.log', '.zip', '.npz', '.npy'):
            raise RuntimeError(f'Forbidden artifact: {name}')
        if path.suffix == '.png' and path.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
            raise RuntimeError(f'Invalid PNG: {name}')
        if name.startswith('reviews/'):
            check_reviews(path)
        if name.startswith('configs/') and subprocess.run(
                ['git', 'cat-file', '-e', 'HEAD:' + name], cwd=ROOT, capture_output=True).returncode == 0:
            raise RuntimeError(f'Existing config is immutable: {name}')
        selected.append(name)
    if selected:
        git('add', '--', *selected)
    return selected


def sync(local=False):
    if not local:
        pull()
    added = update_results()
    selected = stage_notes()
    if selected:
        ident = added[-1] if added else (records()[-1][1]['id'] if records() else 'exp-000')
        git('commit', '-m', f'{ident}: synchronize experiment notes', capture=False)
    if not local:
        # Pull again immediately before push to account for concurrent reviews.
        pull()
        git('push', 'festgen', 'HEAD:refs/heads/main', capture=False)
    print(json.dumps({'added_experiments': added, 'committed_files': selected,
                      'pushed': not local, 'mode': 'local-only' if local else 'festgen/main'}, ensure_ascii=False))
    return added


def alive(meta):
    if meta.get('host') != socket.gethostname() or not meta.get('pid') or not meta.get('process_start'):
        return '미확인: host/PID 정보 부족'
    result = subprocess.run(['ps', '-p', str(meta['pid']), '-o', 'args='], capture_output=True, text=True)
    try:
        stamp = Path(f"/proc/{meta['pid']}/stat").read_text().split()[21]
    except (FileNotFoundError, PermissionError):
        stamp = None
    return ('생존: ' + result.stdout.strip()) if result.returncode == 0 and stamp == meta['process_start'] else 'failed 후보: 프로세스 없음 또는 PID 재사용'


def report(updated):
    print('\n실험 현황')
    results = (ROOT / 'docs' / 'RESULTS.md').read_text()
    counts = {}
    for _, meta in records():
        status = meta.get('status')
        counts[status] = counts.get(status, 0) + 1
        detail = alive(meta) if status == 'running' else (meta.get('error_summary') or '')
        if status == 'done' and not re.search(r'^\| ' + re.escape(meta['id']) + r' \|', results, re.M):
            detail = 'RESULTS 미반영'
        print(f"- {meta['id']} {status}: {detail}")
    print(json.dumps(counts, ensure_ascii=False))
    print('\n이번에 갱신한 문서\n- ' + (', '.join(updated) + ' → docs/RESULTS.md' if updated else '없음'))
    print('\n처리 대기 리뷰')
    reviews = []
    for path in (ROOT / 'reviews').glob('*.json'):
        value = json.loads(path.read_text())
        if value.get('status') == 'open':
            priority = value.get('priority', 'P2')
            ranks = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3,
                     'P0': 0, 'P1': 1, 'P2': 2, 'P3': 3}
            rank = ranks.get(priority, priority if isinstance(priority, int) else 99)
            reviews.append((rank, path.name, value))
    for _, name, value in sorted(reviews):
        print(f"- {name} [{value.get('priority')}]: {value.get('summary', value.get('title', '요약 미기재'))}")
    if not reviews:
        print('- 없음')
    print('\n제안하는 다음 실험\n- GPU 드라이버 상태 확인 후, 미확인 lineage와 split을 검증하고 parent/hypothesis를 명시한 단일 변수 실험을 계획합니다.')


def backfill():
    from run import FIELDS, lock, next_id
    known = {m.get('source_path') for _, m in records()}
    created = []
    with lock():
        for path in sorted((ROOT / 'experiments').rglob('metrics.json')):
            relative = str(path.relative_to(ROOT))
            if relative in known or path.stat().st_size > LIMIT:
                continue
            metrics = json.loads(path.read_text())
            if not isinstance(metrics, dict):
                continue
            ident = next_id()
            directory = ROOT / 'runs' / ident
            directory.mkdir(parents=True)
            meta = dict.fromkeys(FIELDS)
            # A completed evaluation does not imply its parent training completed.
            training = (path.parent / 'final.pt').exists()
            evaluation = isinstance(metrics.get('fid'), (int, float)) and not isinstance(metrics.get('fid'), bool)
            meta.update(id=ident, backfilled=True, source_path=relative,
                        record_type='evaluation' if evaluation else 'training_snapshot',
                        status='done' if evaluation or training else None,
                        git_commit=metrics.get('git_commit'), seed=metrics.get('seed'),
                        error_summary=None,
                        evidence='FID measurement exists' if evaluation else 'final.pt exists' if training else 'Completion unconfirmed')
            config = path.parent / 'config.json'
            if config.exists():
                value = json.loads(config.read_text())
                target = ROOT / 'configs' / f'{ident}.yaml'
                write_json(target, {'id': ident, 'backfilled': True, 'source_path': str(config.relative_to(ROOT)),
                                    'original_config': value})
                meta['config_path'] = str(target.relative_to(ROOT))
            write_json(directory / 'meta.json', meta)
            official = path.with_name('official_metrics.json')
            if official.is_file() and official.stat().st_size <= LIMIT:
                metrics = {'development': metrics, 'official': json.loads(official.read_text())}
                meta['official_source_path'] = str(official.relative_to(ROOT))
                write_json(directory / 'meta.json', meta)
            write_json(directory / 'metrics.json', normalize_metrics(metrics))
            created.append(ident)
    print(json.dumps({'backfilled': created, 'count': len(created)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'sync', 'backfill', 'report'])
    parser.add_argument('--local', action='store_true', help='Offline validation only: no pull or push')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'backfill':
        backfill()
    elif args.action == 'report':
        report([])
    elif args.action == 'sync':
        sync(args.local)
    else:
        if not args.local:
            pull()
        report([])
        updated = sync(args.local)
        report(updated)

if __name__ == '__main__':
    main()
