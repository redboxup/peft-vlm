#!/usr/bin/env python3
"""Keep model weights and other large files out of commits.

Experiment runs drop checkpoints (best_epoch.pth) and caches next to the
code, and .gitignore does not cover every shape of them, so a broad
'git add' can stage hundreds of MB.  This inspects what a git add/commit
is about to pick up and blocks on binary artefacts or large files.
"""
import os
import sys
import json
import subprocess

LARGE_BYTES = 10 * 1024 * 1024
ARTEFACT_SUFFIXES = (
    '.pth', '.pt', '.ckpt', '.safetensors', '.h5', '.hdf5',
    '.npy', '.npz', '.pkl', '.bin',
)


def git(*args, cwd):
    out = subprocess.run(['git', *args], cwd=cwd, capture_output=True,
                         text=True)
    return out.stdout if out.returncode == 0 else ''


def offenders(paths, cwd):
    for path in paths:
        if not path:
            continue
        reason = None
        if path.lower().endswith(ARTEFACT_SUFFIXES):
            reason = 'model weights / binary artefact'
        else:
            try:
                size = os.path.getsize(os.path.join(cwd, path))
            except OSError:
                continue
            if size > LARGE_BYTES:
                reason = f'{size / 1024 / 1024:.0f} MB'
        if reason:
            yield f'{path} ({reason})'


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0

    command = payload.get('tool_input', {}).get('command', '')
    cwd = payload.get('cwd') or os.getcwd()

    if 'git commit' not in command and 'git add' not in command:
        return 0

    paths = []
    if 'git commit' in command:
        paths += git('diff', '--cached', '--name-only', cwd=cwd).splitlines()

    # a broad 'git add' stages whatever is currently dirty or untracked
    if 'git add' in command and any(
        flag in command for flag in (' -A', ' --all', ' .', ' -a ')
    ):
        for line in git('status', '--porcelain', cwd=cwd).splitlines():
            paths.append(line[3:].strip())

    found = sorted(set(offenders(paths, cwd)))
    if found:
        print('Refusing to stage files that do not belong in git:\n  '
              + '\n  '.join(found)
              + '\n Add them to .gitignore, or stage the intended files by '
                'name instead.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
