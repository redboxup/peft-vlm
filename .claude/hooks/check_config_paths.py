#!/usr/bin/env python3
"""Flag machine-specific absolute paths newly added to a config file.

allconfigs/dataset_configs.py already carries hardcoded /mnt/... paths that
only resolve on one machine.  Only lines this edit added are checked, so
the existing ones stay quiet while new ones get caught.
"""
import os
import re
import sys
import json
import subprocess

MACHINE_PATHS = re.compile(r'/(mnt|home|Users|scratch|media)/')


def added_lines(path, cwd):
    """Lines this working tree adds to the file relative to HEAD."""
    tracked = subprocess.run(
        ['git', 'ls-files', '--error-unmatch', '--', path],
        cwd=cwd, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        # untracked file: every line is new
        try:
            with open(os.path.join(cwd, path)) as fh:
                return list(fh)
        except OSError:
            return []

    diff = subprocess.run(
        ['git', 'diff', 'HEAD', '--unified=0', '--', path],
        cwd=cwd, capture_output=True, text=True,
    )
    if diff.returncode != 0:
        return []

    return [
        line[1:] for line in diff.stdout.splitlines()
        if line.startswith('+') and not line.startswith('+++')
    ]


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0

    file_path = payload.get('tool_input', {}).get('file_path', '')
    cwd = payload.get('cwd') or os.getcwd()
    if not file_path:
        return 0

    rel = os.path.relpath(file_path, cwd)
    if not (rel.startswith('allconfigs/') or rel == 'configs.py'):
        return 0

    hits = [line.strip() for line in added_lines(rel, cwd)
            if MACHINE_PATHS.search(line)]
    if hits:
        print(f'{rel} gained a machine-specific absolute path:\n  '
              + '\n  '.join(hits)
              + '\n Configs are resolved against DATAPATH (see run.py), so '
                'prefer a path relative to configs.rootpath.',
              file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
