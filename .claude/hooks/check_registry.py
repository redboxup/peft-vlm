#!/usr/bin/env python3
"""Check that method names line up across the four registration sites.

A method is wired together by one name string: a config registered in
configs.py, a model decorated with @BaseCLIP.register_model, a trainer
decorated with @BaseTrainer.register_trainer, and an import in the package
__init__.py so the decorator actually runs.  This checks all four by text,
so it works even while the packages cannot be imported.

Pre-existing drift is recorded in registry_baseline.json and stays quiet;
only new drift is reported.  Run with --update-baseline after fixing (or
knowingly adding) something.
"""
import re
import sys
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
BASELINE = pathlib.Path(__file__).resolve().parent / 'registry_baseline.json'

CONFIG_RE = re.compile(
    r"cs\.store\(\s*name\s*=\s*['\"]([^'\"]+)['\"]\s*,"
    r"\s*group\s*=\s*['\"]trainer['\"]"
)
MODEL_RE = re.compile(r"register_model\(\s*['\"]([^'\"]+)['\"]")
TRAINER_RE = re.compile(r"register_trainer\(\s*['\"]([^'\"]+)['\"]")
IMPORT_RE = re.compile(r"^from \.(\w+) import", re.MULTILINE)


def read(path):
    try:
        return path.read_text()
    except OSError:
        return ''


def collect():
    """Return the current state of every registration site."""
    configs = set(CONFIG_RE.findall(read(REPO / 'configs.py')))

    models, trainers = set(), set()
    for path in (REPO / 'models').glob('*.py'):
        models |= set(MODEL_RE.findall(read(path)))
    for path in (REPO / 'trainers').glob('*.py'):
        trainers |= set(TRAINER_RE.findall(read(path)))

    missing_modules = []
    for package in ('models', 'trainers'):
        init = REPO / package / '__init__.py'
        for module in IMPORT_RE.findall(read(init)):
            if not (REPO / package / f'{module}.py').exists():
                missing_modules.append(f'{package}/{module}.py')

    return {
        'no_model': sorted(configs - models),
        'no_trainer': sorted(configs - trainers),
        'no_config': sorted((models | trainers) - configs),
        'missing_modules': sorted(missing_modules),
    }


LABELS = {
    'no_model': 'registered in configs.py but no @BaseCLIP.register_model',
    'no_trainer': 'registered in configs.py but no @BaseTrainer.register_trainer',
    'no_config': 'registered in models/ or trainers/ but no cs.store in configs.py',
    'missing_modules': '__init__.py imports a module that does not exist',
}


def main():
    state = collect()

    if '--update-baseline' in sys.argv:
        BASELINE.write_text(json.dumps(state, indent=2) + '\n')
        print(f'baseline written: {BASELINE}')
        return 0

    try:
        baseline = json.loads(BASELINE.read_text())
    except (OSError, ValueError):
        baseline = {k: [] for k in state}

    problems = []
    for key, items in state.items():
        new = [i for i in items if i not in baseline.get(key, [])]
        if new:
            problems.append(f'  {LABELS[key]}:\n    ' + '\n    '.join(new))

    if problems:
        print('Method registry is inconsistent:\n' + '\n'.join(problems) +
              '\n  See .claude/skills/add-method/SKILL.md. If this is '
              'intentional, run .claude/hooks/check_registry.py '
              '--update-baseline', file=sys.stderr)
        return 2

    fixed = {
        key: [i for i in baseline.get(key, []) if i not in state[key]]
        for key in state
    }
    if any(fixed.values()):
        resolved = sorted(i for items in fixed.values() for i in items)
        print('Registry entries in the baseline are now resolved: '
              + ', '.join(resolved)
              + '. Run .claude/hooks/check_registry.py --update-baseline '
                'to stop tracking them.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
