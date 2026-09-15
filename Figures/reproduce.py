"""List and recreate manuscript figures from supplied numerical results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def command(record):
    return [sys.executable, str(HERE/record['builder']), *record.get('arguments', [])]


def build(record):
    output = HERE/record['output']
    if not record.get('builder'):
        if not output.is_file():
            raise FileNotFoundError(output)
        print(f'Supplied artwork: {output.relative_to(ROOT)}')
        return
    for relative in record.get('inputs', []):
        if not (ROOT/relative).is_file():
            raise FileNotFoundError(ROOT/relative)
    print(f"Building {record['id']}: {record['name']}", flush=True)
    subprocess.run(command(record), cwd=ROOT, check=True)
    if not output.is_file():
        raise FileNotFoundError(f'Builder did not produce {output}')
    print(f'Wrote {output.relative_to(ROOT)}', flush=True)


def main():
    records = json.loads((HERE/'catalogue.json').read_text())['figures']
    figures = {record['id']: record for record in records}
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('list', help='list available figures')
    show = actions.add_parser('show', help='show a figure command and output location')
    show.add_argument('figure', choices=sorted(figures))
    render = actions.add_parser('build', help='recreate figures from supplied results')
    render.add_argument('figures', nargs='+', choices=sorted(figures))
    render.add_argument('--dry-run', action='store_true', help='show commands without running them')
    args = parser.parse_args()
    if args.action == 'list':
        for identifier, record in figures.items():
            print(f"{identifier:<25} {record['name']}")
    elif args.action == 'show':
        record = figures[args.figure]
        print(record['name'])
        print('Output:', Path('Figures')/record['output'])
        if record.get('builder'):
            print('Command:', subprocess.list2cmdline(command(record)))
        else:
            print('Supplied artwork; no numerical builder.')
    else:
        for identifier in args.figures:
            record = figures[identifier]
            if args.dry_run:
                print(subprocess.list2cmdline(command(record)) if record.get('builder') else record['output'])
            else:
                build(record)


if __name__ == '__main__':
    main()
