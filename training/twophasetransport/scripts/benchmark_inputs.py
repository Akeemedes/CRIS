"""Input locations and shared I/O for the paired transport benchmarks."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT/'Figures'
BASE = ROOT/'training/twophasetransport/analysis/regular_v3'
DEST = BASE/'benchmarks'
KEYS = {'DX_FILE', 'DY_FILE', 'DZ_FILE', 'KX_FILE', 'KY_FILE', 'KZ_FILE',
        'DISPX_FILE', 'DISPY_FILE', 'DISPZ_FILE', 'BC_FILE', 'INIT_FILE',
        'INTR_VEL_FILE', 'BNDR_VEL_FILE', 'CL_FILE', 'TRANS_FILE', 'DEPTH_FILE',
        'VOL_FILE', 'GLOBAL_IDX_FILE', 'WELLS_FILE'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sources():
    yield '1D', FIG/'fig4_transport/benchmark_1d/CRIS/sim.txt'
    for slug in ('2DSPE10_Layer50', '2DSPE10_Layer75', '2DSPE10_Layer84', 'Norne_5SpotBase'):
        yield slug, FIG/f'fig4_transport/benchmark_cases/cases/{slug}/CRIS/sim.txt'
    yield 'EDFM', FIG/'fig4_transport/benchmark_edfm/CRIS/sim.txt'
    yield '3DSPE10_5SpotBase', FIG/'fig4_transport/benchmark_cases/cases/3DSPE10_5SpotBase/CRIS/sim.txt'
