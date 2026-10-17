"""Reproduce the seven reported sensitivity analyses, three partitions each."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
args = parser.parse_args()
settings = [
    ('cal100', ['--cal-cap', '100']),
    ('cal1000', ['--cal-cap', '1000']),
    ('cutoff_low', ['--cutoff-offset=-.5']),
    ('cutoff_high', ['--cutoff-offset=.5']),
    ('alpha05', ['--alpha', '.05']),
    ('alpha20', ['--alpha', '.2']),
    ('protect_hits', ['--protect-hits']),
]
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
for name, flags in settings:
    subprocess.run([
        sys.executable, str(BASE / 'titraselect.py'), '--data', str(args.data.resolve()),
        '--out', str(BASE / 'sensitivity' / name), '--seeds', '1720', '1721', '1722',
        '--no-predictions', *flags,
    ], check=True, env=env)
    print('Completed', name, flush=True)
