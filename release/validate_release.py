"""Compatibility entry point for v3 release validation."""
from pathlib import Path
import argparse
import runpy


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final',action='store_true')
    parser.parse_args()
    script=Path(__file__).resolve().parents[1]/'modus_x_3.0.0/tools/validate_release.py'
    runpy.run_path(str(script),run_name='__main__')
