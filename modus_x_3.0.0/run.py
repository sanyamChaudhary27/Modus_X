"""Explicit entry points into the frozen v3 research runners."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('smoke', 'seed3', 'audit'))
    args, remaining = parser.parse_known_args()
    source = Path(__file__).resolve().parent/'research/v3/05_local_kv_filtering/src'
    sys.path.insert(0, str(source))
    if args.task == 'smoke':
        if remaining: parser.error('smoke takes no additional arguments')
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        tests = unittest.defaultTestLoader.loadTestsFromNames(
            ['test_endpoint_recovery', 'test_endpoint_audit', 'test_local_kv'])
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    runner = 'run_seed3_frozen_endpoint.py' if args.task == 'seed3' else 'audit_endpoints.py'
    sys.argv = [runner] + (['--seed', '3'] if args.task == 'seed3' else []) + remaining
    runpy.run_path(str(source/runner), run_name='__main__')


if __name__ == '__main__': main()
