"""Validate package integrity and evidence arithmetic without accelerator work."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    count = 0
    for p in ROOT.rglob('*.py'):
        try:
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
            count += 1
        except (SyntaxError, UnicodeError) as error:
            failures.append(str(error))
    source = json.loads((ROOT / 'SOURCE_PROVENANCE.json').read_text())
    for name, expected in source['snapshot_files_sha256'].items():
        p = ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            failures.append('Pinned source changed: ' + name)
    result = json.loads((ROOT / 'research/v3/05_local_kv_filtering/RESULT.json').read_text())
    aggregate = json.loads((ROOT / 'docs/AGGREGATE.json').read_text())
    gains = [r['validation_control'] - r['validation_candidate'] for r in result['seeds']]
    if abs(sum(gains)/3 - aggregate['mean_validation_gain']) > 1e-14:
        failures.append('Aggregate mismatch')
    if not all(r['audit_pass'] for r in result['seeds']):
        failures.append('Missing reported audit pass')
    for p in (ROOT / 'README.md', ROOT / 'docs/EVIDENCE_MAP.md', ROOT / 'REPRODUCE.md'):
        if not p.is_file(): failures.append('Missing guide: ' + str(p))
    manifest = ROOT / 'MANIFEST.sha256'
    if manifest.exists():
        names = set()
        for line in manifest.read_text().splitlines():
            expected, name = line.split('  ', 1); names.add(name)
            p = ROOT / name
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                failures.append('Manifest mismatch: ' + name)
        actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
                  and p != manifest and '__pycache__' not in p.parts
                  and p.suffix not in ('.aux', '.log', '.out', '.synctex')}
        if names != actual: failures.append('Manifest file inventory differs')
    if failures: raise RuntimeError('\n'.join(failures))
    print('RELEASE_VALIDATION_PASS', json.dumps({'parsed_python_files': count,
          'pinned_source_files': len(source['snapshot_files_sha256']),
          'validation_wins': sum(g > 0 for g in gains),
          'checkpoint_payloads_verified': False, 'manifest_present': manifest.exists()}))


if __name__ == '__main__': main()
