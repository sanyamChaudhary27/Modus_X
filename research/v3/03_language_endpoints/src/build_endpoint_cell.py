from __future__ import annotations

import base64
import hashlib
import io
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = (
    (ROOT / "segment_scale_trainer.py", "segment_scale_trainer.py"),
    (ROOT / "audit_dense_checkpoint.py", "audit_dense_checkpoint.py"),
    (ROOT / "run_source_trace.py", "run_source_trace.py"),
    (ROOT / "run_natural_delayed_recall.py", "run_natural_delayed_recall.py"),
    (ROOT / "run_contiguous_training_screen.py", "run_contiguous_training_screen.py"),
    (ROOT / "models.py", "models.py"),
    (ROOT / "tpu_lm_train.py", "tpu_lm_train.py"),
)


def package_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, destination in FILES:
            info = zipfile.ZipInfo(destination, date_time=(2026, 8, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    return buffer.getvalue()


def build_cell():
    package = package_bytes()
    payload = base64.b64encode(package).decode("ascii")
    digest = hashlib.sha256(package).hexdigest()
    return textwrap.dedent(f'''\
import base64, hashlib, json, pathlib, pickle, shutil, subprocess, sys, urllib.request, zipfile

# Promoted segment-retention candidate: seed-1 continuation to 102.4M chars.
# Attach the committed seed-1 20.48M segment-retention notebook as input.
ROOT = pathlib.Path('/kaggle/working/segment_retention_endpoint_pkg')
ARCHIVE = pathlib.Path('/kaggle/working/segment_retention_endpoint_pkg.zip')
OUT = pathlib.Path('/kaggle/working/segment_retention_seed1_102p4m')
ARCHIVE.write_bytes(base64.b64decode({payload!r}))
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == {digest!r}
if ROOT.exists(): shutil.rmtree(ROOT)
if OUT.exists(): shutil.rmtree(OUT)
ROOT.mkdir(parents=True); OUT.mkdir(parents=True)
with zipfile.ZipFile(ARCHIVE) as package:
    for member in package.infolist():
        member.filename = member.filename.replace('\\\\', '/').lstrip('./')
        package.extract(member, ROOT)

def valid_seed1_checkpoint(checkpoint):
    try:
        progress_path = checkpoint.with_name('progress.json')
        config_path = checkpoint.with_name('config.json')
        if not (progress_path.is_file() and config_path.is_file()): return None
        progress = json.loads(progress_path.read_text())
        config = json.loads(config_path.read_text())
        rows = progress.get('rows', [])
        args = config.get('args', config.get('config', {{}}).get('args', {{}}))
        params = int(config.get('params', progress.get('config', {{}}).get('params', -1)))
        if not rows or int(rows[-1].get('step', -1)) != 5000: return None
        if int(rows[-1].get('processed_characters', -1)) != 20_480_000: return None
        if params != 47_437_768 or int(args.get('seed', -1)) != 1: return None
        if args.get('model') != 'Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision': return None
        return (checkpoint.stat().st_size, float(rows[-1]['val_bpc']))
    except Exception:
        return None

candidates = []
for search_root in (pathlib.Path('/kaggle/input'), pathlib.Path('/kaggle/working')):
    if not search_root.exists(): continue
    for checkpoint in search_root.rglob('checkpoint.pkl'):
        signature = valid_seed1_checkpoint(checkpoint)
        if signature is not None: candidates.append((checkpoint.resolve(), signature))
if not candidates:
    raise RuntimeError('No exact seed-1 step-5000 segment-retention checkpoint found. Attach its committed Kaggle notebook output.')
signatures = {{signature for _, signature in candidates}}
if len(signatures) != 1:
    raise RuntimeError(f'Multiple distinct valid seed-1 checkpoints found: {{candidates}}')
source = sorted((path for path, _ in candidates), key=lambda p: (str(p).startswith('/kaggle/working'), len(str(p)), str(p)))[0]
print('SEGMENT_RETENTION_ENDPOINT_SOURCE', source, flush=True)
for name in ('checkpoint.pkl', 'config.json', 'progress.json'):
    shutil.copy2(source.with_name(name), OUT/name)
with (OUT/'checkpoint.pkl').open('rb') as handle:
    restored = pickle.load(handle)
if int(restored.get('step', -1)) != 5000:
    raise RuntimeError('Restored checkpoint failed the step-5000 verification')
print('SEGMENT_RETENTION_ENDPOINT_RECOVERY_VERIFIED', json.dumps({{
    'step': restored['step'], 'checkpoint_bytes': (OUT/'checkpoint.pkl').stat().st_size,
    'source': str(source)
}}), flush=True)
del restored

data = None
for search_root in (pathlib.Path('/kaggle/working'), pathlib.Path('/kaggle/input')):
    if search_root.exists():
        for path in search_root.rglob('enwik8'):
            if path.is_file() and path.stat().st_size == 100_000_000:
                data = path; break
    if data: break
if data is None:
    zipped = pathlib.Path('/kaggle/working/enwik8.zip')
    urllib.request.urlretrieve('https://mattmahoney.net/dc/enwik8.zip', zipped)
    with zipfile.ZipFile(zipped) as source_zip: source_zip.extract('enwik8', '/kaggle/working')
    data = pathlib.Path('/kaggle/working/enwik8')
assert data.stat().st_size == 100_000_000

COMMON = [
    '--data-path', str(data), '--outdir', str(OUT),
    '--model', 'Modus_X_MemoryFeedbackSegmentRetention_DeepSupervision',
    '--batch', '8', '--target-chars', '102400000', '--checkpoint-chars', '4096000',
    '--eval-batch', '8', '--eval-chunks', '128', '--embed-dim', '512',
    '--hidden-dim', '1536', '--state-dim', '512', '--n-layers', '12',
    '--router-hidden', '32', '--optimizer', 'adamw', '--auxiliary-weight', '0.05',
    '--weight-decay', '0.0001', '--schedule', 'constant', '--precision', 'float32',
    '--input-seq-len', '512', '--loss-tail', '512', '--auxiliary-layers', '6',
    '--future-targets', '2', '--future-target-weight', '0.5', '--seed', '1', '--resume'
]
main_command = [sys.executable, '-u', str(ROOT/'segment_scale_trainer.py'), *COMMON,
                '--stop-chars', '81920000', '--lr', '0.0006']
print('RUN_SEGMENT_RETENTION_ENDPOINT_MAIN', ' '.join(main_command), flush=True)
subprocess.run(main_command, check=True, cwd=ROOT)
with (OUT/'checkpoint.pkl').open('rb') as handle: main_state = pickle.load(handle)
if int(main_state.get('step', -1)) != 20000: raise RuntimeError(f'Expected step 20000, found {{main_state.get("step")}}')
del main_state

endpoint_command = [sys.executable, '-u', str(ROOT/'segment_scale_trainer.py'), *COMMON,
                    '--stop-chars', '102400000', '--lr', '0.0003']
print('RUN_SEGMENT_RETENTION_ENDPOINT_ANNEAL', ' '.join(endpoint_command), flush=True)
subprocess.run(endpoint_command, check=True, cwd=ROOT)
with (OUT/'checkpoint.pkl').open('rb') as handle: endpoint_state = pickle.load(handle)
if int(endpoint_state.get('step', -1)) != 25000: raise RuntimeError(f'Expected step 25000, found {{endpoint_state.get("step")}}')
del endpoint_state

dense_out = OUT/'dense_audit'
dense_command = [sys.executable, '-u', str(ROOT/'audit_dense_checkpoint.py'),
    '--data-path', str(data), '--checkpoint', str(OUT/'checkpoint.pkl'),
    '--outdir', str(dense_out), '--eval-batch', '8', '--sample-windows', '1024']
print('RUN_SEGMENT_RETENTION_ENDPOINT_DENSE_AUDIT', ' '.join(dense_command), flush=True)
subprocess.run(dense_command, check=True, cwd=ROOT)

trace_out = OUT/'source_trace'
trace_command = [sys.executable, '-u', str(ROOT/'run_source_trace.py'),
    '--data-path', str(data), '--checkpoint-path', str(OUT/'checkpoint.pkl'),
    '--code-root', str(ROOT), '--outdir', str(trace_out), '--seed', '1',
    '--expected-step', '25000', '--segment-scale-archive']
print('RUN_SEGMENT_RETENTION_ENDPOINT_SOURCE_TRACE', ' '.join(trace_command), flush=True)
subprocess.run(trace_command, check=True, cwd=ROOT)

progress = json.loads((OUT/'progress.json').read_text())
dense = json.loads((dense_out/'evaluation_audit.json').read_text())
trace = json.loads((trace_out/'source_trace.json').read_text())
dense_validation = sum(dense['results']['validation'][name]['bpc'] for name in ('dense_offset_0','dense_offset_half')) / 2
dense_test = sum(dense['results']['test'][name]['bpc'] for name in ('dense_offset_0','dense_offset_half')) / 2
long_cells = [row for row in trace['summary']['cells'] if row['distance_band'] in ('long','very_long')]
archive_ratio = sum(row['target_to_post_source_delta_ratio'][1] for row in long_cells) / len(long_cells)
patch_gain = trace['summary']['aggregate_mean_patch_gain_bpc']['all']
control_dense_validation = 1.459723
validation_gain = control_dense_validation - dense_validation
runtime_ratio = float(progress['rows'][-1]['elapsed_s']) / 22570.22
checks = {{
    'dense_validation_regression_at_most_0p005': validation_gain >= -0.005,
    'archive_retention_improves_at_least_100x': archive_ratio >= 100 * 3.9614571743164456e-06,
    'source_patch_gain_at_least_0p0005': patch_gain >= 0.0005,
    'runtime_at_most_1p05x': runtime_ratio <= 1.05,
}}
decision = {{
    'stage': 'seed-1 matched 102.4M endpoint, dense audit, and frozen source trace',
    'candidate': {{'params': 47437768, 'last_sparse_checkpoint': progress['rows'][-1],
                  'dense_validation_bpc': dense_validation, 'dense_test_bpc_report_only': dense_test}},
    'frozen_control': {{'params': 47437768, 'dense_validation_bpc': control_dense_validation,
                       'dense_test_bpc_report_only': 1.465006, 'elapsed_s': 22570.22}},
    'control_minus_candidate_dense_validation_bpc': validation_gain,
    'strong_language_win': validation_gain >= 0.005,
    'mean_long_range_archive_delta_ratio': archive_ratio,
    'archive_retention_improvement_ratio_vs_canonical_screen': archive_ratio / 3.9614571743164456e-06,
    'aggregate_all_state_patch_gain_bpc': patch_gain,
    'runtime_ratio_vs_control': runtime_ratio,
    'promotion_checks': checks, 'endpoint_pass': all(checks.values()),
    'test_data_read': True, 'test_role': 'report_only_after_frozen_endpoint',
}}
decision['next'] = ('replicate the 102.4M endpoint on seeds 2 and 3' if decision['endpoint_pass']
                    else 'freeze segment-scale retention at mechanistic evidence only')
(OUT/'ENDPOINT_DECISION.json').write_text(json.dumps(decision, indent=2)+'\\n')
print('SEGMENT_RETENTION_ENDPOINT_DECISION', json.dumps(decision), flush=True)

compact = pathlib.Path('/kaggle/working/segment_retention_seed1_102p4m_compact.zip')
if compact.exists(): compact.unlink()
with zipfile.ZipFile(compact, 'w', zipfile.ZIP_DEFLATED) as destination:
    for path in OUT.rglob('*'):
        if path.is_file() and path.name != 'checkpoint.pkl': destination.write(path, path.relative_to(OUT))
print('SEGMENT_RETENTION_ENDPOINT_COMPACT_READY', compact, flush=True)
print('SEGMENT_RETENTION_ENDPOINT_CHECKPOINT', OUT/'checkpoint.pkl', flush=True)
print('SEGMENT_RETENTION_ENDPOINT_CELL_COMPLETE', flush=True)
''')


def main():
    cell = build_cell()
    generated = ROOT / "generated_cells" / "KAGGLE_TPU_SEGMENT_RETENTION_SEED1_102P4M.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(cell, encoding="utf-8", newline="\n")
    print("SEGMENT_RETENTION_ENDPOINT_PACKAGE_SHA256", hashlib.sha256(package_bytes()).hexdigest())
    print("SEGMENT_RETENTION_ENDPOINT_CELL", generated)


if __name__ == "__main__":
    main()
