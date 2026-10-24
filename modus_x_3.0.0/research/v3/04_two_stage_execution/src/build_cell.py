"""Create an embedded-source Kaggle TPU hardware-screen cell."""
import base64
import hashlib
import io
from pathlib import Path
import zipfile

root=Path(__file__).parent
files=['models.py','two_stage.py','segment_scale_trainer.py','run_probe.py','stress_probe.py']
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        archive.writestr(name,(root/name).read_bytes())
payload=buffer.getvalue()
digest=hashlib.sha256(payload).hexdigest()
cell='''# Fresh Kaggle TPU notebook. Hardware screen only, no training or dataset needed.
import base64, hashlib, pathlib, runpy, sys, zipfile, io
payload=base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest()==DIGEST
ROOT=pathlib.Path('/kaggle/working/mx_two_stage_') / DIGEST[:12]
ROOT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for item in archive.infolist():
        if '/' in item.filename or '\\\\' in item.filename or item.filename in ('.','..'):
            raise ValueError('Unexpected package path')
        (ROOT/item.filename).write_bytes(archive.read(item))
import jax
jax.config.update('jax_default_matmul_precision','highest')
jax.clear_caches()
if jax.default_backend()!='tpu' or jax.device_count()!=8:
    raise RuntimeError('Select TPU in notebook accelerator settings, then restart the session.')
print('TWO_STAGE_TPU_READY',jax.__version__,jax.devices(),flush=True)
OUT=ROOT/'output'
OUT.mkdir(exist_ok=True)
old_argv=sys.argv[:]
old_path=sys.path[:]
names=('models','two_stage','segment_scale_trainer','run_probe')
saved={name:sys.modules.pop(name) for name in names if name in sys.modules}
try:
    sys.path.insert(0,str(ROOT))
    sys.argv=[str(ROOT/'stress_probe.py'),'--outdir',str(OUT),'--full']
    # Use this process, so a second Python process does not compete for TPU ownership.
    runpy.run_path(str(ROOT/'stress_probe.py'),run_name='__main__')
finally:
    sys.argv=old_argv
    sys.path[:]=old_path
    for name in names:sys.modules.pop(name,None)
    sys.modules.update(saved)
    path=ROOT.parent/('mx_two_stage_'+DIGEST[:12]+'_results.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for source in ROOT.rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                archive.write(source,source.relative_to(ROOT).as_posix())
    print('TWO_STAGE_RESULTS_ARCHIVE',path,flush=True)
'''.replace('PAYLOAD',repr(base64.b64encode(payload).decode())).replace('DIGEST',repr(digest))
compile(cell,'KAGGLE_CELL.py','exec')
(root/'KAGGLE_CELL.py').write_text(cell,encoding='utf-8')
print('CELL_READY',root/'KAGGLE_CELL.py',digest)
