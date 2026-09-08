"""Acquire byte-identical enwik8 without text decoding or broad ZIP extraction."""
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

SIZE=100_000_000
SHA256='2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8'
URL='https://mattmahoney.net/dc/enwik8.zip'


def valid(path):
    if not path.is_file() or path.stat().st_size!=SIZE: return False
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()==SHA256


def extract(archive_path,target):
    partial=target.with_suffix('.extracting')
    try:
        with zipfile.ZipFile(archive_path) as z:
            members=[m for m in z.infolist() if m.filename.replace('\\','/').split('/')[-1]=='enwik8' and not m.is_dir()]
            if len(members)!=1 or members[0].file_size!=SIZE:
                raise ValueError('ZIP must contain exactly one correctly sized enwik8 member')
            with z.open(members[0]) as src, partial.open('wb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
        if not valid(partial): raise ValueError('enwik8 SHA-256 mismatch')
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def ensure(roots=('/kaggle/input','/kaggle/working'),work='/kaggle/working/mx_verified_data'):
    work=Path(work); work.mkdir(parents=True,exist_ok=True)
    target=work/'enwik8'
    if valid(target): return target
    for root in roots:
        for path in Path(root).rglob('enwik8'):
            if valid(path): return path
    for root in roots:
        for path in Path(root).rglob('enwik8.zip'):
            try: return extract(path,target)
            except (ValueError,OSError,zipfile.BadZipFile) as e:
                print('ENWIK8_ZIP_REJECTED',str(path),str(e),flush=True)
    download=work/'download.zip.partial'
    try:
        print('ENWIK8_DOWNLOAD',URL,flush=True)
        request=urllib.request.Request(URL,headers={'User-Agent':'Modus-X-reproducibility/1.0'})
        with urllib.request.urlopen(request,timeout=120) as src, download.open('wb') as dst:
            shutil.copyfileobj(src,dst,1024*1024)
        return extract(download,target)
    except Exception as e:
        raise RuntimeError('Could not obtain verified enwik8. Enable Internet in Kaggle settings and rerun, or attach enwik8.zip from '+URL+'. Checkpoints were not modified.') from e
    finally:
        download.unlink(missing_ok=True)
