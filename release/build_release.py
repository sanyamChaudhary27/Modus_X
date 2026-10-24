"""Rebuild the v3 source/evidence ZIP, standalone PDF and flat arXiv source."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'modus_x_3.0.0'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'dist')
    parser.add_argument('--final',action='store_true',help='Require the pinned final version identity')
    args=parser.parse_args()
    subprocess.run([sys.executable,str(PACKAGE/'tools/validate_release.py')],check=True)
    import json
    metadata=json.loads((PACKAGE/'.zenodo.json').read_text())
    if metadata['version']!='3.0.0' or metadata['doi']!='10.5281/zenodo.23621101':
        raise RuntimeError('Wrong v3 release identity')
    args.outdir.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.suffix not in ('.pyc','.aux','.log','.out','.synctex'))
    archive_path=args.outdir/'Modus_X_3.0.0_release.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in files:
            name='modus_x_3.0.0/'+p.relative_to(PACKAGE).as_posix()
            info=zipfile.ZipInfo(name,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;archive.writestr(info,p.read_bytes())
    shutil.copy2(PACKAGE/'paper/Modus_X_3.0.0_paper.pdf',args.outdir/'Modus_X_3.0.0_paper.pdf')
    with zipfile.ZipFile(args.outdir/'Modus_X_3.0.0_arxiv_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        text=(PACKAGE/'paper/manuscript/main.tex').read_text()
        archive.writestr('main.tex',text.replace('../numbers','numbers').replace('../endpoint_table','endpoint_table').replace('../figures/','figures/'))
        for name in ('numbers.tex','endpoint_table.tex'): archive.write(PACKAGE/'paper'/name,name)
        for p in sorted((PACKAGE/'paper/figures').glob('*.pdf')): archive.write(p,'figures/'+p.name)
    rows=[]
    for p in sorted(args.outdir.glob('Modus_X_3.0.0*')):
        if p.suffix not in ('.zip','.pdf'): continue
        if p.suffix=='.zip':
            with zipfile.ZipFile(p) as archive:
                if archive.testzip(): raise RuntimeError('Corrupt archive: '+str(p))
        rows.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
    (args.outdir/'ARTIFACTS_SHA256.txt').write_text('\n'.join(rows)+'\n')
    print('V3_RELEASE_ARTIFACTS_READY',args.outdir,flush=True)


if __name__=='__main__': main()
