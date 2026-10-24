"""Compile the canonical v3 manuscript with an installed Tectonic binary."""
from pathlib import Path
import os
import shutil
import subprocess


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    source=root/'modus_x_3.0.0/paper/manuscript'
    compiler=os.environ.get('TECTONIC_BIN') or shutil.which('tectonic')
    if not compiler:
        raise SystemExit('Install Tectonic or set TECTONIC_BIN to its executable. The release already includes the compiled PDF.')
    subprocess.run([compiler,'-X','compile','--untrusted','main.tex'],cwd=source,check=True)
    target=root/'modus_x_3.0.0/paper/Modus_X_3.0.0_paper.pdf'
    shutil.copy2(source/'main.pdf',target)
    shutil.copy2(target,root/'paper'/target.name)
    print('V3_PAPER_COMPILED',target,flush=True)
