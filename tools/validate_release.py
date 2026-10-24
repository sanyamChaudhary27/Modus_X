"""Validate the canonical v3 package from the repository root."""
from pathlib import Path
import runpy


if __name__ == '__main__':
    script = Path(__file__).resolve().parents[1]/'modus_x_3.0.0/tools/validate_release.py'
    runpy.run_path(str(script), run_name='__main__')
