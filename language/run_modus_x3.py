"""Run the versioned Modus_X 3 checks or explicit frozen TPU runners."""
from pathlib import Path
import runpy


if __name__ == '__main__':
    entry = Path(__file__).resolve().parents[1]/'modus_x_3.0.0/run.py'
    runpy.run_path(str(entry), run_name='__main__')
