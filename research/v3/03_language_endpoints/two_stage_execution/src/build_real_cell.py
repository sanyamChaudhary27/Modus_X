from pathlib import Path
root=Path(__file__).parent
source=(root/'build_cell.py').read_text()
source=source.replace("'stress_probe.py']", "'stress_probe.py','real_data_gate.py','diagnose_drift.py','tpu_lm_train.py','enwik8_data.py']")
source=source.replace("'run_probe')", "'run_probe','diagnose_drift','tpu_lm_train','enwik8_data')")
source=source.replace("str(ROOT/'stress_probe.py')", "str(ROOT/'real_data_gate.py')")
source=source.replace(",'--full'", "")
source=source.replace("'__pycache__' not in source.parts", "'__pycache__' not in source.parts and source.suffix != '.pkl'")
source=source.replace('KAGGLE_CELL.py','KAGGLE_REAL_DATA_CELL.py')
source=source.replace('# Fresh Kaggle TPU notebook. Hardware screen only, no training or dataset needed.',
    '# TPU: keep trained 47M SegmentRetention checkpoint + config.json attached. Enable Internet for verified enwik8 download.')
exec(compile(source,str(root/'build_cell.py'),'exec'),{'__file__':str(root/'build_cell.py')})
