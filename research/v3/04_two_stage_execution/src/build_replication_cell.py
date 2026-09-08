from pathlib import Path
root=Path(__file__).parent
source=(root/'build_cell.py').read_text()
source=source.replace("'stress_probe.py']", "'stress_probe.py','replication_gate.py','diagnose_drift.py','tpu_lm_train.py','enwik8_data.py']")
source=source.replace("'run_probe')", "'run_probe','diagnose_drift','tpu_lm_train','enwik8_data')")
source=source.replace("str(ROOT/'stress_probe.py')", "str(ROOT/'replication_gate.py')")
source=source.replace(",'--full'", "")
source=source.replace("'__pycache__' not in source.parts", "'__pycache__' not in source.parts and source.suffix != '.pkl'")
source=source.replace('KAGGLE_CELL.py','KAGGLE_REPLICATION_CELL.py')
source=source.replace('# Fresh Kaggle TPU notebook. Hardware screen only, no training or dataset needed.',
    '# Fresh TPU notebook: attach original SEED1 v3 SegmentRetention step25000 endpoint + config.json. Internet on. 2000 paired updates; no test evaluation.')
exec(compile(source,str(root/'build_cell.py'),'exec'),{'__file__':str(root/'build_cell.py')})
