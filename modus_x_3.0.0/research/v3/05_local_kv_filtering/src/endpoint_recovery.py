"""Verify complete paired checkpoints before loading trusted experiment pickles."""
import hashlib
import json
import pickle
import shutil
import zipfile
from pathlib import Path

CASES=('control','local_kv')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def inspect_pair(root,seed):
    root=Path(root)
    meta=root/'pair_manifest.json'
    legacy=root/'screen_result.json'
    if meta.exists():
        info=json.loads(meta.read_text()); actual_seed=info['seed']
    elif legacy.exists():
        info=json.loads(legacy.read_text()); actual_seed=info['seed']
    else: raise ValueError('Missing seed provenance')
    if actual_seed!=seed: raise ValueError('Wrong seed')
    records={}
    for case in CASES:
        path=root/(case+'_checkpoint.pkl')
        rec=json.loads((root/(case+'_recovery.json')).read_text())
        if not path.is_file() or path.stat().st_size!=rec['bytes']: raise ValueError('Checkpoint missing or truncated')
        if sha(path)!=rec['sha256']: raise ValueError('Checkpoint hash mismatch')
        records[case]=rec
    step=records['control']['step']
    if records['local_kv']['step']!=step: raise ValueError('Mixed checkpoint steps')
    if not 1000<=step<=25000: raise ValueError('Unexpected checkpoint step')
    if meta.exists() and (info['step']!=step or info['records']!=records): raise ValueError('Pair commit mismatch')
    return step,tuple(records[c]['sha256'] for c in CASES)

def discover(seed,work,roots=('/kaggle/input',)):
    candidates=[]; rejected=[]
    for root in map(Path,roots):
        for p in root.rglob('control_checkpoint.pkl'):
            try:
                step,fp=inspect_pair(p.parent,seed)
                if step>=5000: candidates.append((step,fp,p.parent))
            except (ValueError,KeyError,OSError) as e: rejected.append((str(p.parent),str(e)))
    # Notebook outputs may expose both the extracted pair and its ZIP. Hashes deduplicate them.
    if not candidates:
        for root in map(Path,roots):
            for path in root.rglob('*.zip'):
                try:
                    with zipfile.ZipFile(path) as z:
                        names={n.replace('\\','/'):n for n in z.namelist()}
                        for name in names:
                            if not name.endswith('control_checkpoint.pkl'): continue
                            prefix=name[:-len('control_checkpoint.pkl')]
                            required=[c+s for c in CASES for s in ('_checkpoint.pkl','_recovery.json')]
                            metadata=next((m for m in ('pair_manifest.json','screen_result.json') if prefix+m in names),None)
                            if metadata is None or not all(prefix+n in names for n in required): continue
                            dest=Path(work)/('restore_'+hashlib.sha256((str(path)+prefix).encode()).hexdigest()[:12])
                            dest.mkdir(parents=True,exist_ok=True)
                            for n in required+[metadata]:
                                with z.open(names[prefix+n]) as src,(dest/n).open('wb') as dst: shutil.copyfileobj(src,dst)
                            step,fp=inspect_pair(dest,seed)
                            if step>=5000: candidates.append((step,fp,dest))
                except (ValueError,KeyError,OSError,zipfile.BadZipFile) as e: rejected.append((str(path),str(e)))
    if not candidates: raise RuntimeError('No complete matching pair. Attach full notebook output with BOTH checkpoint.pkl files, recovery JSONs and screen_result.json. Compact ZIP alone has no checkpoints. Rejections: '+repr(rejected[:8]))
    highest=max(c[0] for c in candidates)
    unique={fp:path for step,fp,path in candidates if step==highest}
    if len(unique)!=1: raise RuntimeError('Conflicting checkpoint pairs at the same step; attach only the intended run: '+repr(list(map(str,unique.values()))))
    source=next(iter(unique.values()))
    print('LOCAL_KV_RESUME_SOURCE',seed,highest,str(source),flush=True)
    return source,highest

def load_pair(root):
    # Use only the user's own checkpoints. Pickle must not be used with untrusted uploads.
    result={}
    for case in CASES:
        with (Path(root)/(case+'_checkpoint.pkl')).open('rb') as f: result[case]=pickle.load(f)
        v=result[case]
        if v['case']!=case or v['experiment']!='local_kv_screen': raise ValueError('Wrong checkpoint architecture')
    if result['control']['step']!=result['local_kv']['step']: raise ValueError('Steps differ')
    if json.dumps(result['control']['rng_state'],sort_keys=True)!=json.dumps(result['local_kv']['rng_state'],sort_keys=True): raise ValueError('Data RNG states differ')
    return result

def save_pair(out,seed,step,states,rng,cfg):
    import jax
    root=Path(out)/'checkpoints'/('step_'+str(step))
    root.mkdir(parents=True,exist_ok=True); records={}
    for case,(p,s) in states.items():
        target=root/(case+'_checkpoint.pkl'); partial=target.with_suffix('.partial')
        with partial.open('wb') as f:
            pickle.dump({'params':jax.device_get(p),'opt_state':jax.device_get(s),'rng_state':rng.bit_generator.state,
                'step':step,'experiment':'local_kv_screen','case':case,'config':vars(cfg),'seed':seed},f,protocol=pickle.HIGHEST_PROTOCOL)
        partial.replace(target)
        records[case]={'file':target.name,'step':step,'bytes':target.stat().st_size,'sha256':sha(target)}
        (root/(case+'_recovery.json')).write_text(json.dumps(records[case],indent=2))
    manifest=root/'pair_manifest.json'; partial=manifest.with_suffix('.partial')
    partial.write_text(json.dumps({'seed':seed,'step':step,'records':records},indent=2)); partial.replace(manifest)
    inspect_pair(root,seed)
    print('RECOVERY_VERIFIED',json.dumps({'seed':seed,'step':step,'directory':str(root),'files':records}),flush=True)
    # Keep two completed generations. Never touch attached inputs or unrelated directories.
    complete=sorted((p for p in root.parent.glob('step_*') if (p/'pair_manifest.json').exists()),key=lambda p:int(p.name[5:]))
    for old in complete[:-2]:
        if old.resolve().parent!=root.parent.resolve(): raise RuntimeError('Unsafe cleanup target')
        shutil.rmtree(old)
    return root
