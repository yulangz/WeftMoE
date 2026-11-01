"""Build independent source snapshots concurrently; never edit the active checkout."""
from pathlib import Path
import subprocess,sys,os,shutil,concurrent.futures,json,time
root=Path.cwd();base=root/'build/fp8-notify-overlap';exp=root/'benchmarks/experiments/fp8_notify_overlap';variants=json.loads((exp/'variants.json').read_text());labels=sys.argv[1:]
files=subprocess.check_output(['git','ls-files','-z']).decode().strip('\0').split('\0')
for label in labels:
 tree=base/'trees'/label;tree.mkdir(parents=True,exist_ok=False)
 for f in files:
  source=root/f
  if not source.is_file():continue
  target=tree/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 for source in (base/'source').rglob('*'):
  if source.is_file():shutil.copy2(source,tree/source.relative_to(base/'source'))
 patch=exp/f'{label}.patch'
 if patch.exists():subprocess.run(['git','apply','--unsafe-paths',str(patch)],cwd=tree,check=True)
def build(job):
 label,model=job;tree=base/'trees'/label;out=base/f'{label}-{model}';start=time.time();flags=f'-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I{tree}/configs/profiles/{model} -Xptxas=-v '+ ' '.join(variants[label]);env=dict(os.environ,NVCCFLAGS=flags)
 with (base/f'{label}-{model}-build.log').open('w') as log:
  p=subprocess.run([sys.executable,'build_native.py','lib','--output',str(out/'libweftmoe.so'),'--jobs','2'],cwd=tree,env=env,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode==0:
  with (base/f'{label}-{model}-adapter.log').open('w') as log:p=subprocess.run([sys.executable,'tools/build_tensor_runtime.py','--library',str(out/'libweftmoe.so')],cwd=tree,stdout=log,stderr=subprocess.STDOUT)
 r=dict(label=label,model=model,returncode=p.returncode,elapsed_s=time.time()-start,flags=flags,source_snapshot=str(tree));print('BUILT',label,model,p.returncode,round(r['elapsed_s'],1),flush=True);return r
jobs=[(l,m) for l in labels for m in (['qwen35b'] if l in ('delay','ordered_delay') else ['qwen35b','qwen122b','flash'])]
with concurrent.futures.ThreadPoolExecutor(max_workers=21) as pool:results=list(pool.map(build,jobs))
manifest=base/'isolated-builds.json'
previous=json.loads(manifest.read_text()) if manifest.exists() else []
manifest.write_text(json.dumps(previous+results,indent=2)+'\n')
if any(x['returncode'] for x in results):sys.exit(1)
