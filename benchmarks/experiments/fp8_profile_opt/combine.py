"""Construct a reproducible combination from independent patches, without editing source."""
from pathlib import Path
import tempfile,subprocess,sys,difflib,json,shutil
base=Path('build/fp8-profile-opt');exp=Path('benchmarks/experiments/fp8_profile_opt');name,*labels=sys.argv[1:];v=json.loads((exp/'variants.json').read_text())
with tempfile.TemporaryDirectory(prefix='combine-',dir=base) as tmp:
 tree=Path(tmp).resolve()
 for p in (base/'source').rglob('*'):
  if p.is_file():dest=tree/p.relative_to(base/'source');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 flags=[]
 for label in labels:
  p=exp/f'{label}.patch'
  if p.exists():subprocess.run(['git','apply','--unsafe-paths',str(p.resolve())],cwd=tree,check=True)
  flags.extend(v[label])
 patch=''
 for p in (base/'source').rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(base/'source');before=p.read_text();after=(tree/rel).read_text()
  if before!=after:patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+str(rel),tofile='b/'+str(rel)))
 (exp/f'{name}.patch').write_text(patch);v[name]=flags;(exp/'variants.json').write_text(json.dumps(v,indent=2)+'\n')
 (base/f'{name}-components.json').write_text(json.dumps(dict(label=name,components=labels,flags=flags),indent=2)+'\n')
print(name,labels,flags)
