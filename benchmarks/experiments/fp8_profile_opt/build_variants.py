"""Build independent variants from the saved fixed224 baseline, restoring source on failure."""
from pathlib import Path
import sys,subprocess,json
base=Path('build/fp8-profile-opt');exp=Path('benchmarks/experiments/fp8_profile_opt')
variants=json.loads((exp/'variants.json').read_text())
for label in sys.argv[1:]:
 saved={str(p.relative_to(base/'source')):p.read_bytes() for p in (base/'source').rglob('*') if p.is_file()}
 for p,data in saved.items():assert Path(p).read_bytes()==data, f'Unrestored or independently edited source: {p}'
 try:
  patch=exp/f'{label}.patch'
  if patch.exists():subprocess.run(['git','apply',str(patch)],check=True)
  subprocess.run([sys.executable,str(base/'build.py'),label,'-Xptxas=-v',*variants[label]],check=True)
 finally:
  for p,data in saved.items():Path(p).write_bytes(data)
