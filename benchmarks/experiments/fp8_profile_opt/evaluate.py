from pathlib import Path
import subprocess,sys,json,time,os
base=Path('build/fp8-profile-opt');exp=Path('benchmarks/experiments/fp8_profile_opt');labels=sys.argv[1:];records=[];accepted=[]
for label in labels:
 if label=='decode':
  with (base/'decode-unit.log').open('w') as log:
   p=subprocess.run([str(base/'decode_check')],stdout=log,stderr=subprocess.STDOUT)
   if p.returncode==0:p=subprocess.run([str(base/'decode_column_check')],stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:
   records.append(dict(label=label,returncode=p.returncode,stage='decode_unit'));continue
 start=time.time();print('CHECK',label,flush=True)
 p=subprocess.run([sys.executable,str(exp/'check.py'),label])
 records.append(dict(label=label,returncode=p.returncode,stage='full_ep8_all_models',elapsed_s=time.time()-start))
 (base/'checks.json').write_text(json.dumps(records,indent=2)+'\n')
 if p.returncode==0:accepted.append(label)
print('CHECKED',accepted,flush=True)
for model in ['qwen35b','qwen122b','flash']:
 p=base/f'baseline_after-{model}'
 if not p.exists():p.symlink_to((base/f'baseline-{model}').resolve(),target_is_directory=True)
subprocess.run([sys.executable,str(exp/'benchmark.py'),'--native-root',str(base),'--labels','baseline',*accepted,'bf16','baseline_after','--mode','cycle','--warmups','5','--samples','10','--output',str(base/'screen')],check=True)
