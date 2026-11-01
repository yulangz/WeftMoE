from pathlib import Path
import subprocess,sys,json,time
base=Path('build/fp8-profile-opt');exp=Path('benchmarks/experiments/fp8_profile_opt');records=[]
with (base/'swiglu-unit.log').open('w') as log:subprocess.run([str(base/'swiglu_check')],stdout=log,stderr=subprocess.STDOUT,check=True)
for label in ['packed','combined']:
 print('CHECK',label,flush=True);start=time.time()
 subprocess.run([sys.executable,str(exp/'check.py'),label],check=True)
 records.append(dict(label=label,elapsed_s=time.time()-start,all_models_pass=True))
 (base/'followup-checks.json').write_text(json.dumps(records,indent=2)+'\n')
subprocess.run([sys.executable,str(exp/'benchmark.py'),'--native-root',str(base),'--labels','baseline','packed','combined','bf16','baseline_after','--mode','cycle','--warmups','10','--samples','30','--output',str(base/'confirm')],check=True)
