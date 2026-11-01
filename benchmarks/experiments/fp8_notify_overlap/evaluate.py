"""Validate all overlap variants, then measure without concurrent compilation."""
from pathlib import Path
import subprocess,sys,os,json,time
base=Path('build/fp8-notify-overlap');exp=Path('benchmarks/experiments/fp8_notify_overlap');checks=[]
for label in ['both','delay','fwd','bwd']:
 env=dict(os.environ)
 if label=='delay':env['BUILD_MODELS']='qwen35b'
 print('CHECK',label,flush=True);start=time.time()
 subprocess.run([sys.executable,str(exp/'check.py'),label],env=env,check=True)
 checks.append(dict(label=label,passed=True,models=['qwen35b'] if label=='delay' else ['qwen35b','qwen122b','flash'],elapsed_s=time.time()-start))
 (base/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
subprocess.run([sys.executable,str(exp/'benchmark.py'),'--native-root',str(base),'--labels','baseline','fwd','bwd','both','baseline_after','--mode','cycle','--warmups','10','--samples','30','--output',str(base/'cycle')],check=True)
