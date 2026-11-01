from pathlib import Path
import subprocess,sys,os,json
base=Path('build/fp8-notify-overlap');exp=Path('benchmarks/experiments/fp8_notify_overlap')
checks=json.loads((base/'checks.json').read_text())
for label in ['ordered','ordered_fwd','ordered_half','ordered_delay']:
 print('CHECK',label,flush=True);env=dict(os.environ)
 if label=='ordered_delay':env['BUILD_MODELS']='qwen35b'
 subprocess.run([sys.executable,str(exp/'check.py'),label],env=env,check=True)
 checks.append(dict(label=label,passed=True,models=['qwen35b'] if label=='ordered_delay' else ['qwen35b','qwen122b','flash']));(base/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
subprocess.run([sys.executable,str(exp/'profile_ordered.py')],check=True)
subprocess.run([sys.executable,str(exp/'analyze_profile.py')],check=True)
subprocess.run([sys.executable,str(exp/'benchmark.py'),'--native-root',str(base),'--labels','baseline','ordered_fwd','ordered','ordered_half','baseline_after','--mode','cycle','--warmups','10','--samples','30','--output',str(base/'ordered-cycle')],check=True)
subprocess.run([sys.executable,str(exp/'summarize.py'),str(base/'ordered-cycle/results.json')],check=True)
