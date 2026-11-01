from pathlib import Path
import subprocess,sys,os,json,time
base=Path('build/fp8-notify-overlap');exp=Path('benchmarks/experiments/fp8_notify_overlap')
print('CHECK early',flush=True)
subprocess.run([sys.executable,str(exp/'check.py'),'early'],check=True)
checks=json.loads((base/'checks.json').read_text());checks.append(dict(label='early',passed=True,models=['qwen35b','qwen122b','flash']));(base/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('CHECK mixed BF16/FP8 pool',flush=True)
subprocess.run([sys.executable,str(exp/'check.py'),'early'],env=dict(os.environ,BUILD_MODELS='qwen35b',TEST_FP8_COMPACT='0',CHECK_SUFFIX='-mixed'),check=True)
(base/'mixed-check.json').write_text(json.dumps(dict(label='early',model='qwen35b',passed=True,compact=False,log=str(base/'early-qwen35b-mixed-check.log')),indent=2)+'\n')
subprocess.run([sys.executable,str(exp/'profile.py')],check=True)
subprocess.run([sys.executable,str(exp/'analyze_profile.py')],check=True)
subprocess.run([sys.executable,str(exp/'benchmark.py'),'--native-root',str(base),'--labels','baseline','early','baseline_after','--mode','cycle','--warmups','10','--samples','30','--output',str(base/'early-cycle')],check=True)
subprocess.run([sys.executable,str(exp/'summarize.py'),str(base/'early-cycle/results.json')],check=True)
