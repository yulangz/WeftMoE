"""Validate and benchmark the cleaned production implementation."""
from pathlib import Path
import subprocess,sys,os,json
base=Path('build/fp8-notify-overlap');exp=Path('benchmarks/experiments/fp8_notify_overlap')
subprocess.run([sys.executable,str(exp/'check.py'),'production'],check=True)
checks=json.loads((base/'checks.json').read_text());checks.append(dict(label='production',passed=True,models=['qwen35b','qwen122b','flash']));(base/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
subprocess.run([sys.executable,str(exp/'check.py'),'production'],env=dict(os.environ,BUILD_MODELS='qwen35b',TEST_FP8_COMPACT='0',CHECK_SUFFIX='-mixed'),check=True)
(base/'production-mixed-check.json').write_text(json.dumps(dict(label='production',model='qwen35b',passed=True,compact=False,log=str(base/'production-qwen35b-mixed-check.log')),indent=2)+'\n')
subprocess.run([sys.executable,str(exp/'benchmark.py'),'--native-root',str(base),'--labels','baseline','production','baseline_after','--mode','cycle','--warmups','10','--samples','30','--output',str(base/'production-cycle')],check=True)
subprocess.run([sys.executable,str(exp/'summarize.py'),str(base/'production-cycle/results.json')],check=True)
