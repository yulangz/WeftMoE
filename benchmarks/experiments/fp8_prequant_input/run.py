from pathlib import Path
import subprocess,sys,os,json
base=Path("build/fp8-prequant-input");exp=Path("benchmarks/experiments/fp8_prequant_input")
subprocess.run([sys.executable,str(exp/"check.py"),"prequant"],check=True)
subprocess.run([sys.executable,str(exp/"check.py"),"prequant"],env=dict(os.environ,BUILD_MODELS="qwen35b",TEST_FP8_COMPACT="0",CHECK_SUFFIX="-mixed"),check=True)
for mode in ["cycle", "forward", "backward"]:
 subprocess.run([sys.executable,str(exp/"benchmark.py"),"--native-root",str(base),"--labels","baseline","prequant","baseline_after","--mode",mode,"--warmups","10","--samples","30","--output",str(base/mode)],check=True)
 subprocess.run([sys.executable,str(exp/"summarize.py"),str(base/mode/"results.json")],check=True)
