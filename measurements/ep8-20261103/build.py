import os, subprocess, pathlib, json
root=pathlib.Path(__file__).resolve().parents[2]
os.chdir(root)
for model in ['qwen122b','qwen35b','flash']:
 out=root/'build/perf_remeasure_20261103/native'/model
 out.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,NVCCFLAGS=f'-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I{root}/configs/profiles/{model}')
 with (out/'build.log').open('w') as log:
  subprocess.run(['/opt/.venv/bin/python','build_native.py','lib','--output',str(out/'libweftmoe.so'),'--jobs','4'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['/opt/.venv/bin/python','tools/build_tensor_runtime.py','--library',str(out/'libweftmoe.so')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BUILT',model,flush=True)
