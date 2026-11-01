"""Capture normal, concurrent CUDA activities; do not use these runs as wall benchmarks."""
from pathlib import Path
import os,sys,subprocess,json,time
root=Path.cwd();base=root/'build/fp8-notify-overlap';manifest=[]
for model in ['qwen35b','qwen122b','flash']:
 for label in (['baseline','both','early'] if model=='qwen35b' else ['baseline','early']):
  native=base/f'{label}-{model}';out=base/'profile'/f'{label}-{model}';out.mkdir(parents=True,exist_ok=False)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',NCCL_DEBUG='WARN',TEST_FP8_COMPACT='1',WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64')
  env.pop('WEFTMOE_AUTOTUNE_TABLE',None)
  cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','benchmarks/torchrun_training.py','--backend','pf','--model',model,'--tokens','20480','--warmups','3','--samples','5','--stream-ordered','--prime-trace','--fp8','--weight-grad-dtype','bf16','--cuda-trace-dir',str(out/'cuda')]
  print('PROFILE',model,label,flush=True);start=time.time()
  with (out/'launch.log').open('w') as log:subprocess.run(['timeout','--kill-after=10s','240s',*cmd],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  manifest.append(dict(model=model,label=label,command=cmd,seconds=time.time()-start));(base/'profile-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
