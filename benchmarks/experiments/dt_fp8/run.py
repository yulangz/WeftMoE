"""Reproduce DT FP8 validation/benchmarks with immutable per-profile builds."""
import os, pathlib, subprocess, sys
root=pathlib.Path.cwd();base=root/'build/dt-fp8';label=sys.argv[1];mode=sys.argv[2]
for model,h,i,k in [('qwen35b',2048,512,8),('qwen122b',3072,1024,8),('flash',4096,2048,6)]:
 if model not in os.getenv('BUILD_MODELS','qwen35b qwen122b flash').split():continue
 native=base/f'{label}-{model}'
 env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',NCCL_DEBUG='WARN',OMP_NUM_THREADS='1',WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64',TEST_HIDDEN=str(h),TEST_INTERMEDIATE=str(i),BENCH_TOPK=str(k))
 env.pop('WEFTMOE_AUTOTUNE_TABLE',None);env.pop('WEFTMOE_STATIC_RING_TOKENS',None)
 test='tests/torchrun_dt_fp8.py' if mode=='check' else 'benchmarks/experiments/dt_fp8/benchmark.py'
 print('START',label,model,mode,flush=True)
 with (base/f'{label}-{model}-{mode}{os.getenv("RUN_SUFFIX","")}.log').open('w') as log:
  subprocess.run(['timeout','-k','10s','300s',sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',test],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',label,model,mode,flush=True)
