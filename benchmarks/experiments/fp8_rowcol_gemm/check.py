import os,pathlib,subprocess,sys
base=pathlib.Path('build/fp8-rowcol-gemm').resolve(); label=sys.argv[1]
for model,h,i in [('qwen35b',2048,512),('qwen122b',3072,1024),('flash',4096,2048)]:
 if model not in os.environ.get('BUILD_MODELS','qwen35b qwen122b flash').split(): continue
 native=base/f'{label}-{model}'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64',TEST_HIDDEN=str(h),TEST_INTERMEDIATE=str(i),TEST_TOKENS='257',TEST_ZERO_RANK='1',TEST_ROW_SCALES='1',TEST_FP8_SOURCE_X='1',TEST_FP8_COMPACT='1',WEFTMOE_STATIC_RING_TOKENS='128')
 env.pop('WEFTMOE_AUTOTUNE_TABLE',None)
 with (base/f'{label}-{model}-check.log').open('w') as log:
  subprocess.run(['timeout','--kill-after=10s','150s',sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','tests/torchrun_fp8.py'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',label,model,flush=True)
