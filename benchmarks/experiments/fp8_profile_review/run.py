"""Profile immutable fixed224 builds; no experimental row-col GEMM kernels."""
import os,sys,pathlib,subprocess,time,json
root=pathlib.Path.cwd();base=root/'build/fp8-profile-review';mode=sys.argv[1]
models=os.environ.get('PROFILE_MODELS','qwen35b qwen122b flash').split()
manifest=[]
for model in models:
 for dtype in ['fp8','bf16']:
  native=base/f'{"baseline" if mode=="production" else "full"}-{model}'
  out=base/mode/f'{model}-{dtype}';out.mkdir(parents=True,exist_ok=False)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',NCCL_DEBUG='WARN',TEST_FP8_COMPACT='1',WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64')
  env.pop('WEFTMOE_AUTOTUNE_TABLE',None)
  if mode!='forward':env['WEFTMOE_BACKWARD_PROGRESS']='1'
  command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','--log-dir',str(out/'ranks'),'--redirects','3',str(base/'training_timeline.py') if mode=='service' else 'benchmarks/torchrun_training.py','--backend','pf','--model',model,'--tokens',*(['20480'] if mode=='forward' else ['8192','20480']),'--warmups','3','--samples','1' if mode=='forward' else '5','--report-communication','--report-window-padding']
  if mode=='production':command+=['--stream-ordered','--prime-trace','--cuda-trace-dir',str(out/'cuda')]
  elif mode=='service':command+=['--stream-ordered','--prime-trace','--backward-only','--timeline-dir',str(out/'timeline')]
  else:command+=['--forward-only','--timeline-dir',str(out/'timeline'),'--profile-stages']
  if dtype=='fp8':command+=['--fp8','--weight-grad-dtype','bf16']
  print('START',mode,model,dtype,flush=True);start=time.time()
  with (out/'launch.log').open('w') as log:
   p=subprocess.run(['timeout','--kill-after=10s','240s',*command],env=env,stdout=log,stderr=subprocess.STDOUT)
  record=dict(model=model,dtype=dtype,mode=mode,command=command,native=str(native),elapsed_s=time.time()-start,returncode=p.returncode)
  manifest.append(record);(base/f'{mode}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  if p.returncode:raise RuntimeError(record)
  print('DONE',mode,model,dtype,round(time.time()-start,1),flush=True)
