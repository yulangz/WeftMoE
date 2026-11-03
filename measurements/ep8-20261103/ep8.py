import os, sys, subprocess, pathlib, json, hashlib, time, signal
root=pathlib.Path(__file__).resolve().parents[2]
os.chdir(root)
out=root/'build/perf_remeasure_20261103/ep8'
out.mkdir(parents=True,exist_ok=True)
models={'qwen122b':(3072,1024,8),'qwen35b':(2048,512,8),'flash':(4096,2048,6)}
base={k:v for k,v in os.environ.items() if not k.startswith(('WEFTMOE_', 'TEST_', 'BENCH_'))}
base.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='1', NCCL_DEBUG='WARN', PYTHONPATH=str(root),LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64')
launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8']
manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'world':8,'order':['bf16','fp8','fp8','bf16'],'seed':123,'warmups':10,'samples':30,'priming':'one complete trace','ring_tokens':512,'ring_depth':8,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader'],text=True),'native':{},'runner_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
for model in models:
 native=root/'build/perf_remeasure_20261103/native'/model
 manifest['native'][model]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in native.glob('*.so')}
 assert len(manifest['native'][model])==2,model
manifest['environment']={k:base.get(k) for k in ['CUDA_LAUNCH_BLOCKING','CUDA_DEVICE_MAX_CONNECTIONS','NCCL_IB_DISABLE','PYTORCH_NO_CUDA_MEMORY_CACHING','TORCH_NCCL_BLOCKING_WAIT','OMP_NUM_THREADS']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
records=[]
workloads={}
def run(key, command, env, prefix, expected):
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if busy:raise RuntimeError('GPU processes present before job '+key+': '+busy)
 path=out/(key+'.log'); telemetry=out/(key+'.gpu.csv')
 print('START',key,time.strftime('%H:%M:%S'),flush=True)
 t=time.time()
 with telemetry.open('w') as monitor, path.open('w') as log:
  probe=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used','--format=csv,noheader,nounits','--loop-ms=500'],stdout=monitor,stderr=subprocess.DEVNULL)
  p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   status=p.wait(timeout=600)
   if status:raise RuntimeError(f'{key} exit {status}')
  finally:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:
     os.killpg(p.pid,signal.SIGKILL);p.wait()
   probe.terminate();probe.wait()
 rows=[json.loads(line[len(prefix):]) for line in path.read_text().splitlines() if line.startswith(prefix)]
 if len(rows)!=expected:raise RuntimeError('Incomplete output '+key)
 record={'key':key,'seconds':time.time()-t,'command':command[len(launcher):], 'results':rows}
 records.append(record)
 (out/'results.json').write_text(json.dumps(records,indent=2))
 for row in rows:
  if prefix.startswith('TRAINING'):
   k=(model,row['nominal_tokens_per_rank'])
   hashes=row['workload_sha256_per_rank']
   if k in workloads:assert workloads[k]==hashes, k
   workloads[k]=hashes
   print('RESULT',key,k[1],round(row['wall_slowest_rank_p50_us']/1000,4),'ms',flush=True)
  else:print('RESULT',key,row['tokens'],{k:round(row[k],2) for k in ('bf16_us','fp8_online_us','fp8_prequant_us')},flush=True)
 print('END',key,round(time.time()-t,1),'s',flush=True)
for model,(h,i,k) in models.items():
 native=root/'build/perf_remeasure_20261103/native'/model
 env=base|{'WEFTMOE_TENSOR_EXTENSION':str(next(native.glob('_runtime*.so'))),'TEST_HIDDEN':str(h),'TEST_INTERMEDIATE':str(i),'TEST_EXPERTS':'256','TEST_TOKENS':'65','TEST_FP8_SOURCE_X':'1'}
 if model != 'flash':
  print('CHECK_ALREADY_PASSED',model,flush=True)
  continue
 print('CHECK',model,flush=True)
 with (out/(model+'-correctness.log')).open('w') as log:
  subprocess.run(launcher+['tests/torchrun_fp8.py'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
 print('CHECK_PASS',model,flush=True)
for mode in ['cycle','forward','backward']:
 for model in models:
  native=root/'build/perf_remeasure_20261103/native'/model
  env=base|{'WEFTMOE_TENSOR_EXTENSION':str(next(native.glob('_runtime*.so')))}
  for repeat,dtype in enumerate(manifest['order']):
   cmd=launcher+['benchmarks/torchrun_training.py','--backend','pf','--model',model,'--tokens','8192','20480','--warmups','10','--samples','30','--stream-ordered','--prime-trace','--ring-tokens','512','--ring-depth','8','--seed','123']
   if dtype=='fp8':cmd+=['--fp8','--weight-grad-dtype','bf16']
   if mode!='cycle':cmd+=['--'+mode+'-only']
   run(f'{mode}-{model}-{repeat}-{dtype}',cmd,env,'TRAINING_CYCLE_JSON ',2)
for model,(h,i,k) in models.items():
 native=root/'build/perf_remeasure_20261103/native'/model
 env=base|{'WEFTMOE_TENSOR_EXTENSION':str(next(native.glob('_runtime*.so'))),'TEST_HIDDEN':str(h),'TEST_INTERMEDIATE':str(i),'BENCH_TOPK':str(k),'BENCH_EXPERTS':'256','BENCH_SIZES':'64 512','BENCH_REPEATS':'100','BENCH_SWEEPS':'11'}
 for seed in [1300,1700]:
  run(f'dt-{model}-{seed}',launcher+['benchmarks/benchmark_dt_fp8.py'],env|{'BENCH_SEED':str(seed)},'RESULT ',2)
print('COMPLETE',flush=True)
