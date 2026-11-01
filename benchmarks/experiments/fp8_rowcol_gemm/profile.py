"""Separate diagnostic replay; never mix these timings with unprofiled benchmarks."""
import os, pathlib, subprocess, sys, json, statistics, re
root=pathlib.Path.cwd(); base=root/'build/fp8-rowcol-gemm'; output=base/'diagnostic'
output.mkdir(exist_ok=False)
summary=[]
for label in ('baseline','wide'):
 native=base/f'{label}-qwen122b'
 traces=output/label
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',NCCL_DEBUG='WARN',TEST_FP8_COMPACT='1',WEFTMOE_BACKWARD_PROGRESS='1',WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64')
 env.pop('WEFTMOE_AUTOTUNE_TABLE',None)
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8','benchmarks/torchrun_training.py','--backend','pf','--model','qwen122b','--tokens','20480','--warmups','3','--samples','5','--stream-ordered','--prime-trace','--fp8','--weight-grad-dtype','bf16','--cuda-trace-dir',str(traces)]
 with (output/f'{label}.log').open('w') as log:
  subprocess.run(['timeout','--kill-after=10s','180s',*command],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 results={}
 for kind,needle in [('fwd','weftmoe_pf::local::kernel'),('bwd','weftmoe_pf::backward_stream::kernel')]:
  ranks=[]
  for rank in range(8):
   events=json.loads((traces/f't20480.rank{rank}.json').read_text())['traceEvents']
   values=[e['dur'] for e in events if e.get('cat')=='kernel' and needle in e.get('name','')]
   assert len(values)==5,(label,kind,rank,len(values))
   ranks.append(values)
  results[kind+'_kernel_slowest_rank_p50_us']=statistics.median(map(max,zip(*ranks)))
 statuses=[]
 for line in (output/f'{label}.log').read_text().splitlines():
  if line.startswith('BACKWARD_STATUS '):
   statuses.append({k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)})
 results['status_records']=statuses
 item=dict(label=label,model='qwen122b',tokens=20480,results=results,command=command)
 summary.append(item)
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(label,{k:v for k,v in results.items() if k!='status_records'},'status_records',len(statuses),flush=True)
