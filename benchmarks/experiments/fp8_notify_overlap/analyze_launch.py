"""Separate host submission gaps from GPU scheduling delays using CUPTI correlations."""
from pathlib import Path
import json,statistics
base=Path('build/fp8-notify-overlap');out=[]
for root in sorted((base/'profile').iterdir()):
 label,model=root.name.split('-',1)
 for stage in range(2):
  rows=[]
  for rank in range(8):
   events=json.loads((root/f'cuda/t20480.rank{rank}.json').read_text())['traceEvents']
   apis={e['args']['correlation']:e for e in events if e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{})}
   kernels=[e for e in events if e.get('cat')=='kernel']
   q=sorted([e for e in kernels if 'fp8::prepare_source_x' in e['name']],key=lambda e:e['ts']);n=sorted([e for e in kernels if 'weftmoe_pf::notify_kernel' in e['name']],key=lambda e:e['ts'])
   assert len(q)==len(n)==10
   for i in range(stage,10,2):
    qa,na=apis[q[i]['args']['correlation']],apis[n[i]['args']['correlation']];qe,ne=qa['ts']+qa['dur'],na['ts']+na['dur']
    rows.append(dict(notify_gpu_after_api_end_us=n[i]['ts']-ne,quant_remaining_at_notify_api_end_us=q[i]['ts']+q[i]['dur']-ne,quant_gpu_after_api_end_us=q[i]['ts']-qe,notify_remaining_at_quant_api_end_us=n[i]['ts']+n[i]['dur']-qe,notify_api_start_minus_quant_api_start_us=na['ts']-qa['ts']))
  out.append(dict(label=label,model=model,phase=['fwd','bwd'][stage],scope='Median8ranks x5samples, profiled CUDA API/kernel correlation; API return is an upper bound on submission time.',metrics={k:statistics.median(x[k] for x in rows) for k in rows[0]}))
(base/'launch-gaps.json').write_text(json.dumps(out,indent=2)+'\n')
