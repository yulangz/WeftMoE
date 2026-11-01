from pathlib import Path
import json,statistics
base=Path('build/fp8-prequant-input');records=[];hashes={}
for model in ['qwen35b','qwen122b','flash']:
 for label in ['baseline','prequant']:
  root=base/'profile'/f'{label}-{model}'
  logs=[json.loads(s.split(' ',1)[1]) for s in (root/'launch.log').read_text().splitlines() if s.startswith('TRAINING_CYCLE_JSON ')]
  assert len(logs)==1
  h=logs[0]['workload_sha256_per_rank'];assert model not in hashes or hashes[model]==h;hashes[model]=h
  phase={'fwd':[],'bwd':[]}
  for rank in range(8):
   events=json.loads((root/'cuda'/f't20480.rank{rank}.json').read_text())['traceEvents'];kernels=[x for x in events if x.get('cat')=='kernel']
   def select(s):return sorted([x for x in kernels if s in x['name']],key=lambda x:x['ts'])
   quant=select('fp8::copy_source_x' if label=='prequant' else 'fp8::prepare_source_x');notify=select('weftmoe_pf::notify_kernel');ready=select('weftmoe_pf::source_ready_kernel')
   fwd=select('weftmoe_pf::local::kernel');bwd=select('weftmoe_pf::backward_stream::kernel')
   assert len(quant)==len(notify)==10 and len(fwd)==len(bwd)==5,(model,label,rank,len(quant),len(notify))
   assert len(ready)==10
   assert not select('fp8::prepare_source_x' if label=='prequant' else 'fp8::copy_source_x'), (model,label,rank,'wrong source kernel')
   for i in range(5):
    for part,main,offset in [('fwd',fwd[i],0),('bwd',bwd[i],1)]:
     q,n=quant[2*i+offset],notify[2*i+offset]
     overlap=max(0,min(q['ts']+q['dur'],n['ts']+n['dur'])-max(q['ts'],n['ts']))
     phase[part].append(dict(rank=rank,sample=i,source_prepare_us=q['dur'],notify_us=n['dur'],overlap_us=overlap,source_overlap_fraction=overlap/q['dur'],preparation_to_main_us=main['ts']-min(q['ts'],n['ts']),main_us=main['dur'],publication_us=ready[2*i+offset]['dur'] if ready else 0,quant_stream=q['args'].get('stream'),notify_stream=n['args'].get('stream')))
  for part,rows in phase.items():
   metrics={key:statistics.median(x[key] for x in rows) for key in ['source_prepare_us','notify_us','overlap_us','source_overlap_fraction','preparation_to_main_us','main_us','publication_us']}
   r=dict(model=model,label=label,phase=part,tokens=20480,metric_scope='Median across8ranks x5profiled samples; CUPTI timings, not unprofiled critical-rank wall latency.',metrics=metrics,workload_sha256_per_rank=h,per_rank_samples=rows)
   records.append(r);print(model,label,part,{k:round(v,3) for k,v in metrics.items()},flush=True)
(base/'profile-analysis.json').write_text(json.dumps(records,indent=2)+'\n')
