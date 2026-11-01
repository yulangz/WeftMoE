import json,pathlib,re,statistics,math
base=pathlib.Path('build/fp8-profile-review'); records=[];hashes={}
med=statistics.median
def corr(a,b):
 ma,mb=statistics.mean(a),statistics.mean(b);da=[v-ma for v in a];db=[v-mb for v in b]
 return sum(x*y for x,y in zip(da,db))/math.sqrt(sum(x*x for x in da)*sum(x*x for x in db))
for model in ['qwen35b','qwen122b','flash']:
 for dtype in ['fp8','bf16']:
  root=base/'production'/f'{model}-{dtype}'
  if not root.exists():continue
  logs={int(p.parent.name):p for p in (root/'ranks').rglob('stdout.log')}
  raw=[json.loads(l.split(' ',1)[1]) for l in logs[0].read_text().splitlines() if l.startswith('TRAINING_CYCLE_JSON ')]
  statuses={rank:[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)} for line in p.with_name('stderr.log').read_text().splitlines() if line.startswith('BACKWARD_STATUS ')] for rank,p in logs.items()}
  assert all(len(v)==32 for v in statuses.values())
  for case,v in enumerate(raw):
   tokens=v['nominal_tokens_per_rank'];h=v['workload_sha256_per_rank'];key=(model,tokens)
   if key in hashes:assert hashes[key]==h
   hashes[key]=h
   traces=[]
   for rank in range(8):
    j=json.loads((root/'cuda'/f't{tokens}.rank{rank}.json').read_text());events=[e for e in j['traceEvents'] if e.get('cat')=='kernel']
    traces.append(events)
   def select(needle):return [sorted([e for e in rank if needle in e['name']],key=lambda e:e['ts']) for rank in traces]
   fwd=select('weftmoe_pf::local::kernel');bwd=select('weftmoe_pf::backward_stream::kernel');quant=select('fp8::prepare_source_x');notify=select('weftmoe_pf::notify_kernel')
   assert all(len(r)==5 for r in fwd+bwd)
   assert all(len(r)==10 for r in notify)
   status=[[statuses[r][case*16+i] for i in range(11,16)] for r in range(8)]
   phase=[]
   for r in range(8):
    row=[]
    for s in status[r]:
     dwend=max(s['dw1_complete'],s['dw2_complete']);dwstart=min(s['dw1_start'],s['dw2_start'])
     row.append(dict(dgrad_us=(s['dx_complete']-s['start'])/1000,dw_us=(dwend-dwstart)/1000,rank_join_us=(s['return_complete']-s['dx_complete'])/1000,reduce_us=(s['reduce_complete']-s['reduce_start'])/1000,dw_exposed_after_reduce_us=max(0,dwend-s['reduce_complete'])/1000,reduce_exposed_after_dw_us=max(0,s['reduce_complete']-dwend)/1000,ready_us=(max(dwend,s['reduce_complete'])-s['start'])/1000))
    phase.append(row)
   critical=[max(range(8),key=lambda r:phase[r][i]['dgrad_us']) for i in range(5)]
   metrics=dict(fwd_kernel_us=med(max(r[i]['dur'] for r in fwd) for i in range(5)),bwd_kernel_us=med(max(r[i]['dur'] for r in bwd) for i in range(5)))
   metrics.update({'critical_'+k:med(phase[critical[i]][i][k] for i in range(5)) for k in phase[0][0]})
   metrics.update({'max_'+k:med(max(r[i][k] for r in phase) for i in range(5)) for k in ['dw_exposed_after_reduce_us','reduce_exposed_after_dw_us','rank_join_us']})
   metrics['max_notify_fwd_us']=med(max(r[2*i]['dur'] for r in notify) for i in range(5));metrics['max_notify_bwd_us']=med(max(r[2*i+1]['dur'] for r in notify) for i in range(5))
   if dtype=='fp8':
    assert all(len(r)==10 for r in quant)
    for which,name in [(0,'X'),(1,'dY')]:
     metrics[f'max_quant_{name}_us']=med(max(r[2*i+which]['dur'] for r in quant) for i in range(5))
     metrics[f'quant_notify_{name}_overlap_us']=med(max(max(0,min(quant[r][2*i+which]['ts']+quant[r][2*i+which]['dur'],notify[r][2*i+which]['ts']+notify[r][2*i+which]['dur'])-max(quant[r][2*i+which]['ts'],notify[r][2*i+which]['ts'])) for r in range(8)) for i in range(5))
   padded=v['window_padding']['canonical_m128_rows'];actual=v['window_padding']['actual_rows']
   metrics['load_max_over_mean']=med(max(row)/statistics.mean(row) for row in padded)
   metrics['padding_fraction']=med(sum(p)/sum(a)-1 for p,a in zip(padded,actual))
   metrics['row_dgrad_correlation']=med(corr(padded[i],[phase[r][i]['dgrad_us'] for r in range(8)]) for i in range(5))
   record=dict(model=model,dtype=dtype,tokens=tokens,metrics=metrics,critical_dgrad_ranks=critical,per_rank_phase=phase,per_step_padded_rows=padded,workload_sha256_per_rank=h)
   records.append(record)
   print(model,dtype,tokens,{k:round(metrics[k],3) for k in ['fwd_kernel_us','bwd_kernel_us','critical_dgrad_us','critical_dw_us','critical_rank_join_us','critical_reduce_us','max_rank_join_us','max_dw_exposed_after_reduce_us','load_max_over_mean','row_dgrad_correlation']})
(base/'production-analysis.json').write_text(json.dumps(records,indent=2)+'\n')
