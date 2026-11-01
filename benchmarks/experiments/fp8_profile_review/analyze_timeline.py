import pathlib,json,statistics,collections,re
base=pathlib.Path('build/fp8-profile-review');records=[];med=statistics.median
pairs={'fc1_task_to_math_end':(82,83),'fc1_after_first_operand':(113,83),'fc1_first_operand_wait':(82,113),'fc1_ring_wait':(83,123),'fc1_epilogue':(123,84),'fc2_task_to_math_end':(85,86),'fc2_after_first_operand':(114,86),'fc2_first_operand_wait':(85,114),'fc2_epilogue':(86,87),'du':(124,125),'dx':(126,127),'swiglu':(128,129)}
for model in ['qwen35b','qwen122b','flash']:
 for dtype in ['fp8','bf16']:
  root=base/'service'/f'{model}-{dtype}';paths=list(root.glob('timeline/*.json'))
  if len(paths)!=16:continue
  for tokens in [8192,20480]:
   sums=collections.defaultdict(float);counts=collections.Counter();samples=collections.defaultdict(list);ranks=[]
   for rank in range(8):
    j=json.loads((root/'timeline'/f't{tokens}.rank{rank}.json').read_text());assert j['overflow_events']==0
    start={};done={};dwstarts={};dwends={};rank_sums=collections.defaultdict(float)
    for e in j['events']:
     code=e['event'];key=(e['cta'],e['channel'],e['arg0'])
     if code==111:start[('producer',key,e['arg1'])]=e['timestamp_ns']
     if code==112:
      duration=(e['timestamp_ns']-start.pop(('producer',key,e['arg1'])))/1000
      label='fc2_producer_input_wait' if e['channel'] else 'fc1_producer_input_wait'
      samples[label].append(duration);sums[label]+=duration;counts[label]+=1
     for name,(a,b) in pairs.items():
      if code==a:start[(name,key)]=e['timestamp_ns']
      if code==b:
       k=(name,key);assert k in start,(name,key)
       duration=(e['timestamp_ns']-start.pop(k))/1000;assert duration>=0;samples[name].append(duration);sums[name]+=duration;rank_sums[name]+=duration;counts[name]+=1
     if code==118:
      label='fc2_tma_wait' if e['channel'] else 'fc1_tma_wait';samples[label].append(e['arg0']/1000);sums[label]+=e['arg0']/1000
     if code==130:
      samples['rowcol'].append(e['arg0']/1000);sums['rowcol']+=e['arg0']/1000;rank_sums['rowcol']+=e['arg0']/1000
     if code==131:dwstarts[e['cta']]=e['timestamp_ns']
     if code==132:dwends[e['cta']]=e['timestamp_ns']
    if not dwstarts:continue
    assert len(dwstarts)==len(dwends)
    first=min(dwstarts.values());last=max(dwends.values());span=last-first
    ends=sorted(dwends.values());n=len(ends)
    tail_half=(last-ends[n//2-1])/span;tail_quarter=(last-ends[3*n//4-1])/span
    rank_item=dict(rank=rank,dw_ctas=n,dw_span_us=span/1000,dw_work_interval_fill=sum(dwends[c]-dwstarts[c] for c in dwstarts)/(n*span),dw_final_half_ctas_fraction=tail_half,dw_final_quarter_ctas_fraction=tail_quarter,dw_end_spread_us=(last-min(dwends.values()))/1000,math_task_service_us=dict(rank_sums))
    ranks.append(rank_item)
   total=sums['du']+sums['dx']+sums['swiglu']
   result=dict(model=model,dtype=dtype,tokens=tokens,scope='Instrumented single last sample; sums are CTA service time, not wall latency. FWD trace is from the untimed tape preparation; BWD task traces come from the same iteration.',tile_count=dict(counts),tile_median_us={k:med(v) for k,v in samples.items()},cta_sum_us=dict(sums),bwd_service_fraction={k:sums[k]/total if total else None for k in ['du','dx','swiglu','rowcol']},per_rank=ranks)
   records.append(result)
   critical=max(ranks,key=lambda r:r['dw_span_us']) if ranks else dict(dw_work_interval_fill=0,dw_final_quarter_ctas_fraction=0);print(model,dtype,tokens,'tile_us',{k:round(result['tile_median_us'][k],2) for k in ['fc1_ring_wait','fc1_epilogue','fc1_first_operand_wait','fc1_after_first_operand','fc2_after_first_operand']},'bwd_share',{k:round(v,3) if v is not None else None for k,v in result['bwd_service_fraction'].items()},'dw_fill',round(critical['dw_work_interval_fill'],3),'tail_q',round(critical['dw_final_quarter_ctas_fraction'],3))
(base/'timeline-analysis.json').write_text(json.dumps(records,indent=2)+'\n')
