import json,pathlib,statistics
b=pathlib.Path('build/fp8-profile-review');result=[]
for dtype in ['fp8','bf16']:
 for tokens in [8192,20480]:
  sample=[]
  for rank in range(8):
   data=json.loads((b/'service'/f'qwen35b-{dtype}'/'timeline'/f't{tokens}.rank{rank}.json').read_text());d={}
   for e in data['events']:
    code=e['event']
    if code in [111,112] and e['channel']==0:
     k=(e['cta'],e['arg0'],e['arg1']);name='producer_begin' if code==111 else 'producer_end'
    elif code in [82,113]:
     k=(e['cta'],e['arg0'],e['channel']);name='compute_begin' if code==82 else 'compute_ready'
    else:continue
    d.setdefault(k,{})[name]=e['timestamp_ns']
   for v in d.values():
    assert len(v)==4,v
    wait=v['compute_ready']-v['compute_begin'];overlap=max(0,min(v['compute_ready'],v['producer_end'])-max(v['compute_begin'],v['producer_begin']))
    sample.append((wait/1000,overlap/1000))
  r=dict(model='qwen35b',dtype=dtype,tokens=tokens,tile_count=len(sample),first_operand_wait_median_us=statistics.median(x[0] for x in sample),overlap_with_producer_x_full_wait_median_us=statistics.median(x[1] for x in sample),overlap_fraction_of_sum_wait=sum(x[1] for x in sample)/sum(x[0] for x in sample));result.append(r);print(r)
(b/'input-wait-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
