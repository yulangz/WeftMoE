import pathlib,json,re,statistics
base=pathlib.Path('build/fp8-profile-review');med=statistics.median;records=[]
for model in ['qwen35b','qwen122b','flash']:
 for dtype in ['fp8','bf16']:
  root=base/'service'/f'{model}-{dtype}';logs={int(p.parent.name):p for p in (root/'ranks').rglob('stdout.log')}
  if len(logs)!=8:continue
  cases=[json.loads(l.split(' ',1)[1]) for l in logs[0].read_text().splitlines() if l.startswith('TRAINING_CYCLE_JSON ')]
  statuses={rank:[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)} for line in p.with_name('stderr.log').read_text().splitlines() if line.startswith('BACKWARD_STATUS ')] for rank,p in logs.items()}
  assert all(len(v)==32 for v in statuses.values())
  for case,v in enumerate(cases):
   samples=[]
   for i in range(5):
    status=[statuses[r][case*16+11+i] for r in range(8)];rows=sum(v['window_padding']['actual_rows'][i]);padded=sum(v['window_padding']['canonical_m128_rows'][i])
    sums={k:sum(s[k] for s in status)/1000 for k in ['du_service_ns','dx_service_ns','swiglu_service_ns','du_ready_wait_ns','dx_ready_wait_ns']}
    total=sum(sums[k] for k in ['du_service_ns','dx_service_ns','swiglu_service_ns'])
    samples.append(dict(cta_service_us=sums,service_fraction={k:sums[k]/total for k in ['du_service_ns','dx_service_ns','swiglu_service_ns']},cta_us_per_actual_row={k:sums[k]/rows for k in sums},rows=rows,padded_rows=padded))
   r=dict(model=model,dtype=dtype,tokens=v['nominal_tokens_per_rank'],hidden=v['hidden'],intermediate=v['intermediate'],experts=v['experts'],top_k=v['top_k'],scope='CTA service sums include stalls; producer waits overlap consumer service and are not additive. Five measured samples after priming.',samples=samples,service_fraction={k:med(x['service_fraction'][k] for x in samples) for k in samples[0]['service_fraction']},cta_us_per_actual_row={k:med(x['cta_us_per_actual_row'][k] for x in samples) for k in sums})
   records.append(r);print(model,dtype,r['tokens'],'shares',r['service_fraction'],'cta_us/row',r['cta_us_per_actual_row'])
(base/'service-analysis.json').write_text(json.dumps(records,indent=2)+'\n')
