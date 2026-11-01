"""Summarize matched measurements without mixing CUDA-profile and wall timings."""
from pathlib import Path
import json,sys,statistics
path=Path(sys.argv[1]);data=json.loads(path.read_text());groups={}
for r in data['records']:groups.setdefault((r['model'],r['tokens']),{})[r['label']]=r
out=[]
for (model,tokens),rows in groups.items():
 if 'baseline' not in rows:continue
 b=rows['baseline'];reference=b['result']['wall_slowest_rank_p50_us'];end=rows.get('baseline_after',b)['result']['wall_slowest_rank_p50_us'];denom=(reference+end)/2
 result=dict(model=model,tokens=tokens,baseline_us=reference,baseline_after_us=end,baseline_drift_pct=100*(end/reference-1),rows=[])
 for label,r in rows.items():
  assert r['result']['workload_sha256_per_rank']==b['result']['workload_sha256_per_rank']
  value=r['result']['wall_slowest_rank_p50_us'];v=dict(label=label,us=value,speedup_vs_start=reference/value,speedup_vs_bracket_mean=denom/value)
  pairs=list(zip(r['result']['wall_slowest_rank_samples_us'], b['result']['wall_slowest_rank_samples_us'], rows.get('baseline_after',b)['result']['wall_slowest_rank_samples_us']))
  deltas=[100*(x/((lo+hi)/2)-1) for x,lo,hi in pairs]
  v.update(paired_median_latency_change_pct=statistics.median(deltas),paired_faster_samples=sum(x<0 for x in deltas),paired_samples=len(deltas))
  if r['mode']=='forward':
   ready=r['result']['forward_ready_slowest_rank_p50_us'];lo=b['result']['forward_ready_slowest_rank_p50_us'];hi=rows.get('baseline_after',b)['result']['forward_ready_slowest_rank_p50_us']
   v.update(forward_ready_us=ready,forward_ready_speedup_vs_bracket_mean=(lo+hi)/2/ready)
  result['rows'].append(v)
 print(model,tokens,'drift',round(result['baseline_drift_pct'],2),[(x['label'],round(x['us']/1000,3),round(x['speedup_vs_bracket_mean'],3)) for x in result['rows']]);out.append(result)
path.with_name('comparison.json').write_text(json.dumps(out,indent=2)+'\n')
