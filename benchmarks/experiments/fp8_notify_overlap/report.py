"""Archive results and raw-data hashes for the notify/pre-quant overlap experiment."""
from pathlib import Path
import json,hashlib,subprocess
root=Path.cwd();base=root/'build/fp8-notify-overlap';exp=root/'benchmarks/experiments/fp8_notify_overlap'
read=lambda p:json.loads(p.read_text())
r=read(exp/'README.json');r['base_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();r['checks']=read(base/'checks.json');r['builds']=read(base/'isolated-builds.json');r['measurements']={};r['artifacts']={}
for entry in read(base/'production-builds.json'):
 r['builds'].append(dict(entry,label='production'))
markers=['FP8_PARITY_PASS','FP8_MULTIVERSION_ASYNC_PASS','FP8_CACHED_VERSION_CHECK_PASS','FP8_CHECKPOINT_PASS','FP8_OPTIMIZER_PASS','FP8_INFERENCE_PASS']
for check in r['checks']:
 check['logs']=[]
 for model in check['models']:
  path=base/f"{check['label']}-{model}-check.log";s=path.read_text();counts={k:s.count(k) for k in markers};assert all(v>=8 for v in counts.values()),path
  check['logs'].append(dict(model=model,path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),markers=counts))
for build in r['builds']:
 p=base/f"{build['label']}-{build['model']}/libweftmoe.so";build['native_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
for path in base.glob('*/results.json'):
 data=read(path);hashes={}
 for row in data['records']:
  result=row['result'];key=(row['model'],row['tokens']);h=result['workload_sha256_per_rank'];assert key not in hashes or hashes[key]==h;hashes[key]=h
  keep=['gpu','experts','top_k','hidden','intermediate','world','seed','warmups','samples','priming_steps','compute_dtype','weight_grad_dtype','weight_preparation_in_timing','workload_sha256_per_rank','wall_slowest_rank_p50_us','wall_slowest_rank_samples_us','measurement_scope']
  row['result']={k:result[k] for k in keep}
 data['raw_path']=str(path.relative_to(root));data['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();r['measurements'][path.parent.name]=data
for name in ['profile-analysis','profile-manifest','profile-ordered-manifest','launch-gaps','mixed-check','production-mixed-check','decision','final-validation']:
 p=base/f'{name}.json'
 if p.exists():r[name]=read(p)
for p in exp.iterdir():
 if p.is_file():r['artifacts'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
r['production_diff']=subprocess.check_output(['git','diff','--','src'],text=True)
r['reproduction']=['Apply starting-state.patch to base_commit in a separate checkout.','python benchmarks/experiments/fp8_notify_overlap/prepare.py','python benchmarks/experiments/fp8_notify_overlap/build.py baseline -Xptxas=-v. Original baseline natives alias build/fp8-profile-opt/subgroup-{model}.','python benchmarks/experiments/fp8_notify_overlap/build_isolated.py fwd bwd both delay','python benchmarks/experiments/fp8_notify_overlap/evaluate.py','python benchmarks/experiments/fp8_notify_overlap/build_isolated.py early', 'python benchmarks/experiments/fp8_notify_overlap/followup.py','python benchmarks/experiments/fp8_notify_overlap/build_isolated.py ordered ordered_fwd ordered_half ordered_delay', 'python benchmarks/experiments/fp8_notify_overlap/ordered.py', 'python benchmarks/experiments/fp8_notify_overlap/analyze_launch.py', 'Apply production.patch to the starting state, then python benchmarks/experiments/fp8_notify_overlap/build.py production -Xptxas=-v and python benchmarks/experiments/fp8_notify_overlap/production.py']
p=root/'docs/benchmarks/fp8_notify_overlap_20261031.json';p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(p)
