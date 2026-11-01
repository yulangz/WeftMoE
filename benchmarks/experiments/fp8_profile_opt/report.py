from pathlib import Path
import json,hashlib,re,subprocess,statistics
root=Path.cwd();base=root/'build/fp8-profile-opt';exp=root/'benchmarks/experiments/fp8_profile_opt';metadata=json.loads((exp/'README.json').read_text())
report=dict(metadata,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),attempts=[],measurements={},artifacts={},limitations=['Single-node EP8 only; no multi-node performance claim.','Unprofiled wall measurements and profiler GPU kernel durations remain separate.','Performance ratios compare matching input/routing hashes within a run; different sample counts use different workload traces.','Compiler spill byte counts are static, not dynamic traffic.'])
checks=[]
for name in ['checks.json','followup-checks.json','last-checks.json']:
 if (base/name).exists():checks.extend(json.loads((base/name).read_text()))
report['profile_basis']='docs/benchmarks/fp8_profile_review_20261030.json'
report['limitations'] += ['Correctness tests use 257 tokens with tails, empty ranks and protocol stress; no training convergence claim.', 'Most parity checks use 16 global experts; combined Qwen35B uses 256. Performance uses the full model expert counts.']
for check in checks:
 r=dict(check,models=[])
 for m in ['qwen35b','qwen122b','flash']:
  path=base/f"{check['label']}-{m}-check.log"
  if path.exists():
   s=path.read_text();r['models'].append(dict(model=m,log=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),markers={k:s.count(k) for k in ['FP8_PARITY_PASS','FP8_MULTIVERSION_ASYNC_PASS','FP8_CACHED_VERSION_CHECK_PASS','FP8_CHECKPOINT_PASS','FP8_OPTIMIZER_PASS','FP8_INFERENCE_PASS']}))
 report['attempts'].append(r)
for path in sorted(base.glob('*/results.json')):
 r=json.loads(path.read_text());hashes={}
 for row in r['records']:
  key=(row['model'],row['tokens']);workload=row['result']['workload_sha256_per_rank']
  assert key not in hashes or hashes[key]==workload, (path,key)
  hashes[key]=workload
 r['raw_results']=str(path.relative_to(root));r['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 fields=['schema_version','backend','world','gpu','nominal_tokens_per_rank','seed','token_std_fraction','route_bias_std','workload_sha256_per_rank','experts','top_k','hidden','intermediate','warmups','priming_steps','samples','forward_only','backward_only','compute_dtype','weight_grad_dtype','weights_prepared_once','weight_preparation_in_timing','stream_ordered','wall_slowest_rank_p50_us','wall_slowest_rank_samples_us','forward_ready_slowest_rank_p50_us','forward_ready_timing_scope','measurement_scope']
 for row in r['records']:
  original=row['result'];row['result']={key:original[key] for key in fields if key in original}
  if original.get('forward_only'):
   row['result']['forward_ready_slowest_rank_samples_us']=list(map(max,zip(*original['per_rank_forward_ready_us'])))
 report['measurements'][path.parent.name]=r
for name in ['compiler-resources.json','native-manifest.json','decision.json','final-validation.json','kernel-profile.json','sass-comparison.json']:
 if (base/name).exists():report[name.removesuffix('.json')]=json.loads((base/name).read_text())
for name in ['decode-unit','swiglu-unit']:
 if (base/f'{name}.log').exists():report[name]=(base/f'{name}.log').read_text()
for p in sorted(exp.iterdir()):
 if p.is_file():report['artifacts'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
report['production_diff']=subprocess.check_output(['git','diff','--','src'],text=True)
report['reproduction']=['Use base_commit with benchmarks/experiments/fp8_profile_review/baseline.patch applied (fixed224/56).','python benchmarks/experiments/fp8_profile_opt/prepare.py','python build/fp8-profile-opt/build.py baseline -Xptxas=-v','python benchmarks/experiments/fp8_profile_opt/build_isolated.py subgroup decode swiglu dw_order dw_pipeline dw_scale pull3 pull_direct','Compile decode_check.cu and tests/cuda/fp8_column.cu with -O2 -std=c++17 -arch=sm_90a and -Ibuild/fp8-profile-opt/trees/decode/src -Iinclude -I/opt/cutlass/include; output binaries in build/fp8-profile-opt.','python benchmarks/experiments/fp8_profile_opt/evaluate.py subgroup decode swiglu dw_order dw_pipeline dw_scale pull3 pull_direct','Use combine.py label [selected variants] and build_isolated.py label for combinations. Keep each comparison on identical workload and fixed224/56. Build packed and combined, compile swiglu_check.cu against trees/packed/src with the flags above, then run followup.py. Build and check fwd_pack, then run benchmark.py --native-root build/fp8-profile-opt --labels baseline subgroup fwd_pack bf16 baseline_after --mode cycle --warmups 10 --samples 30 --output build/fp8-profile-opt/final-cycle.','benchmark.py supports cycle/forward/backward; use --warmups 10 --samples 30 for confirmation and preserve raw outputs in new directories.']
p=root/'docs/benchmarks/fp8_profile_opt_20261031.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(p)
