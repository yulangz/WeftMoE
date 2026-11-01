"""Archive prequantized-source checks, timings and CUDA trace evidence."""
from pathlib import Path
import ast,hashlib,json,subprocess
root=Path.cwd();base=root/'build/fp8-prequant-input';exp=root/'benchmarks/experiments/fp8_prequant_input'
read=lambda p:json.loads(p.read_text())
r=read(exp/'README.json');r['base_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
r['builds']=read(base/'prequant-builds.json');r['checks']=[];r['measurements']={};r['comparisons']={}
markers=['FP8_PARITY_PASS','FP8_PREQUANT_INPUT_PASS','FP8_MULTIVERSION_ASYNC_PASS','FP8_CACHED_VERSION_CHECK_PASS','FP8_CHECKPOINT_PASS','FP8_OPTIMIZER_PASS','FP8_INFERENCE_PASS']
for model in ['qwen35b','qwen122b','flash','qwen35b-mixed']:
 path=base/f'prequant-{model}-check.log';s=path.read_text();counts={k:s.count(k) for k in markers}
 assert all(v>=8 for v in counts.values()),(path,counts)
 if model.endswith('mixed'):assert s.count('MIXED_BF16_FP8_POOL_PASS')>=16
 assert counts['FP8_PREQUANT_INPUT_PASS']==16
 r['checks'].append(dict(model=model,markers=counts,passed=True,log=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
for mode in ['cycle','forward','backward']:
 path=base/mode/'results.json';d=read(path);assert len(d['records'])==18
 hashes={}
 for row in d['records']:
  result=row['result'];key=(row['model'],row['tokens']);h=result['workload_sha256_per_rank']
  assert key not in hashes or hashes[key]==h;hashes[key]=h
  assert result['input_dtype']==('fp8_e4m3fn' if row['label']=='prequant' else 'bf16')
  assert result['weight_preparation_in_timing'] is False
  if row['label']=='prequant':assert result['input_preparation_in_timing'] is False and result['source_packet_copy_in_timing'] is True
 r['measurements'][mode]=dict(raw_path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data=d)
 r['comparisons'][mode]=read(base/mode/'comparison.json')
r['profile']=read(base/'profile-analysis.json');r['profile_manifest']=read(base/'profile-manifest.json')
r['backward_repeat']=read(base/'backward-repeat/results.json')
assert len(r['backward_repeat']['records'])==4
r['backward_repeat_comparison']=read(base/'backward-repeat/comparison.json')
assert len(r['profile'])==12
r['summary']=[]
for mode, comparisons in r['comparisons'].items():
 for c in comparisons:
  candidate=next(row for row in c['rows'] if row['label']=='prequant')
  baseline=(c['baseline_us']+c['baseline_after_us'])/2
  r['summary'].append(dict(mode=mode,model=c['model'],tokens_per_rank=c['tokens'],baseline_us=baseline,prequant_us=candidate['us'],latency_change_pct=100*(candidate['us']/baseline-1),baseline_drift_pct=c['baseline_drift_pct']))

r['initial_test_failure']=dict(log='build/fp8-prequant-input/initial-test-slot-lifetime-failure.log',cause='New inference comparison held the first inference tape while requesting a second, exhausting its single slot. Test corrected to retain only a cloned output and release tape before the next call. Production implementation unchanged.')
subprocess.run(['git','diff','--check'],check=True)
for p in [root/'weftmoe/runtime.py',root/'tests/torchrun_fp8.py',root/'benchmarks/torchrun_training.py',*exp.glob('*.py')]:ast.parse(p.read_text(),filename=str(p))
r['validation']=dict(diff_check='passed',python_ast='passed',ep8_runs=4,models=3,prequantized_source_bitwise_comparisons='rtol=0, atol=0 for y/dx/dgate/dw1/dw2; 2 routing patterns x 4 input-format combinations per model/run',profile_confirms_no_source_quantization=True)
r['artifacts']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in exp.iterdir() if p.is_file()}
r['starting_diff_sha256']=hashlib.sha256((base/'starting.patch').read_bytes()).hexdigest();r['current_diff']=subprocess.check_output(['git','diff'],text=True)
for b in r['builds']:
 p=base/f"prequant-{b['model']}/libweftmoe.so";b['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
p=root/'docs/benchmarks/fp8_prequant_input_20261031.json';p.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');print(p)
