import hashlib,json,pathlib,re,statistics,subprocess
root=pathlib.Path.cwd();base=root/'build/fp8-rowcol-gemm';experiment=root/'benchmarks/experiments/fp8_rowcol_gemm'
report=dict(source_id='weftmoe-fp8-rowcol-gemm-20261030',base_commit='2715b12914aa70cba4169e7fb1a3e0943ca0e5cb',hardware='Single-node EP8, 8 x NVIDIA L20X (SM90)',register_partition=dict(math=224,control=56),decision='Reject both candidates for production; restore fixed224 starting implementation. Keep reproducible experimental patches.',scope='Local EP8 only. GIN/network code keeps baseline path; no multi-node performance claim.',algorithm=dict(baseline='During dgrad, convert saved row FP8 X in place once and save dY in column FP8. All dW tiles reuse these globally stored column operands.',candidate='Keep saved X row FP8; save dY row FP8. dW TMA loads row operands into shared scratch. Convert next K128 operand into the next shared WGMMA stage while current K128 MMA is in flight. No global column X/dY cache write. H and dZ remain directly column-quantized from BF16 in SwiGLU.',numerics='Actual per-column amax, pow2 scales, E4M3 RN, K128 FP32 promotion, BF16 dW; existing AB/Z path unchanged.',memory='Removes one global X read and one global X write during dgrad. dY global bytes unchanged. dW input read bytes unchanged, but conversion repeats per tile. Global allocation size unchanged. Extra shared scratch is reserved, not a separate global buffer.',variants=dict(fused='4 packed BF16 pairs per lane, warp-wide reduction, 16 channels per conversion iteration.',wide='16 packed BF16 pairs per lane, 8-lane reduction, 64 channels per conversion iteration. Fewer iterations but more static spill.')),quantization_reuse=[dict(model=m,x_repetitions=x,dy_repetitions=y,combined_conversion_work_vs_once=(x+y)/2) for m,x,y in [('qwen35b',8,2),('qwen122b',16,4),('flash',32,8)]],method='Unprofiled per-step slowest-rank wall median, 10 warmups + 30 samples after 40 full priming steps. Variable per-rank token counts around 8192/20480, identical random input/routing hashes. Source FP8 communication, BF16 dW. Weight preparation excluded. No compilation, disassembly or profiler overlaps performance runs. BWD-only prepares a fresh FWD tape outside timing; cycle includes FWD+BWD. These are not pure kernel timings.',records=[],correctness=[],compiler_static_resources=[],artifacts={})
hashes={}
for run in ['backward-1','backward-wide','cycle']:
 p=base/run/'results.json'
 if not p.exists():continue
 for r in json.loads(p.read_text())['records']:
  v=r['result'];key=(r['model'],r['tokens']);h=v['workload_sha256_per_rank']
  if key in hashes:assert hashes[key]==h,(key,run)
  hashes[key]=h
  report['records'].append(dict(run=run,label=r['label'],model=r['model'],tokens=r['tokens'],mode=r['mode'],hidden=v['hidden'],intermediate=v['intermediate'],experts=v['experts'],top_k=v['top_k'],wall_slowest_rank_p50_us=v['wall_slowest_rank_p50_us'],native_sha256=r['native_sha256'],workload_sha256_per_rank=h,command=r['command'],raw_log=str((base/run/f"{r['label']}-{r['model']}.log").relative_to(root))))
for label in ['fused','wide']:
 for model in ['qwen35b','qwen122b','flash']:
  p=base/f'{label}-{model}-check.log';s=p.read_text()
  markers={k:s.count(k) for k in ['FP8_PARITY_PASS','FP8_MULTIVERSION_ASYNC_PASS','FP8_CACHED_VERSION_CHECK_PASS','FP8_CHECKPOINT_PASS','FP8_OPTIMIZER_PASS','FP8_INFERENCE_PASS']}
  report['correctness'].append(dict(label=label,model=model,markers=markers,log=str(p.relative_to(root))))
  s=(base/f'{label}-{model}-build.log').read_text();match=re.search(r'Function properties for [^\n]*backward_stream6kernelILb0ELb1[^\n]*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',s)
  assert match
  report['compiler_static_resources'].append(dict(label=label,model=model,stack_bytes=int(match[1]),spill_store_bytes=int(match[2]),spill_load_bytes=int(match[3]),note='ptxas static function statistics; not measured runtime memory traffic. Math/control partition remains 224/56.'))
for model,stack,store,load in [('qwen35b',56,124,148),('qwen122b',48,200,208),('flash',56,144,164)]:
 report['compiler_static_resources'].append(dict(label='baseline',model=model,stack_bytes=stack,spill_store_bytes=store,spill_load_bytes=load))
report['conversion_tests']=dict(cases_per_variant=108,variants=['fused','wide'],reference='Independent host decode, double-precision amax/log2 scale, FP8 RN; original global conversion, shared full128 and split64 paths. Exact FP8 bytes and scales; NaN scale equivalence.',coverage=['zero rows','1/31/65/127/128 valid rows','poisoned tails','all zeros','BF16 subnormals','wide row-scale exponent range','NaN input','NaN scales'],racecheck='Both variants: 0 hazards, 0 errors, 0 warnings')
p=base/'diagnostic/summary.json'
if p.exists():
 report['diagnostic']=[]
 for item in json.loads(p.read_text()):
  rs=item['results'];statuses=rs['status_records'];ranks=[]
  for rank in range(8):
   entries=sorted([s for s in statuses if s['rank']==rank and not s['pending']],key=lambda s:s['epoch'])[-5:]
   ranks.append(entries)
  spans={}
  if all(len(r)==5 for r in ranks):
   for label,fn in [('start_to_dgrad_complete_us',lambda s:(s['dx_complete']-s['start'])/1000),('dw_phase_us',lambda s:(max(s['dw1_complete'],s['dw2_complete'])-min(s['dw1_start'],s['dw2_start']))/1000),('return_tail_after_dgrad_us',lambda s:max(0,s['return_complete']-s['dx_complete'])/1000),('dw2_span_us',lambda s:(s['dw2_complete']-s['dw2_start'])/1000),('dw1_span_us',lambda s:(s['dw1_complete']-s['dw1_start'])/1000)]:
    spans[label]=statistics.median(max(fn(r[i]) for r in ranks) for i in range(5))
  report['diagnostic'].append(dict(label=item['label'],model=item['model'],tokens=item['tokens'],kernel_us={k:v for k,v in rs.items() if k!='status_records'},status_clock_slowest_rank_p50=spans,note='Separate five-sample CUPTI + native-clock diagnostic with identical input/routing hashes. Different sample count and profiler instrumentation mean these absolute times must not be mixed with unprofiled results. dW1 and dW2 spans may overlap and must not be added. Wall timings here are not benchmark results.'))
for p in sorted(experiment.glob('*')):
 if p.is_file(): report['artifacts'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
report['comparison']=[]
for run in ['backward-wide','cycle']:
 selected=[r for r in report['records'] if r['run']==run]
 for model in ['qwen35b','qwen122b','flash']:
  for tokens in [8192,20480]:
   rows={r['label']:r for r in selected if r['model']==model and r['tokens']==tokens}
   if 'baseline' not in rows or 'wide' not in rows:continue
   old=rows['baseline']['wall_slowest_rank_p50_us'];new=rows['wide']['wall_slowest_rank_p50_us']
   report['comparison'].append(dict(mode=rows['baseline']['mode'],model=model,tokens=tokens,baseline_ms=old/1000,wide_ms=new/1000,wide_over_baseline=new/old))
report['sass_evidence']='Qwen35B first candidate: REDUX.MAX after GMMA issue and before WARPGROUP.DEPBAR; conversion is not wholly serialized after MMA completion. Static SASS in build/fp8-rowcol-gemm/fused-local-bwd.sass.'
report['raw_results']='build/fp8-rowcol-gemm/{backward-1,backward-wide,cycle}/results.json'
report['reproduction']=['Use an isolated checkout of base_commit.', 'git apply benchmarks/experiments/fp8_rowcol_gemm/baseline.patch', 'mkdir -p build/fp8-rowcol-gemm', '/opt/.venv/bin/python benchmarks/experiments/fp8_rowcol_gemm/build.py baseline -Xptxas=-v', 'git apply benchmarks/experiments/fp8_rowcol_gemm/wide.patch', '/opt/.venv/bin/python benchmarks/experiments/fp8_rowcol_gemm/build.py wide -DWEFTMOE_FP8_WGRAD_ROWCOL_FUSION=1 -Xptxas=-v', '/opt/.venv/bin/python benchmarks/experiments/fp8_rowcol_gemm/check.py wide', '/opt/.venv/bin/python benchmarks/experiments/fp8_rowcol_gemm/benchmark.py --native-root build/fp8-rowcol-gemm --labels baseline wide bf16 --mode backward --output build/fp8-rowcol-gemm/reproduce-bwd', 'For the first candidate: reverse wide.patch, apply fused.patch, build/check label fused with the same compile definition. To measure complete training cycles use --mode cycle and a new output directory.']
(root/'docs/benchmarks/fp8_rowcol_gemm_20261030.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('report:',len(report['records']),'measurements;',len(report['correctness']),'EP8 checks')
