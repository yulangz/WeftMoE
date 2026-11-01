"""Collect raw log provenance, validation and DT FP8 timing comparisons."""
import hashlib,json,pathlib,re,subprocess,sys
root=pathlib.Path.cwd();base=root/'build/dt-fp8';selected=sys.argv[1] if len(sys.argv)>1 else 'final'
checks=[];runs=[];traces=[]
for p in sorted(base.glob('*.log')):
 s=p.read_text(errors='replace')
 rows=[json.loads(m) for m in re.findall(r'\{[^{}]+\}',s) if '"diff"' in m and 'case' in m]
 if rows:
  checks.append(dict(log=str(p.relative_to(root)),cases=len(rows),max_diff=max(x['diff'] for x in rows),failed='Traceback' in s or 'AssertionError' in s))
 for l in s.splitlines():
  if l.startswith('RESULT '):
   r=json.loads(l[7:]);r.update(log=str(p.relative_to(root)),online_speedup=r['bf16_us']/r['fp8_online_us'],prequant_speedup=r['bf16_us']/r['fp8_prequant_us'])
   runs.append(r)
  if l.startswith(('KERNELS ','PHASES ')):
   kind,value=l.split(' ',1);traces.append(dict(log=str(p.relative_to(root)),kind=kind,result=json.loads(value)))
files=['src/cuda/weftmoe_dt/fp8_compute.cuh','src/cuda/weftmoe_dt/kernel.cuh','src/cuda/weftmoe_dt/schedule.cuh','src/cuda/weftmoe_dt/compute.cuh','src/cuda/activation_transport.cu','src/cuda/forward/kernel_contract.cuh','src/cuda/forward/kernel_support.cuh','src/python/runtime.cpp','weftmoe/runtime.py','weftmoe/fp8.py','tests/torchrun_dt_fp8.py','benchmarks/experiments/dt_fp8/benchmark.py']
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report=dict(source_id='weftmoe-dt-fp8-20261031',selected_build=selected,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={p:sha(p) for p in files},libraries={str(p.relative_to(root)):sha(p) for p in base.glob('*/libweftmoe.so')},hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).splitlines(),
 implementation=dict(source='BF16 -> rowwise E4M3 pow2 quantization, or unchanged FP8+scale copy, directly into symmetric source packets; concurrent with CTA0 mini notify inside DT; existing BeforePull publishes both',compute='Fixed M64 WGMMA FP8, two math warpgroups with N128 partials; scale promotion every K128, FP32 totals; FC1/FC2 N256',activation='Full 128-channel gate/up blocks; preserve BF16 and FP8 AB rounding, gated SwiGLU, BF16 round then rowwise FP8 S',output='BF16 return and deterministic combine',registers='Fixed math224/control56, after source/notify/geometry; no model-specific tuning',api='TensorRuntime.forward_fp8; TensorRuntime.inference_fp8 for warmed stable-address CUDA graphs; inference-only DT, single node'),
 selection='Keep FC1/FC2 N256 and hoisted scale products. Reject bulk-copy pull: geometric mean speedup across 21 points was only 1.0069x, with Qwen35B regressions up to about 2.4%; retain the simpler vector pull and no extra tuning switch.',
 profile_caveat='Device phase clocks from one-shot eager launches include inter-rank launch skew and overlap; do not sum phases or substitute these values for the interleaved CUDA-graph benchmark.',
 regression_logs={name:dict(log='build/dt-fp8/'+name+'.log',failed=('Traceback' in (base/(name+'.log')).read_text()),pass_markers=(base/(name+'.log')).read_text().count('PASS')) for name in ['final-bf16-regression','final-pf-regression'] if (base/(name+'.log')).exists()},
 methodology='Single-node EP8. Per case: one warm sweep then six timed sweeps in alternating variant order; each measurement is 50 graph calls averaged, reduce max across ranks, report median. Includes source preparation, notify, dispatch, both GEMMs, activation, BF16 return/combine. Weight preparation excluded; prequant input prepared before measurement. Final and *-random logs use uniform random top-k without replacement, fixed within timed replay; earlier exploratory logs use cyclic contiguous routing. Synthetic routed layer, steady state; no attention, shared expert or host scheduling. Profile runs are excluded from final timing comparisons.',
 validation=checks,profiles=traces,measurements=runs,
 final_measurements=[r for r in runs if ('/'+selected+'-') in r['log'] and '-bench-random.log' in r['log']],
 limitations=['DT backward and multi-node transport remain out of scope; use PF for training/multi-node.','FP8 E4M3 GEMMs and extra activation rounding change values versus BF16; no convergence evaluation.','Numerical acceptance uses global diff < 5e-3 against original BF16 for every M/token size; dprob is report-only with finite checks. Historical logs retain their original measurement definitions.','Earlier timing variants are exploratory; only compare variants with matching routing policy.'])
out=root/'docs/benchmarks/dt_fp8_20261031.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
for r in report['final_measurements']:print(r['hidden'],r['tokens'],*[round(r[k],3) for k in ('bf16_us','fp8_online_us','fp8_prequant_us','online_speedup')])
