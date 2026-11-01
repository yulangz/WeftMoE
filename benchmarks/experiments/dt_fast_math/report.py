import hashlib
import json
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'build/dt-fast-math-20261101'

def records(path, prefix):
    data = path.read_text()
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(data[m.end():])[0] for m in re.finditer(re.escape(prefix), data)]

parity, checks, timings = [], [], []
previous = json.loads((ROOT/'docs/benchmarks/pf_stage_release_20261101.json').read_text())['pf_dt_forward']
for model in ('qwen35b', 'qwen122b', 'flash'):
    path = BASE/f'{model}-parity.log'
    if path.exists():
        rows = records(path, 'PF_DT ')
        for fmt in ('bf16', 'fp8', 'prequant'):
            group = [r for r in rows if r['format'] == fmt]
            if group:
                parity.append(dict(model=model, format=fmt, cases=len(group),
                    bitwise_equal_cases=sum(r['bitwise_equal'] for r in group),
                    max_diff=max(r['diff'] for r in group),
                    worst=max(group, key=lambda r: r['diff'])))
    path = BASE/f'{model}-dt.log'
    data = path.read_text() if path.exists() else ''
    decoder = json.JSONDecoder()
    rows = [decoder.raw_decode(data[m.start():])[0] for m in re.finditer(r'\{"rank": ', data)]
    if rows:
        checks.append(dict(model=model, cases=len(rows),
            known_case_waivers=sum(r['known_failure_waived'] for r in rows),
            max_unwaived_diff=max(r['diff'] for r in rows if not r['known_failure_waived'])))
    paths = [BASE/f'{model}-{label}-bench.log' for label in ('baseline','fast')]
    if all(p.exists() for p in paths):
        old = {r['tokens']: r for r in records(paths[0], 'RESULT ')}
        new = {r['tokens']: r for r in records(paths[1], 'RESULT ')}
        for n in old.keys() & new.keys():
            timings.append(dict(model=model, tokens=n, before=old[n], after=new[n]))
for row in parity:
    old = next(r for r in previous if r['model'] == row['model'] and r['format'] == row['format'])
    row['previous_max_diff'] = old['max_diff']
    row['previous_bitwise_equal_cases'] = old['bitwise_equal_cases']
hashes = json.loads((BASE/'manifest.json').read_text())['source_hashes']
for rel, digest in hashes.items():
    if rel.startswith(('src/', 'include/', 'configs/')):
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
out = dict(source_id='weftmoe-dt-fast-math-20261101', gpu='8x NVIDIA L20X', world=8,
    change='DT BF16 and FP8 SwiGLU use forward_tuning::kLocalPrefillFastMath, exactly like PF',
    unchanged=['MMA instructions and tile selection', 'K128 FP8 scale promotion', 'BF16 rounding and FP8 quantization boundaries', 'PF kernels'],
    pf_dt_workload=dict(experts=16, topk=2, tokens=[1,16,65,129,512],
        extra='rank0 empty, other ranks18..24; distributed and duplicate all-to-one routing'),
    dt_checks=checks, pf_dt_forward=parity, timings=timings,
    performance_method='EP8 max-rank CUDA-event median; CUDA graphs,50 calls,7 sweeps,first discarded; seed1700; online source work included, weight quantization excluded')
(ROOT/'docs/benchmarks/dt_fast_math_20261101.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(dict(checks=checks, parity=parity), indent=2))
