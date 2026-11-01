"""Summarize strict training/inference parity, full DT checks and paired timings."""
import hashlib
import json
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'build/dt-m64-20261101'
MODELS = ('qwen35b', 'qwen122b', 'flash')

def records(path, prefix):
    s = path.read_text()
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(s[m.end():])[0] for m in re.finditer(re.escape(prefix), s)]

parity = []
for phase in ('before', 'after'):
    for model in MODELS:
        for compact in (False, True):
            path = BASE / f'{phase}-{model}-parity{"-compact" if compact else ""}.log'
            rs = records(path, 'PF_DT ')
            assert len(rs) == (480 if compact else 720), path
            assert all(r['bitwise_equal'] and r['diff'] == 0 for r in rs), path
            for fmt in (('fp8', 'prequant') if compact else ('bf16', 'fp8', 'prequant')):
                group = [r for r in rs if r['format'] == fmt]
                parity.append(dict(phase=phase, model=model, compact_workspace=compact,
                                   format=fmt, cases=len(group), bitwise_equal=len(group), max_diff=0))
checks = []
rows = []
for model in MODELS:
    s = (BASE / f'{model}-check.log').read_text()
    decoder = json.JSONDecoder()
    rs = [decoder.raw_decode(s[m.start():])[0] for m in re.finditer(r'\{"rank": ', s)]
    assert len(rs) == 272
    checks.append(dict(model=model, rank_cases=len(rs),
                       known_case_waivers=sum(r['known_failure_waived'] for r in rs),
                       max_unwaived_diff=max(r['diff'] for r in rs if not r['known_failure_waived'])))
    before = records(BASE / f'before-{model}-bench.log', 'RESULT ')
    after = records(BASE / f'{model}-bench.log', 'RESULT ')
    assert len(before) == len(after) == 7
    for a, b in zip(before, after):
        assert a['tokens'] == b['tokens']
        metrics = ('bf16_us', 'fp8_online_us', 'fp8_prequant_us')
        rows.append(dict(model=model, tokens=b['tokens'],
                         before={k:a[k] for k in metrics}, after={k:b[k] for k in metrics},
                         latency_change_percent={k:100*(b[k]/a[k]-1) for k in metrics}))
manifest = json.loads((BASE / 'manifest.json').read_text())
for rel, digest in manifest['source_hashes'].items():
    if rel.startswith(('src/', 'include/', 'configs/')):
        assert hashlib.sha256((BASE / 'source' / rel).read_bytes()).hexdigest() == digest, rel
report = dict(source_id='weftmoe-dt-m64-20261101', gpu='8x NVIDIA L20X', world=8,
    policy='Fixed M64 in BF16 and FP8 DT; unchanged PF geometry and FP8 register partition224/56.',
    parity_method='PF training=True with retained backward state, consumed by actual backward with finite-gradient checks. DT inference on identical input/routes/weights. Exact BF16 storage-bit equality; normal and compact FP8 workspaces. Three dimensions,16experts/topk2,14 uniform token counts1..512 plus unequal lengths/rank0 empty; distributed and single-rank hotspot routing.',
    parity=parity, dt_checks=checks,
    performance_method='Paired old forced-M64 and new fixed-M64 binaries, seed1700. EP8,256experts,model topk8/8/6,100 calls per CUDA graph,9 alternating datatype sweeps/discardfirst,max-rank median. Online source quantization included, weight preparation excluded. No concurrent compilation/profiling.',
    performance=rows,
    compiler_resources=json.loads((BASE / 'compiler_resources.json').read_text()),
    regression_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ROOT/'tests/torchrun_pf_dt.py', ROOT/'tests/torchrun_dt_fp8.py']},
    limitations='The existing Qwen35B rank7 one-token BF16-reference failure is waived only by the preexisting bounded rule; PF/DT parity has no waivers. Finite backward gradients establish consumable caches, not a new backward accuracy claim. No cross-hardware or arbitrary-input bitwise guarantee.',
    removed=['M16 BF16/FP8 MMA and epilogues','M template parameter and runtime crossover','narrow-only CTA multiplier','extra M16 TMA descriptors','M selection macros, profile entries and autotune axes','M16 experiment builders/patches archived outside active source'],
    source_hashes={k:v for k,v in manifest['source_hashes'].items() if k.startswith(('src/cuda/weftmoe_dt/', 'configs/profiles/'))})
out = ROOT / 'docs/benchmarks/dt_m64_20261101.json'
out.write_text(json.dumps(report, indent=2)+'\n')
print('parity', sum(r['cases'] for r in parity), 'DT checks', sum(r['rank_cases'] for r in checks))
for model in MODELS:
    for metric in ('bf16_us','fp8_online_us','fp8_prequant_us'):
        values=[r['latency_change_percent'][metric] for r in rows if r['model']==model]
        print(model, metric, 'change percent min/max', min(values), max(values))
