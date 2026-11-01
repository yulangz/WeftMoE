"""Summarize paired PF BWD timing and exact-output/DT parity checks."""
import collections
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'build/pf-stage-release-20261101'
MODELS = ('qwen35b', 'qwen122b', 'flash')

def records(path, prefix):
    text = path.read_text()
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(text[m.end():])[0]
            for m in re.finditer(re.escape(prefix), text)]

checks, parity, timings = [], [], []
for model in MODELS:
    hashes = {}
    for label in ('baseline', 'early'):
        path = BASE / f'{label}-{model}-check.log'
        if not path.exists():
            continue
        rows = records(path, 'FINGERPRINT ')
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r['rank'], r['name']].append(r['sha256'])
        hashes[label] = dict(groups)
    if len(hashes) == 2:
        checks.append(dict(model=model, comparisons=sum(map(len, hashes['early'].values())),
            all_bitwise_equal=hashes['baseline'] == hashes['early']))
    path = BASE / f'pf-dt-{model}.log'
    if path.exists():
        rows = records(path, 'PF_DT ')
        for fmt in ('bf16', 'fp8', 'prequant'):
            group = [r for r in rows if r['format'] == fmt]
            if group:
                parity.append(dict(model=model, format=fmt, cases=len(group),
                    bitwise_equal_cases=sum(r['bitwise_equal'] for r in group),
                    max_diff=max(r['diff'] for r in group),
                    worst=max(group, key=lambda r: r['diff'])))
for path in sorted(BASE.glob('bwd-*/results.json')):
    rows = json.loads(path.read_text())['records']
    for model in MODELS:
        for n in (8192, 20480):
            group = {r['label']: r['result']['wall_slowest_rank_p50_us']
                     for r in rows if r['model'] == model and r['tokens'] == n}
            if {'baseline', 'early'} <= group.keys():
                baseline = statistics.mean([group['baseline'], group['baseline_after']]) if 'baseline_after' in group else group['baseline']
                timings.append(dict(round=path.parent.name, model=model, tokens=n,
                    baseline_us=group['baseline'], baseline_repeat_us=group.get('baseline_after'),
                    baseline_mean_us=baseline, early_us=group['early'],
                    latency_reduction_percent=100*(1-group['early']/baseline)))
report = dict(source_id='weftmoe-pf-stage-release-20261101', gpu='8x NVIDIA L20X', world=8,
    change='Release operand stage after dR row-to-column, before independent X row-to-column',
    numerical_policy='BF16 global diff < 5e-3 except finite-only dprob; no waiver in PF tests',
    exact_output_checks=checks, pf_dt_forward=parity,
    pf_dt_workload=dict(experts=16, topk=2, tokens=[1,16,65,129,512],
        extra_case='rank0 empty; ranks1..7 have 18..24 tokens',
        routing=['distributed', 'all-to-one with duplicate expert routes'],
        weight_seed=1729, input_seed='901+rank',
        scope='Forward only; DT has no backward implementation'),
    backward_timings=timings,
    timing_method='Unprofiled per-step slowest-rank wall median, online FP8, BF16 dW; weight preparation excluded; round1 baseline/candidate/baseline, round2 candidate/baseline')
report['decision'] = ('Rejected: all six shapes were slower in both measurement orders; production BWD unchanged'
    if len(timings) == 12 and all(r['latency_reduction_percent'] < 0 for r in timings) else 'Measurements incomplete or require review')
out = ROOT / 'docs/benchmarks/pf_stage_release_20261101.json'
out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
