import csv
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
records = json.loads((root / 'ep8/results.json').read_text())
assert len(records) == 42, len(records)
groups = {}
decode = {}
for record in records:
    parts = record['key'].split('-')
    mode, model = parts[:2]
    for row in record['results']:
        if mode == 'dt':
            key = (model, row['tokens'])
            decode.setdefault(key, []).append(row)
            continue
        dtype = parts[-1]
        assert row['world'] == 8 and row['experts'] == 256
        assert row['weight_grad_dtype'] == 'bf16'
        if mode == 'cycle':
            samples = row['wall_slowest_rank_samples_us']
        else:
            rank_samples = row['per_rank_' + mode + '_ready_us']
            samples = list(map(max, zip(*rank_samples)))
        assert len(samples) == 30
        key = (mode, model, row['nominal_tokens_per_rank'], dtype)
        groups.setdefault(key, []).append(samples)
summary = {'pf': [], 'dt': [], 'telemetry': {}}
for model in ['qwen35b', 'qwen122b', 'flash']:
    for tokens in [8192, 20480]:
        for mode in ['cycle', 'forward', 'backward']:
            result = dict(model=model, tokens=tokens, mode=mode)
            for dtype in ['bf16', 'fp8']:
                samples = groups[(mode, model, tokens, dtype)]
                assert len(samples) == 2
                result[dtype + '_ms'] = statistics.median(samples[0] + samples[1]) / 1000
                result[dtype + '_round_ms'] = [statistics.median(s) / 1000 for s in samples]
            result['latency_change_pct'] = (result['fp8_ms'] / result['bf16_ms'] - 1) * 100
            summary['pf'].append(result)
for (model, tokens), rows in decode.items():
    assert len(rows) == 2
    result = dict(model=model, tokens=tokens, max_rank_diff=max(r['fp8_vs_bf16_max_rank_diff'] for r in rows))
    for idx, key in enumerate(['bf16_us', 'fp8_online_us', 'fp8_prequant_us']):
        result[key] = statistics.median([s for r in rows for s in r['samples_us'][idx]])
        result[key + '_rounds'] = [r[key] for r in rows]
    result['online_latency_change_pct'] = (result['fp8_online_us'] / result['bf16_us'] - 1) * 100
    summary['dt'].append(result)
for path in sorted((root / 'ep8').glob('*.gpu.csv')):
    rows = list(csv.reader(path.open()))
    summary['telemetry'][path.stem] = dict(
        sm_clock_mhz=sorted({float(r[3]) for r in rows}),
        memory_clock_mhz=sorted({float(r[4]) for r in rows}),
        temperature_c=[min(float(r[5]) for r in rows), max(float(r[5]) for r in rows)],
        max_power_w=max(float(r[6]) for r in rows),
        max_memory_mib=max(float(r[7]) for r in rows),
    )
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k!='telemetry'},indent=2))
