"""Manual tuning results: exact workloads and inclusive token intervals.

No measurements or searches are performed by this module or the runtime loader.
"""
import hashlib
import json
import os
from pathlib import Path
import re

KEY_FIELDS = ('gpu', 'interconnect', 'dtype', 'hidden', 'intermediate', 'experts',
              'top_k', 'nodes', 'gpus_per_node', 'input_layout', 'mode')
# Arithmetic, not CTA scheduling. These stay equal across every length bucket,
# training PF and DT. DT continues to use its fixed M64 MMA path.
ARITHMETIC_MACROS = (
    'WEFTMOE_FWD_MAIN_HIDDEN', 'WEFTMOE_FWD_MAIN_INTERMEDIATE',
    'WEFTMOE_FWD_TILE_M', 'WEFTMOE_FWD_FC1_TILE_N', 'WEFTMOE_FWD_FC2_TILE_N',
    'WEFTMOE_FWD_TILE_K', 'WEFTMOE_FWD_MATH_WARPGROUPS',
    'WEFTMOE_FWD_PRODUCER_WARPGROUPS', 'WEFTMOE_WGMMA_N256_ATOM_N',
    'WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS',
    'WEFTMOE_FWD_LOCAL_PREFILL_FAST_MATH', 'WEFTMOE_DT_FP8_SLICE_SCHEDULE',
    'WEFTMOE_FWD_LOCAL_PREFILL_STREAMING', 'WEFTMOE_DT_SLICE_SCHEDULE',
    'WEFTMOE_FWD_STATIC_V4_NETWORK_PREFILL_MIN_TOKENS',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def macros(header):
    return dict(re.findall(r'^#define\s+(\w+)\s+([^\n]+)', header, re.MULTILINE))


def arithmetic(header):
    values = macros(header)
    defaults = {'WEFTMOE_WGMMA_N256_ATOM_N': '256',
                'WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS': '0',
                'WEFTMOE_FWD_LOCAL_PREFILL_FAST_MATH': '0',
                'WEFTMOE_DT_FP8_SLICE_SCHEDULE': '1',
                'WEFTMOE_FWD_LOCAL_PREFILL_STREAMING': '0',
                'WEFTMOE_DT_SLICE_SCHEDULE': '0',
                'WEFTMOE_FWD_STATIC_V4_NETWORK_PREFILL_MIN_TOKENS': '0'}
    return {name: values.get(name, defaults.get(name)) for name in ARITHMETIC_MACROS}


def cache_root():
    return Path(os.environ.get('WEFTMOE_ROOT', '~/.cache/weftmoe')).expanduser().resolve()


def default_table():
    return Path(os.environ.get('WEFTMOE_AUTOTUNE_TABLE') or cache_root() / 'tuning.json')


def find_entry(path, key, tokens, *, source, canonical_header, batch_invariant=True):
    """Missing/stale/incompatible entries fall back; malformed tables fail loudly."""
    path = Path(path)
    if not path.exists():
        return None
    table = json.loads(path.read_text())
    if table.get('schema') != 2:
        raise ValueError('automatic compilation requires autotune schema 2; rerun tools/autotune.py')
    matches = []
    for entry in table['entries']:
        if entry['key'] != {k: key[k] for k in KEY_FIELDS} or entry['source'] != source:
            continue
        lower, upper = entry['tokens']
        if not (type(lower) is int and type(upper) is int and 0 <= lower <= upper):
            raise ValueError('invalid autotune token interval')
        if not lower <= tokens <= upper:
            continue
        if digest(entry['header']) != entry['header_digest']:
            raise ValueError('autotune header digest mismatch')
        if batch_invariant and (not entry.get('batch_invariant') or
                                arithmetic(entry['header']) != arithmetic(canonical_header)):
            continue
        matches.append(entry)
    if len(matches) > 1:
        raise ValueError('overlapping autotune token intervals for the same workload')
    return matches[0] if matches else None


def publish_entry(path, entry):
    """Replace one bucket atomically, retaining other models and precisions."""
    import fcntl
    import tempfile

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        table = json.loads(path.read_text()) if path.exists() else dict(schema=2, entries=[])
        if table.get('schema') != 2:
            raise ValueError('refusing to overwrite an old autotune table; choose a new --table path')
        entries = []
        for old in table['entries']:
            if old['key'] == entry['key']:
                if old['source'] != entry['source'] or old['tokens'] == entry['tokens']:
                    continue
                if max(old['tokens'][0], entry['tokens'][0]) <= min(old['tokens'][1], entry['tokens'][1]):
                    raise ValueError('overlapping autotune token intervals')
                if old.get('batch_invariant') and entry.get('batch_invariant') and arithmetic(old['header']) != arithmetic(entry['header']):
                    raise ValueError('batch-invariant buckets have different arithmetic')
            entries.append(old)
        table['entries'] = entries + [entry]
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
            json.dump(table, f, indent=2)
            temporary = Path(f.name)
        temporary.replace(path)


def select_profile(path, key, compiled_id):
    """Explicit prebuilt-library compatibility for existing schema-1 tables."""
    table = json.loads(Path(path).read_text())
    if table.get('schema') != 1:
        raise ValueError('explicit legacy profile selection requires schema 1')
    matches = [entry for entry in table['entries'] if entry['key'] == key]
    if len(matches) != 1:
        raise ValueError(f'expected one validated autotune entry for {key}')
    entry = matches[0]
    if entry['compiled_digest'] != compiled_id:
        raise ValueError('autotune profile requires a native rebuild')
    return {name: entry[name] for name in ('config_digest', 'window_tokens', 'contribution_windows')}
