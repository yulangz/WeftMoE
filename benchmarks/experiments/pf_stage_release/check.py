"""Run the full PF suite and fingerprint all BF16-reference comparisons."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('pf_check', ROOT / 'tests/torchrun_fp8.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
compare = suite.compare

def checked(name, actual, expected, rank):
    compare(name, actual, expected, rank)
    raw = actual.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    print('FINGERPRINT ' + json.dumps(dict(rank=rank, name=name, shape=list(actual.shape),
          sha256=hashlib.sha256(raw).hexdigest())), flush=True)

suite.compare = checked
suite.main()
