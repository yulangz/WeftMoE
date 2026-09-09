#!/usr/bin/env python3
"""Compare every output bit across EP sizes, optionally checking dumped inputs.

For repartition mode, EP16 rank 2*r and 2*r+1 contain the first and second
halves of EP8 rank r. Prefix mode keeps the first eight source shards intact.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_tensor(directory, rank, suffix, dtype, tokens, width):
    path = directory / f"weftmoe-rank-{rank}{suffix}"
    expected = tokens * width * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes")
    return np.memmap(path, dtype=dtype, mode="r", shape=(tokens, width))


def compare(args):
    tensors = [("output", ".bf16", "<u2", args.hidden)]
    if args.check_inputs:
        tensors += [("input", ".input.bf16", "<u2", args.hidden),
                    ("routes", ".routes.i32", "<u4", args.top_k),
                    ("gates", ".gates.f32", "<u4", args.top_k)]
    if args.mode == "repartition" and args.tokens % 2:
        raise ValueError("repartition requires an even EP8 token count")
    records = []
    for name, suffix, dtype, width in tensors:
        for rank in range(args.ranks):
            left = read_tensor(args.left, rank, suffix, dtype, args.tokens, width)
            if args.mode == "repartition":
                halves = [read_tensor(args.right, rank * 2 + part, suffix,
                                      dtype, args.tokens // 2, width)
                          for part in range(2)]
            else:
                halves = [read_tensor(args.right, rank, suffix, dtype,
                                      args.tokens, width)]
            mismatch_count, first, offset = 0, None, 0
            left_hash, right_hash = hashlib.sha256(), hashlib.sha256()
            max_abs, squared = 0.0, 0.0
            for half in halves:
                for begin in range(0, len(half), 128):
                    b = half[begin:begin + 128]
                    a = left[offset + begin:offset + begin + len(b)]
                    left_hash.update(a.tobytes())
                    right_hash.update(b.tobytes())
                    mismatch = a != b
                    count = int(np.count_nonzero(mismatch))
                    mismatch_count += count
                    if count and first is None:
                        row, column = np.argwhere(mismatch)[0]
                        first = {"token": int(offset + begin + row),
                                 "column": int(column),
                                 "left_bits": int(a[row, column]),
                                 "right_bits": int(b[row, column])}
                    if name == "output":
                        af = (a.astype(np.uint32) << 16).view(np.float32)
                        bf = (b.astype(np.uint32) << 16).view(np.float32)
                        if not (np.isfinite(af).all() and np.isfinite(bf).all()):
                            raise ValueError("nonfinite Forward output")
                        diff = af.astype(np.float64) - bf
                        max_abs = max(max_abs, float(np.max(np.abs(diff))))
                        squared += float(np.sum(diff * diff))
                offset += len(half)
            record = {"tensor": name, "rank": rank,
                      "elements": int(left.size),
                      "mismatched_elements": mismatch_count,
                      "byte_identical": mismatch_count == 0,
                      "first_mismatch": first,
                      "left_sha256": left_hash.hexdigest(),
                      "right_sha256": right_hash.hexdigest()}
            if name == "output":
                record.update(max_abs=max_abs, rmse=(squared / left.size) ** 0.5)
            records.append(record)
    return {"schema": 1, "mode": args.mode, "tokens_per_left_rank": args.tokens,
            "left": str(args.left), "right": str(args.right),
            "byte_identical": all(r["byte_identical"] for r in records),
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefix", "repartition"), required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens, args.ranks, args.hidden, args.top_k) <= 0:
        parser.error("dimensions must be positive")
    result = compare(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"byte_identical": result["byte_identical"],
                      "output": str(args.output)}))
    return 0 if result["byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
