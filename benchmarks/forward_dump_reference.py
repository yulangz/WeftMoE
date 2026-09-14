#!/usr/bin/env python3
"""Check sampled Forward dump coordinates against an independent NumPy oracle.

Reads the actual input, routes and gates. Rebuilds deterministic fixture weights
at the requested H/I/E shape, evaluates dot products in FP64, and applies the
documented BF16 phase casts and canonical FP32 reduction. No GPU is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forward_fp64_reference import bf16_bits_to_float64, quantize_bf16, weight


def reference(x, routes, gates, columns, intermediate, experts, world, pre_combine):
    hidden = len(x)
    input_columns = np.arange(hidden, dtype=np.uint32)[None, :]
    output_columns = np.asarray(columns, dtype=np.uint32)[:, None]
    k = np.arange(intermediate, dtype=np.uint32)[None, :]
    contributions = []
    for slot in np.argsort(routes, kind="stable"):
        expert = int(routes[slot])
        ab = np.empty(2 * intermediate, dtype=np.float64)
        for begin in range(0, len(ab), 128):
            end = min(begin + 128, len(ab))
            rows = np.arange(begin, end, dtype=np.uint32)[:, None]
            ab[begin:end] = weight(expert, 1, rows, input_columns) @ x
        ab = quantize_bf16(ab).astype(np.float32)
        a, b = np.split(ab, 2)
        sigmoid = np.float32(1) / (np.float32(1) + np.exp(-a))
        u = quantize_bf16(((a * sigmoid) * b) * np.float32(gates[slot]))
        y = quantize_bf16(weight(expert, 2, output_columns, k) @ u)
        contributions.append((expert // (experts // world), y.astype(np.float32)))
    if pre_combine:
        groups = {}
        for owner, y in contributions:
            groups.setdefault(owner, np.zeros(len(columns), dtype=np.float32))
            groups[owner] += y
        contributions = [(owner, quantize_bf16(y).astype(np.float32))
                         for owner, y in groups.items()]
    total = np.zeros(len(columns), dtype=np.float32)
    for _, y in contributions:
        total += y
    return quantize_bf16(total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 7])
    parser.add_argument("--tokens", type=int, nargs="+", default=[0, 1, 127, 8191])
    parser.add_argument("--pre-combine", action="store_true")
    parser.add_argument("--max-relative-l2", type=float, default=0.01)
    args = parser.parse_args()
    if min(args.hidden, args.intermediate, args.experts, args.world, args.top_k) <= 0:
        parser.error("dimensions must be positive")
    if args.experts % args.world or args.top_k > args.experts:
        parser.error("invalid expert partition or top-k")
    columns = sorted({0, 1, 17, args.hidden // 4, args.hidden // 2,
                      3 * args.hidden // 4, args.hidden - 1})
    observed, expected, coordinates = [], [], []
    for rank in args.ranks:
        if not 0 <= rank < args.world:
            parser.error("rank outside world")
        prefix = args.directory / f"weftmoe-rank-{rank}"
        def tensor(suffix, dtype, width):
            data = np.memmap(str(prefix) + suffix, dtype=dtype, mode="r")
            if data.size % width:
                raise ValueError(f"invalid dump size: {prefix}{suffix}")
            return data.reshape(-1, width)
        x = tensor(".input.bf16", "<u2", args.hidden)
        y = tensor(".bf16", "<u2", args.hidden)
        routes = tensor(".routes.i32", "<i4", args.top_k)
        gates = tensor(".gates.f32", "<f4", args.top_k)
        if not len(x) == len(y) == len(routes) == len(gates):
            raise ValueError("dump token counts disagree")
        for token in args.tokens:
            if not 0 <= token < len(x):
                parser.error("sample token outside dump")
            ref = reference(bf16_bits_to_float64(x[token]), routes[token],
                            gates[token], columns, args.intermediate,
                            args.experts, args.world, args.pre_combine)
            expected.extend(ref)
            observed.extend(bf16_bits_to_float64(y[token, columns]))
            coordinates.extend([[rank, token, column] for column in columns])
    actual, wanted = np.asarray(observed), np.asarray(expected)
    finite = bool(np.isfinite(actual).all() and np.isfinite(wanted).all())
    diff = actual - wanted
    relative_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(wanted), 1e-30))
    worst = int(np.argmax(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    # A norm bound plus an independent peak bound catches localized corruption.
    peak_bound = max(2e-7, 0.02 * float(np.max(np.abs(wanted))))
    passed = finite and relative_l2 <= args.max_relative_l2 and max_abs <= peak_bound
    result = dict(schema=1, passed=passed, finite=finite,
                  implementation="NumPy FP64 dot products with BF16 phase casts",
                  shape=dict(hidden=args.hidden, intermediate=args.intermediate,
                             experts=args.experts, top_k=args.top_k, world=args.world),
                  pre_combine=args.pre_combine, directory=str(args.directory),
                  coordinates=coordinates, reference=expected, observed=observed,
                  relative_l2=relative_l2, max_abs=max_abs,
                  rmse=float(np.sqrt(np.mean(diff * diff))),
                  worst_coordinate=coordinates[worst],
                  bounds=dict(relative_l2=args.max_relative_l2, max_abs=peak_bound))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "relative_l2", "max_abs")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
