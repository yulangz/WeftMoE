#!/usr/bin/env python3
"""Independent FP64 oracle for the main H=7168/I=3072 Forward fixture.

This module does not import MegaMOE, Megatron, TransformerEngine, or PyTorch.
It rebuilds the documented BF16 input/weight/gate fixture with NumPy, performs
FP64 dot products, applies the documented BF16 phase casts, and checks selected
output columns plus the complete cached FC1 A/B checkpoint.
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


MAX_ABS = 0.02
MAX_RMSE = 0.008
RELATIVE_FLOOR = 0.01
MAX_RELATIVE = 0.10
HIDDEN = 7168
INTERMEDIATE = 3072
ROUTES = ((1, 68, 135, 202, 269, 336),
          (145, 289, 10, 90, 230, 350))
OUTPUT_COLUMNS = (0, 1, 17, 511, 1024, 4095, 7167)
GATE_SEED = 0x4D4547414D4F4547


def mix32(value):
    value = np.asarray(value, dtype=np.uint32)
    value ^= value >> np.uint32(16)
    value *= np.uint32(0x7FEB352D)
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x846CA68B)
    value ^= value >> np.uint32(16)
    return value


def weight(expert, kind, rows, columns):
    rows = np.asarray(rows, dtype=np.uint32)
    columns = np.asarray(columns, dtype=np.uint32)
    key = (mix32(np.uint32(expert) ^ np.uint32(0x9E3779B9)) ^
           mix32(np.uint32(kind) ^ np.uint32(0x85EBCA6B)) ^
           mix32(rows ^ np.uint32(0xC2B2AE35)) ^
           mix32(columns ^ np.uint32(0x27D4EB2F)))
    return ((mix32(key) & np.uint32(1)).astype(np.int32) * 2 - 1).astype(
        np.float64) / 256.0


def float32_to_bf16_bits(values):
    fp32 = np.asarray(values, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & 1)
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_float64(bits):
    expanded = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return expanded.view(np.float32).astype(np.float64)


def quantize_bf16(values):
    return bf16_bits_to_float64(float32_to_bf16_bits(values))


def selected_gates(token):
    scores = np.array([
        1 + ((GATE_SEED + token * 1315423911 + slot * 2654435761) %
             65521)
        for slot in range(len(ROUTES[token]))], dtype=np.float64)
    # C++ stores the double division in FP32.
    return (scores / scores.sum()).astype(np.float32)


def sampled_weight_hash(expert, kind):
    """FNV-1a over public, canonical coordinates for fixture evidence."""
    if kind == 1:
        coordinates = (((sample * 7919 + 5) % (2 * INTERMEDIATE),
                        (sample * 104729 + 17) % HIDDEN)
                       for sample in range(32))
    else:
        coordinates = (((sample * 65537 + 11) % HIDDEN,
                        (sample * 8191 + 3) % INTERMEDIATE)
                       for sample in range(32))
    result = 1469598103934665603
    for row, column in coordinates:
        bits = int(float32_to_bf16_bits(weight(
            expert, kind, np.array([row]), np.array([column])))[0])
        for byte in struct.pack("<H", bits):
            result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return result


def fc1(expert, activation):
    accum = np.empty(2 * INTERMEDIATE, dtype=np.float64)
    columns = np.arange(HIDDEN, dtype=np.uint32)[None, :]
    for begin in range(0, 2 * INTERMEDIATE, 64):
        end = min(2 * INTERMEDIATE, begin + 64)
        rows = np.arange(begin, end, dtype=np.uint32)[:, None]
        accum[begin:end] = weight(expert, 1, rows, columns) @ activation
    a = accum[:INTERMEDIATE]
    b = accum[INTERMEDIATE:]
    # FP64 is the high-precision oracle; production uses Hopper __expf at this
    # documented location and then casts S once to BF16.
    s = quantize_bf16((a / (1.0 + np.exp(-a))) * b)
    return accum, s


def expert_selected_output(expert, activation):
    accum, s = fc1(expert, activation)
    rows = np.asarray(OUTPUT_COLUMNS, dtype=np.uint32)[:, None]
    columns = np.arange(INTERMEDIATE, dtype=np.uint32)[None, :]
    y = weight(expert, 2, rows, columns) @ s
    return accum, quantize_bf16(y)


def metrics(actual, reference, coordinates):
    error = np.abs(actual - reference)
    worst = int(np.argmax(error))
    eligible = np.abs(reference) >= RELATIVE_FLOOR
    relative = np.zeros_like(error)
    relative[eligible] = error[eligible] / np.abs(reference[eligible])
    return {
        "max_abs": float(error.max(initial=0.0)),
        "rmse": float(math.sqrt(np.mean(error * error))),
        "max_relative_above_floor": float(relative.max(initial=0.0)),
        "worst_coordinate": coordinates[worst],
        "worst_actual": float(actual[worst]),
        "worst_reference": float(reference[worst]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    activations = [
        (((token * HIDDEN + np.arange(HIDDEN, dtype=np.int64)) % 127) - 63)
        .astype(np.float64) / 64.0 for token in range(len(ROUTES))]
    expert_results = {}
    for token, token_routes in enumerate(ROUTES):
        for expert in token_routes:
            expert_results[expert] = expert_selected_output(
                expert, activations[token])
    combined_reference = []
    for token, token_routes in enumerate(ROUTES):
        gates = selected_gates(token)
        combined = np.zeros(len(OUTPUT_COLUMNS), dtype=np.float32)
        for expert in sorted(token_routes):
            slot = token_routes.index(expert)
            combined = combined + (
                gates[slot] * expert_results[expert][1]).astype(np.float32)
        combined_reference.append(quantize_bf16(combined))
    combined_reference = np.stack(combined_reference)
    output_bits = np.fromfile(args.output, dtype="<u2")
    if output_bits.size != len(ROUTES) * HIDDEN:
        raise RuntimeError(
            f"output elements expected={len(ROUTES) * HIDDEN} "
            f"actual={output_bits.size}")
    output_bits = output_bits.reshape(len(ROUTES), HIDDEN)
    actual_output = bf16_bits_to_float64(
        output_bits[:, np.asarray(OUTPUT_COLUMNS)])
    selected_unique_values = np.unique(
        output_bits[:, np.asarray(OUTPUT_COLUMNS)]).size
    first_token_full_unique_values = np.unique(output_bits[0]).size
    if selected_unique_values < len(OUTPUT_COLUMNS):
        raise RuntimeError(
            "selected oracle outputs are insufficiently diverse: "
            f"unique={selected_unique_values}")
    if first_token_full_unique_values <= 1:
        raise RuntimeError("7168-column output collapsed to one value")
    output_metrics = metrics(
        actual_output.reshape(-1), combined_reference.reshape(-1),
        [f"output[{token},{column}]" for token in range(len(ROUTES))
         for column in OUTPUT_COLUMNS])

    cache_bits = np.fromfile(args.cache, dtype="<u2")
    canonical_experts = sorted(expert_results)
    expected_cache_elements = len(canonical_experts) * 2 * INTERMEDIATE
    if cache_bits.size != expected_cache_elements:
        raise RuntimeError(
            f"cache elements expected={expected_cache_elements} "
            f"actual={cache_bits.size}")
    # Download schema is all A rows followed by all B rows.
    actual_cache = bf16_bits_to_float64(cache_bits).reshape(
        2, len(canonical_experts), INTERMEDIATE)
    cache_reference = np.stack(
        [expert_results[expert][0] for expert in canonical_experts])
    reference_cache = np.stack((cache_reference[:, :INTERMEDIATE],
                                cache_reference[:, INTERMEDIATE:]))
    cache_coordinates = [
        f"cache[{plane},{route},{column}]"
        for plane in ("A", "B")
        for route in range(len(canonical_experts))
        for column in range(INTERMEDIATE)]
    cache_metrics = metrics(actual_cache.reshape(-1),
                            reference_cache.reshape(-1), cache_coordinates)
    passed = (
        output_metrics["max_abs"] <= MAX_ABS and
        output_metrics["rmse"] <= MAX_RMSE and
        output_metrics["max_relative_above_floor"] <= MAX_RELATIVE and
        cache_metrics["max_abs"] <= MAX_ABS and
        cache_metrics["rmse"] <= MAX_RMSE and
        cache_metrics["max_relative_above_floor"] <= MAX_RELATIVE)
    sample_hashes = {
        f"expert_{expert}_{tensor}": sampled_weight_hash(expert, kind)
        for expert in (0, 1, 47, 48, 191, 383)
        for tensor, kind in (("w1", 1), ("w2", 2))
    }
    if len(set(sample_hashes.values())) != len(sample_hashes):
        raise RuntimeError("expert-aware W1/W2 fixture sample hashes collided")
    result = {
        "schema": 1,
        "oracle": "independent NumPy FP64; BF16 phase casts",
        "shape": {"tokens": len(ROUTES), "hidden": HIDDEN,
                  "intermediate": INTERMEDIATE, "global_experts": 384,
                  "top_k": len(ROUTES[0])},
        "routes": [list(routes) for routes in ROUTES],
        "output_columns": list(OUTPUT_COLUMNS),
        "thresholds_fixed_before_measurement": {
            "max_abs": MAX_ABS, "rmse": MAX_RMSE,
            "relative_floor": RELATIVE_FLOOR,
            "max_relative": MAX_RELATIVE},
        "output": output_metrics,
        "output_diversity": {
            "selected_coordinate_unique_bf16_values":
                int(selected_unique_values),
            "first_token_7168_column_unique_bf16_values":
                int(first_token_full_unique_values),
            "selected_tokens": len(ROUTES),
            "selected_expert_routes": sum(len(routes) for routes in ROUTES),
        },
        "fc1_cache": cache_metrics,
        "expert_weight_sample_hashes": sample_hashes,
        "passed": passed,
    }
    print("FORWARD_FP64_REFERENCE_JSON " + json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
