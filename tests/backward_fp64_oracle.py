#!/usr/bin/env python3
"""Independent FP64 oracle for the production main-shape Backward kernel.

The executable supplies only selected result bytes.  This module independently
reconstructs BF16 X/dY/W1/W2 and evaluates FC1, SwiGLU backward, dX, dW and
dGate in NumPy FP64.  It intentionally imports no MegaMOE Python reference or
Megatron/TransformerEngine result.
"""

import argparse
import json
import os
import subprocess

import numpy as np


HIDDEN = 7168
INTERMEDIATE = 3072
ABS_LIMIT = 0.02
RMSE_LIMIT = 0.008
RELATIVE_FLOOR = 0.05
RELATIVE_LIMIT = 0.25


def mix32(value):
    value = np.asarray(value, dtype=np.uint32)
    value ^= value >> np.uint32(16)
    value *= np.uint32(0x7FEB352D)
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x846CA68B)
    value ^= value >> np.uint32(16)
    return value


def weight_block(expert, kind, rows, columns):
    key = mix32(np.uint32(expert) ^ np.uint32(0x9E3779B9))
    key = key ^ mix32(np.uint32(kind) ^ np.uint32(0x85EBCA6B))
    key = key ^ mix32(np.asarray(rows, dtype=np.uint32) ^
                      np.uint32(0xC2B2AE35))
    key = key ^ mix32(np.asarray(columns, dtype=np.uint32) ^
                      np.uint32(0x27D4EB2F))
    sign = ((mix32(key) & np.uint32(1)).astype(np.float64) * 2.0 - 1.0)
    return sign / 256.0


def bf16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & 1)
    return ((rounded >> np.uint32(16)) << np.uint32(16)).view(np.float32)


def bf16_from_bits(text):
    bits = np.asarray([int(value) for value in text.split(",")],
                      dtype=np.uint32)
    return (bits << np.uint32(16)).view(np.float32)


def fixture_vector(length, multiplier, modulus, center, divisor):
    index = np.arange(length, dtype=np.int64)
    return bf16(((index * multiplier) % modulus - center) / divisor).astype(
        np.float64)


def matvec_rows(expert, kind, output_rows, input_columns, vector,
                block_rows=128):
    result = np.empty(output_rows, dtype=np.float64)
    columns = np.arange(input_columns, dtype=np.uint32)[None, :]
    for begin in range(0, output_rows, block_rows):
        end = min(output_rows, begin + block_rows)
        rows = np.arange(begin, end, dtype=np.uint32)[:, None]
        result[begin:end] = weight_block(expert, kind, rows, columns) @ vector
    return result


def transposed_matvec(expert, kind, input_rows, output_columns, vector,
                      block_columns=128):
    result = np.empty(output_columns, dtype=np.float64)
    rows = np.arange(input_rows, dtype=np.uint32)[:, None]
    for begin in range(0, output_columns, block_columns):
        end = min(output_columns, begin + block_columns)
        columns = np.arange(begin, end, dtype=np.uint32)[None, :]
        result[begin:end] = vector @ weight_block(
            expert, kind, rows, columns)
    return result


def run_production(binary):
    command = [
        "python3", "tests/run_local_activation_p2p.py", "--binary", binary,
        "--world", "1", "--top-k", "1", "--backward-smoke",
        "--timeout", "600",
    ]
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True,
        env={**os.environ, "NCCL_DEBUG": "WARN"})
    marker = "BACKWARD_ORACLE_JSON "
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("production runner did not emit BACKWARD_ORACLE_JSON")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()

    actual = run_production(os.path.abspath(args.binary))
    x = fixture_vector(HIDDEN, 1, 127, 63, 64.0).astype(np.float64)
    dy = fixture_vector(HIDDEN, 7, 113, 56, 128.0).astype(np.float64)
    # world=1/top-k=1/token=0 routes to global expert 0 and has gate 1.
    ab_fp64 = matvec_rows(0, 1, 2 * INTERMEDIATE, HIDDEN, x)
    a = bf16(ab_fp64[:INTERMEDIATE]).astype(np.float64)
    b = bf16(ab_fp64[INTERMEDIATE:]).astype(np.float64)
    sigmoid = 1.0 / (1.0 + np.exp(-a))
    silu = a * sigmoid
    s = bf16(silu * b).astype(np.float64)
    u = bf16(s).astype(np.float64)
    dv = bf16(transposed_matvec(
        0, 2, HIDDEN, INTERMEDIATE, dy)).astype(np.float64)
    da = bf16(dv * b * sigmoid * (1.0 + a * (1.0 - sigmoid))).astype(
        np.float64)
    db = bf16(dv * silu).astype(np.float64)
    dab = np.concatenate((da, db))
    hidden_samples = np.asarray([0, 17, 1024, HIDDEN - 1], dtype=np.uint32)
    w1_columns = weight_block(
        0, 1, np.arange(2 * INTERMEDIATE, dtype=np.uint32)[:, None],
        hidden_samples[None, :])
    dx_reference = bf16(dab @ w1_columns).astype(np.float64)
    dgate_reference = float(np.sum(dv * s, dtype=np.float64))
    dw1_reference = bf16(np.asarray([da[0] * x[0], b[0] * 0.0 +
                                     db[0] * x[17]], dtype=np.float64))
    dw2_reference = bf16(np.asarray([dy[0] * u[0], dy[17] * u[17]],
                                    dtype=np.float64))
    ab_reference = np.asarray([a[0], a[17], b[0], b[17]])
    dab_reference = np.asarray([da[0], da[17], db[0], db[17]])

    actual_groups = {
        "dx": bf16_from_bits(actual["oracle_dx_bits"]).astype(np.float64),
        "dw1": bf16_from_bits(actual["oracle_dw1_bits"]).astype(np.float64),
        "dw2": bf16_from_bits(actual["oracle_dw2_bits"]).astype(np.float64),
        "ab": bf16_from_bits(actual["oracle_ab_bits"]).astype(np.float64),
        "dab": bf16_from_bits(actual["oracle_dab_bits"]).astype(np.float64),
        "dgate": np.asarray([float(actual["oracle_dgate"])], dtype=np.float64),
    }
    references = {
        "dx": dx_reference, "dw1": dw1_reference, "dw2": dw2_reference,
        "ab": ab_reference, "dab": dab_reference,
        "dgate": np.asarray([dgate_reference]),
    }
    errors = []
    worst = None
    for field, values in actual_groups.items():
        reference = references[field]
        delta = np.abs(values - reference)
        for index, error in enumerate(delta):
            errors.append(float(error))
            if worst is None or error > worst[0]:
                worst = (float(error), field, index, float(values[index]),
                         float(reference[index]))
    errors_array = np.asarray(errors)
    max_abs = float(np.max(errors_array))
    rmse = float(np.sqrt(np.mean(errors_array * errors_array)))
    relative = []
    for field, values in actual_groups.items():
        reference = references[field]
        mask = np.abs(reference) >= RELATIVE_FLOOR
        relative.extend((np.abs(values[mask] - reference[mask]) /
                         np.abs(reference[mask])).tolist())
    max_relative = max(relative, default=0.0)
    if max_abs > ABS_LIMIT or rmse > RMSE_LIMIT or \
            max_relative > RELATIVE_LIMIT:
        raise AssertionError(
            f"Backward FP64 oracle failed: max_abs={max_abs}, rmse={rmse}, "
            f"max_relative={max_relative}, worst={worst}")
    print("BACKWARD_FP64_ORACLE_JSON " + json.dumps({
        "max_abs": max_abs, "rmse": rmse,
        "max_relative_at_floor": max_relative,
        "relative_floor": RELATIVE_FLOOR,
        "limits": {"max_abs": ABS_LIMIT, "rmse": RMSE_LIMIT,
                   "max_relative": RELATIVE_LIMIT},
        "worst": {"field": worst[1], "index": worst[2],
                  "actual": worst[3], "reference": worst[4]},
        "sample_count": len(errors),
        "comparisons": {
            field: {"actual": actual_groups[field].tolist(),
                    "reference": references[field].tolist()}
            for field in sorted(actual_groups)
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
