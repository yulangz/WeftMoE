#!/usr/bin/env python3
"""Compare explicit MegaMOE and high-precision/common Forward outputs."""

import argparse
import json
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megamoe-dir", required=True)
    parser.add_argument("--megatron-dir", required=True)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--reference-key", default="common_forward",
                        choices=("common_forward", "native_forward"))
    parser.add_argument("--max-abs-tolerance", type=float, default=0.02)
    parser.add_argument("--rmse-tolerance", type=float, default=0.008)
    parser.add_argument("--relative-denominator-floor", type=float,
                        default=1.0e-3)
    args = parser.parse_args()

    count = args.tokens * args.hidden
    ranks = []
    within_tolerance = True
    for rank in range(args.world):
        mega_path = os.path.join(
            args.megamoe_dir, f"megamoe-rank-{rank}.bf16")
        megatron_path = os.path.join(
            args.megatron_dir, f"megatron-rank-{rank}.pt")
        if os.path.getsize(mega_path) != count * 2:
            raise RuntimeError(f"rank={rank} malformed MegaMOE output")
        mega_bits = torch.from_file(mega_path, dtype=torch.int16,
                                    size=count).clone()
        record = torch.load(megatron_path, map_location="cpu",
                            weights_only=True)
        reference = record[args.reference_key].contiguous().view(-1)
        if reference.numel() != count or reference.dtype != torch.bfloat16:
            raise RuntimeError(f"rank={rank} malformed Megatron output")
        reference_bits = reference.view(torch.int16)
        equal = torch.equal(mega_bits, reference_bits)
        mega = mega_bits.view(torch.bfloat16).float()
        expected = reference.float()
        absolute = (mega - expected).abs()
        relative = absolute / expected.abs().clamp_min(
            args.relative_denominator_floor)
        rmse = torch.sqrt(torch.mean((mega - expected) ** 2)).item()
        accepted = (absolute.max().item() <= args.max_abs_tolerance and
                    rmse <= args.rmse_tolerance)
        ranks.append({
            "rank": rank,
            "byte_identical": equal,
            "mismatched_elements": int(
                torch.count_nonzero(mega_bits != reference_bits).item()),
            "max_abs_error": absolute.max().item(),
            "max_relative_error": relative.max().item(),
            "rmse": rmse,
            "within_tolerance": accepted,
        })
        within_tolerance = within_tolerance and accepted
    result = {"schema": 1, "world": args.world,
              "reference_key": args.reference_key,
              "byte_identical": all(item["byte_identical"] for item in ranks),
              "within_tolerance": within_tolerance,
              "tolerance": {
                  "max_abs": args.max_abs_tolerance,
                  "rmse": args.rmse_tolerance,
                  "relative_denominator_floor":
                      args.relative_denominator_floor,
              },
              "ranks": ranks}
    print("FORWARD_OUTPUT_COMPARISON_JSON " +
          json.dumps(result, sort_keys=True))
    if not within_tolerance:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
