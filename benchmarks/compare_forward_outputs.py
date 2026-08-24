#!/usr/bin/env python3
"""Compare explicit MegaMOE and Megatron common-semantics Forward outputs."""

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
    args = parser.parse_args()

    count = args.tokens * args.hidden
    ranks = []
    all_equal = True
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
        reference = record["common_forward"].contiguous().view(-1)
        if reference.numel() != count or reference.dtype != torch.bfloat16:
            raise RuntimeError(f"rank={rank} malformed Megatron output")
        reference_bits = reference.view(torch.int16)
        equal = torch.equal(mega_bits, reference_bits)
        mega = mega_bits.view(torch.bfloat16).float()
        expected = reference.float()
        absolute = (mega - expected).abs()
        relative = absolute / expected.abs().clamp_min(1.0e-12)
        ranks.append({
            "rank": rank,
            "byte_identical": equal,
            "max_abs_error": absolute.max().item(),
            "max_relative_error": relative.max().item(),
        })
        all_equal = all_equal and equal
    result = {"schema": 1, "world": args.world,
              "byte_identical": all_equal, "ranks": ranks}
    print("FORWARD_OUTPUT_COMPARISON_JSON " +
          json.dumps(result, sort_keys=True))
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
