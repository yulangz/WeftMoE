"""Replay a measured route trace and export representative hot-rank GEMM shapes.

Run without concurrent GPU benchmarks. The workload digest and every received
row count must match the input log; selection uses row counts, never timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_training import make_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(line.split(" ", 1)[1])
               for line in args.log.read_text().splitlines()
               if line.startswith("TRAINING_CYCLE_JSON ")]
    for report in reports:
        tokens, world = report["nominal_tokens_per_rank"], report["world"]
        workload = SimpleNamespace(**report)
        counts = torch.zeros(report["warmups"] + report["samples"],
                             report["experts"], dtype=torch.int64)
        for rank in range(world):
            batches = make_workload(workload, tokens, rank, "cuda",
                                    report["route_distribution"])
            digest = hashlib.sha256()
            for step, (routes, gates) in enumerate(batches):
                digest.update(routes.shape[0].to_bytes(8, "little"))
                digest.update(routes.numpy().tobytes())
                digest.update(gates.numpy().tobytes())
                counts[step] += torch.bincount(routes.flatten().long(),
                                                minlength=report["experts"])
            assert digest.hexdigest() == report["workload_sha256_per_rank"][rank]
        counts = counts[report["warmups"]:]
        per_rank = counts.reshape(report["samples"], world, -1).sum(-1)
        assert per_rank.T.tolist() == report["per_rank_received_rows"]
        # Lower median when the trace has an even number of samples.
        hot_rows, hot_ranks = per_rank.max(-1)
        step = int(hot_rows.argsort(stable=True)[(report["samples"] - 1) // 2])
        rank = int(hot_ranks[step])
        selected = counts[step].reshape(world, -1)[rank].tolist()
        stem = args.output / f"ep{world}_t{tokens}"
        stem.with_suffix(".rows").write_text("\n".join(map(str, selected)) + "\n")
        metadata = dict(source=str(args.log.resolve()), world=world, tokens=tokens,
                        timed_step=step, rank=rank, rows=selected,
                        sum_rows=sum(selected), workload_sha256_per_rank=
                        report["workload_sha256_per_rank"])
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{stem.name}: step={step} rank={rank} rows={sum(selected)}", flush=True)


if __name__ == "__main__":
    main()
