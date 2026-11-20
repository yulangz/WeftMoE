"""Validate GIN payloads and measure both directions of every physical bond member."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.sm100.gin_ports import PhysicalCounters
from weftmoe.blackwell_gin import RailWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if min(args.mib, args.iterations) < 1:
        parser.error("Payload size and iterations must be positive")
    local, lanes = int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    nodes, node = world // lanes, rank // lanes
    if nodes < 2:
        raise ValueError("The rail benchmark requires at least two nodes")
    stride = args.mib * 1024**2
    window = RailWindow(dist.group.WORLD, 2 * nodes * stride, nodes)
    src, dst = window.data.view(2, nodes, stride)
    sizes = torch.full((nodes,), stride, device=device, dtype=torch.int64)
    for peer in range(nodes):
        src[peer].fill_((1 + rank + peer * 17) % 256)
    for broadcast in (False, True):
        dst.zero_()
        window.exchange(0, nodes * stride, stride, sizes, broadcast=broadcast)
        for peer in range(nodes):
            if peer != node:
                expected = (1 + peer * lanes + local + 17 * (peer if broadcast else node)) % 256
                assert (dst[peer] == expected).all().item(), (rank, peer, broadcast)
    counters = PhysicalCounters() if local == 0 else None
    dist.barrier()
    torch.cuda.synchronize()
    before = counters.read() if counters is not None else None
    dist.barrier()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(args.iterations):
        window.exchange(0, nodes * stride, stride, sizes, broadcast=False)
    stop.record()
    stop.synchronize()
    elapsed = start.elapsed_time(stop) / 1000
    dist.barrier()
    if counters is not None:
        after = counters.read()
        print(
            "GIN_PORT_BYTES "
            + json.dumps(
                dict(
                    rank=rank,
                    tx_rx={port: [a - b for a, b in zip(pair, before[port])] for port, pair in after.items()},
                )
            ),
            flush=True,
        )
    times = [None] * world
    dist.all_gather_object(times, elapsed)
    if rank == 0:
        print(
            "GIN_BW_JSON "
            + json.dumps(
                dict(
                    ep=world,
                    backend=window.backend,
                    contexts=window.handle.contexts,
                    bytes_per_peer=stride,
                    iterations=args.iterations,
                    seconds=times,
                    slowest_payload_GBps=stride * (nodes - 1) * args.iterations / max(times) / 1e9,
                )
            ),
            flush=True,
        )
    del src, dst
    window.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
