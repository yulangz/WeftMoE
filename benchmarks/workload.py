"""Reproducible varying-token route traces shared by all MoE benchmarks."""

import hashlib
import math
import random

import torch


def balanced_routes(tokens, experts, top_k, device, generator):
    """Randomize complete balanced periods; only the final period is partial."""
    period = experts // math.gcd(experts, top_k)
    blocks = math.ceil(tokens / period)
    permutations = torch.rand(blocks, experts, device=device, generator=generator).argsort(dim=-1)
    slots = torch.arange(period * top_k, device=device) % experts
    routes = permutations[:, slots].reshape(-1, top_k)[:tokens]
    return routes[torch.randperm(tokens, device=device, generator=generator)]


def make_workload(args, nominal_tokens, rank, device, route_distribution="random"):
    """Freeze a reproducible CPU route trace, without keeping activations live."""
    if getattr(args, "route_trace", None):
        path = args.route_trace.format(rank=rank)
        routes = torch.load(path, map_location="cpu", weights_only=True)
        gates = torch.load(path.removesuffix(".routes.pt") + ".gates.pt",
                           map_location="cpu", weights_only=True)
        assert len(routes) == len(gates) and routes
        trace = list(zip(routes, gates))
        assert all(r.shape == g.shape and r.shape[1] == args.top_k for r, g in trace)
        return [trace[i % len(trace)] for i in range(args.warmups + args.samples)]
    seed = args.seed + nominal_tokens * 1000003
    lengths = random.Random(seed + rank)
    routes_rng = torch.Generator(device=device).manual_seed(seed + rank)
    # A common per-step bias creates expert hotspots shared across source ranks.
    # A separate stream keeps it identical despite unequal token counts.
    bias_rng = torch.Generator(device=device).manual_seed(seed + 1000000007)
    balanced_rng = torch.Generator(device=device).manual_seed(seed + rank + 1000000009)
    sigma = nominal_tokens * args.token_std_fraction
    low = max(1, math.ceil(nominal_tokens - 3 * sigma))
    high = math.floor(nominal_tokens + 3 * sigma)
    batches = []
    for _ in range(args.warmups + args.samples):
        tokens = round(lengths.gauss(nominal_tokens, sigma))
        while not low <= tokens <= high:
            tokens = round(lengths.gauss(nominal_tokens, sigma))
        logits = torch.randn(tokens, args.experts, device=device, generator=routes_rng)
        logits.add_(torch.randn(args.experts, device=device, generator=bias_rng),
                    alpha=args.route_bias_std)
        scores, routes = logits.topk(args.top_k, dim=-1)
        if route_distribution == "balanced":
            # Keep lengths, selected gate scores and the input RNG unchanged.
            routes = balanced_routes(tokens, args.experts, args.top_k, device, balanced_rng)
        batches.append((routes.int().cpu(), scores.softmax(dim=-1).cpu()))
    return batches


def workload_sha256(batches):
    """Hash CPU route traces, including each variable-length batch boundary."""
    digest = hashlib.sha256()
    for routes, gates in batches:
        digest.update(len(routes).to_bytes(8, "little"))
        digest.update(routes.numpy().tobytes())
        digest.update(gates.numpy().tobytes())
    return digest.hexdigest()


def decode_routes(args, logits):
    """Yield global decode routes for capture, parity, warmup and timed steps."""
    tokens, experts = logits.shape
    rng = torch.Generator(device=logits.device).manual_seed(args.seed + tokens)
    bias = torch.randn(experts, device=logits.device, generator=rng) * args.route_bias_std
    while True:
        scores, ids = logits.normal_(generator=rng).add_(bias).topk(args.top_k, dim=-1)
        yield ids.int(), scores.softmax(-1)
