"""Untimed, expert-at-a-time Torch reference for distributed MoE training."""

import torch
import torch.distributed as dist


def gather_rows(value, lengths):
    padded = value.new_zeros((max(lengths), *value.shape[1:]))
    padded[:value.shape[0]].copy_(value)
    peers = [torch.empty_like(padded) for _ in lengths]
    dist.all_gather(peers, padded)
    return torch.cat([peer[:count] for peer, count in zip(peers, lengths)])


def verify_step(runtime, x, routes, gates, w1, w2, dy):
    """Check all five results without replicating the global expert weights.

    Each rank evaluates its owned experts against all source tokens. FP32
    reduction combines their output/dX/dGate contributions. Only one expert's
    autograd graph is live at a time; this is not a performance benchmark.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    lengths = [None] * world
    dist.all_gather_object(lengths, x.shape[0])
    tape = runtime.forward(x, routes, gates, w1, w2)
    gradients = runtime.backward(tape, dy)
    if isinstance(gradients[2], (list, tuple)):
        gradients = (*gradients[:2], torch.stack(gradients[2]), torch.stack(gradients[3]))
    actual = (tape.output, *gradients)
    all_x, all_routes, all_gates, all_dy = (
        gather_rows(value, lengths) for value in (x, routes, gates, dy))
    output = torch.zeros_like(all_x, dtype=torch.float32)
    dx = torch.zeros_like(output)
    dg = torch.zeros_like(all_gates)
    dw1, dw2 = torch.zeros_like(w1), torch.zeros_like(w2)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(all_routes == rank * w1.shape[0] + expert)
        if not rows.numel():
            continue
        inputs = (all_x[rows].requires_grad_(),
                  all_gates[rows, slots, None].requires_grad_(),
                  w1[expert].detach().requires_grad_(), w2[expert].detach().requires_grad_())
        ex, eg, first, second = inputs
        a, b = (ex @ first.T).chunk(2, dim=-1)
        u = (torch.nn.functional.silu(a.float()) * b.float() * eg).to(x.dtype)
        y = u @ second.T
        gx, gg, g1, g2 = torch.autograd.grad(y, inputs, all_dy[rows])
        output.index_add_(0, rows, y.detach().float())
        dx.index_add_(0, rows, gx.float())
        dg[rows, slots] = gg.flatten()
        dw1[expert], dw2[expert] = g1, g2
    for value in (output, dx, dg):
        dist.all_reduce(value)
    begin = sum(lengths[:rank])
    owned_rows = slice(begin, begin + lengths[rank])
    expected = (output[owned_rows].to(x.dtype), dx[owned_rows].to(x.dtype),
                dg[owned_rows], dw1, dw2)
    errors = {}
    for name, value, target in zip(("output", "dx", "dgate", "dw1", "dw2"), actual, expected):
        value, target = value.float(), target.float()
        difference = value - target
        relative_l2 = (difference.norm() / target.norm().clamp_min(1e-30)).item()
        errors[name] = dict(relative_l2=relative_l2, max_abs=difference.abs().max().item())
        assert relative_l2 < 0.01, (rank, name, errors[name])
        torch.testing.assert_close(value, target, rtol=0.02, atol=1e-4,
                                   msg=lambda detail: f"rank={rank} tensor={name}\n{detail}")
    return errors
