"""Deterministic block-16 NVFP4 grouped gradients for Blackwell."""


import torch
import triton

from .blackwell_backward import _gather_dy, _return, _swiglu
from .blackwell_cutlass import grouped_extension


def quantize(value, ends, scale, *, row=False, regular=0):
    rows, features = value.shape

    def allocate(enabled):
        if enabled:
            return tuple(
                torch.empty((rows, features // div), device=value.device, dtype=torch.uint8)
                for div in (2, 16)
            )
        return value, value

    d, s = allocate(row)
    dt, st = allocate(True)
    ext = grouped_extension("nvfp4", *torch.cuda.get_device_capability(value.device))
    ext.quantize_with_layer_scale(
        *(v.data_ptr() for v in (value, ends, d, s, dt, st)),
        rows,
        features,
        len(ends),
        regular,
        scale,
        row,
        torch.cuda.current_stream(value.device).cuda_stream,
    )
    return (d, s), (dt, st)


def grouped(a, b, out, ends, outer, cols, k_grouped, scale):
    device = out.device.index
    ext = grouped_extension("nvfp4", *torch.cuda.get_device_capability(device))
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    size = ext.workspace_bytes(len(ends), device, sms)
    space = torch.empty(size, device=out.device, dtype=torch.uint8)
    ext.run(
        *(v.data_ptr() for v in (*a, *b, out, ends)),
        len(ends),
        outer,
        cols,
        k_grouped,
        scale,
        device,
        sms,
        space.data_ptr(),
        size,
        torch.cuda.current_stream(device).cuda_stream,
    )


def expert_gradients(runtime, w1, w2, x, ab, gates, origins, ends):
    b = runtime.buffer
    rows, h = x.shape
    i, k, cap = b.intermediate_hidden, b.num_topk, b.num_max_tokens_per_rank
    offset = b.buffer.nbytes - cap * k * h * 2
    if b.group.size() > 1:
        b.handle.barrier()
    recv = torch.empty_like(x)
    grid = (triton.cdiv(rows, 4),)
    _gather_dy[grid](runtime.dy_peers, origins, recv, rows, 0, cap, k, h, triton.next_power_of_2(h), 4)
    # Requantize each transpose along its own reduction axis.
    sx, sw1, su, sw2, sdy, sdab = runtime.scales
    dyr, dyc = quantize(recv, ends, sdy, row=True)

    def transposed(w, scale):
        return quantize(w.flatten(0, 1), ends, scale, regular=w.shape[1])[1]

    du = torch.empty((rows, i), device=x.device, dtype=x.dtype)
    grouped(dyr, transposed(w2, sw2), du, ends, h, i, False, sdy * sw2)
    u, dab, eg = torch.empty_like(du), torch.empty_like(ab), torch.empty_like(gates)
    # At BF16 midpoints a one-ulp change before E2M1 quantization can move
    # a whole value to its neighboring FP4 code. Preserve the FP32 recipe.
    _swiglu[grid](
        ab, du, gates, u, dab, eg, rows, i, triton.next_power_of_2(i), 4, True, enable_fp_fusion=False
    )
    dr, dc = quantize(dab, ends, sdab, row=True)
    dx = torch.empty_like(x)
    grouped(dr, transposed(w1, sw1), dx, ends, 2 * i, h, False, sdab * sw1)
    _return[grid](
        runtime.peers,
        origins,
        dx,
        eg,
        rows,
        offset,
        b.topk_weights.data_ptr() - b.buffer.data_ptr(),
        cap,
        k,
        h,
        triton.next_power_of_2(h),
        4,
    )
    dw1, dw2 = torch.empty_like(w1), torch.empty_like(w2)
    compute = getattr(runtime, "wgrad_stream", None)
    if compute is not None:
        compute.wait_stream(torch.cuda.current_stream())
        for tensor in (*dc, *dyc, x, u, dw1, dw2, ends):
            tensor.record_stream(compute)
    with torch.cuda.stream(compute):
        grouped(dc, quantize(x, ends, sx)[1], dw1, ends, 2 * i, h, True, sdab * sx)
        grouped(dyc, quantize(u, ends, su)[1], dw2, ends, h, i, True, sdy * su)
    if b.group.size() > 1:
        b.handle.barrier()
    return b.buffer[offset:].view(torch.bfloat16), dw1, dw2
