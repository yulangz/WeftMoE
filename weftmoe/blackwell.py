"""Single-node BF16 Blackwell PF and DT, using canonical expert parameters and NVLink."""

import weakref

import torch
import triton
import triton.language as tl


def _validate_expert_weights(w1, w2):
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape != (w2.size(0), 2 * w2.size(2), w2.size(1)):
        raise ValueError("Expected canonical expert weights [E,2I,H] and [E,H,I]")
    if w1.device != w2.device or any(
        not w.is_cuda or w.dtype != torch.bfloat16 or not w.is_contiguous() for w in (w1, w2)
    ):
        raise ValueError("Expected contiguous CUDA BF16 weights on the same device")
    if min(w1.shape) <= 0 or w2.size(1) % 128 or w2.size(2) % 128:
        raise ValueError("Expected positive expert dimensions and H and I divisible by 128")


def _weight_tile_shape(experts, rows, columns, width):
    # Columns and width are storage elements (packed bytes for E2M1).
    if rows % 128 or columns % width:
        raise ValueError(f"Expert matrix [{rows},{columns}] must contain whole 128x{width} weight tiles")
    return experts, rows // 128, columns // width, 128, width


def _bf16_weight_views(w1, w2):
    e, h, i = w2.shape
    # Each FC1 tile holds eight groups of eight gate rows and eight up rows.
    gate_up = w1.view(e, 2, i // 64, 8, 8, h // 64, 64).permute(0, 2, 5, 3, 1, 4, 6)
    down = w2.view(e, h // 128, 128, i // 64, 64).transpose(2, 3)
    return gate_up, down


@torch.no_grad()
def prepare_bf16_weights(w1, w2):
    """Snapshot canonical weights into contiguous 128x64 tiles for single-node SM103 inference."""
    _validate_expert_weights(w1, w2)
    gate_up, down = _bf16_weight_views(w1, w2)
    return gate_up.contiguous().view(_weight_tile_shape(*w1.shape, 64)), down.contiguous()


@triton.jit
def _sorted_route_keys(ids, slots, experts: tl.constexpr, block: tl.constexpr):
    # Invalid routes sort last. The original slot breaks ties between experts.
    return tl.sort(tl.where((ids >= 0) & (ids < experts), ids, experts) * block + slots)


@triton.jit
def _stage_inputs(
    X,
    Routes,
    Gates,
    SX,
    SR,
    SG,
    tokens,
    H: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, BH)
    offset = rows[:, None] * H + columns[None, :]
    live = (rows[:, None] < tokens) & (columns[None, :] < H)
    tl.store(SX + offset, tl.load(X + offset, live, 0), live)

    # The combine warp visits slots in ascending order. Sort (expert, original
    # slot) before dispatch so that its FP32 reduction follows the PF contract.
    slots = tl.arange(0, BK)
    indices = rows[:, None] * K + slots[None, :]
    valid = (rows[:, None] < tokens) & (slots[None, :] < K)
    ids = tl.load(Routes + indices, valid, -1)
    weights = tl.load(Gates + indices, valid, 0.0)
    ordered = _sorted_route_keys(ids, slots[None, :], E, BK)
    expert = ordered // BK
    tl.store(SR + indices, tl.where(expert < E, expert, -1), valid)
    tl.store(SG + indices, tl.gather(weights, ordered % BK, axis=1), valid)


class BlackwellRuntime:
    """One symmetric workspace, serialized on the supplied CUDA stream."""

    def __init__(
        self, group, experts, top_k, hidden, intermediate, capacity, stream, training, max_saved, *, fp4=False, compute_sms=None, decode_only=False, decode_plan=None,
    ):
        from .blackwell_cutlass import ExpertWorkspace, launch

        # A legacy default stream cannot join a caller's CUDA Graph capture.
        self.stream = (
            torch.cuda.Stream(device=stream.device)
            if not training and stream == torch.cuda.default_stream(stream.device)
            else stream
        )
        self.training = training
        self.max_saved = max_saved
        self.tapes = weakref.WeakSet()
        self.launch = launch
        self.buffer = ExpertWorkspace(
            group,
            experts,
            max(1, capacity),
            top_k,
            hidden,
            intermediate,
            fp4=fp4, compute_sms=compute_sms, decode_only=decode_only, decode_plan=decode_plan,
        )

        if training:
            self.wgrad_stream = None if fp4 else torch.cuda.Stream(device=stream.device)
            self.alignment = 128
            self.peers = torch.tensor(self.buffer.handle.buffer_ptrs, dtype=torch.int64, device=stream.device)

    def _run(self, x, routes, gates, w1, w2, *, save=False):
        caller = torch.cuda.current_stream()
        if caller.device != self.stream.device or x.device != self.stream.device:
            raise ValueError("Blackwell workspace and tensors must share the current device")
        if caller != self.stream:
            self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            b = self.buffer
            n, h = x.shape
            if n:
                _stage_inputs[(triton.cdiv(n, 4),)](
                    x,
                    routes,
                    gates,
                    b.x,
                    b.topk_idx,
                    b.topk_weights,
                    n,
                    h,
                    b.num_experts,
                    b.num_topk,
                    triton.next_power_of_2(b.num_topk),
                    triton.next_power_of_2(h),
                    4,
                )
            y = torch.empty_like(x)
            if save:
                from .blackwell_backward import Tape, checkpoint

                if len(self.tapes) >= self.max_saved:
                    raise RuntimeError("Blackwell training tape capacity exhausted")
                buffers, ends = checkpoint(b, n, x.device, self.alignment)
                self.launch(y, w1, w2, b, saved=buffers)
                tape = Tape(y, (routes.clone(), w1, w2, *buffers[2:], ends))
                self.tapes.add(tape)
            else:
                self.launch(y, w1, w2, b)
            if caller != self.stream:
                for tensor in (x, routes, gates, w1, w2):
                    tensor.record_stream(self.stream)
                y.record_stream(caller)
        if caller != self.stream:
            caller.wait_stream(self.stream)
        return tape if save else y

    def forward_async(self, x, routes, gates, w1, w2, retain_backward_state=True):
        from .blackwell_backward import Tape

        if self.training and retain_backward_state:
            return self._run(x, routes, gates, w1, w2, save=True)
        return Tape(self._run(x, routes, gates, w1, w2))

    def forward(self, x, routes, gates, w1, w2):
        tape = self.forward_async(x, routes, gates, w1, w2)
        self.stream.synchronize()
        return tape

    def backward_async(self, tape, dy):
        from .blackwell_backward import gradients

        return gradients(self, tape, dy)

    def backward(self, tape, dy):
        result = self.backward_async(tape, dy)
        self.stream.synchronize()
        return result

    def usage(self):
        return dict(notify_slots_in_use=len(self.tapes), symmetric_bytes=self.buffer.buffer.nbytes)

    def session(self):
        return None

    def graph_forward(self, layer, x, routes, gates, w1, w2, caller_stream):
        return self._run(x, routes, gates, w1, w2)

    def eager_inference_async(self, x, routes, gates, w1, w2, caller_stream, layer=None):
        return self._run(x, routes, gates, w1, w2)

    def eager_inference(self, x, routes, gates, w1, w2, caller_stream, layer=None):
        output = self._run(x, routes, gates, w1, w2)
        self.stream.synchronize()
        return output

    def invalidate_inference_layer(self, layer, caller_stream):
        self.stream.synchronize()

    def drain_eager_inference(self):
        self.stream.synchronize()

    def drain_training(self):
        self.stream.synchronize()

    def retire_peer_mappings(self):
        self.buffer.destroy()
