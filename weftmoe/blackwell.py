"""Single-node BF16 Blackwell PF, using canonical expert parameters and NVLink."""

import weakref

import torch
import triton
import triton.language as tl


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
    key = tl.where((ids >= 0) & (ids < E), ids, E) * BK + slots[None, :]
    ordered = tl.sort(key, descending=False)
    expert = ordered // BK
    tl.store(SR + indices, tl.where(expert < E, expert, -1), valid)
    tl.store(SG + indices, tl.gather(weights, ordered % BK, axis=1), valid)


class BlackwellRuntime:
    """One symmetric workspace, serialized on the supplied CUDA stream."""

    def __init__(self, group, experts, top_k, hidden, intermediate, capacity, stream, training, max_saved):
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
        )

        if training:
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
