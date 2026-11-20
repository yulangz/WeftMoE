"""Overlap two independent token windows on the rails with serialized NVLink GEMMs."""

import weakref

import torch
import triton
import triton.language as tl

from .blackwell import BlackwellRuntime
from .blackwell_backward import Tape
from .blackwell_multinode import MultiNodeRuntime, NVFP4MultiNodeRuntime
from .nvfp4 import prepare_nvfp4_weights


@triton.jit
def _merge_checkpoint(
    X,
    AB,
    G,
    Origin,
    End,
    OtherEnd,
    MX,
    MAB,
    MG,
    MO,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    BH: tl.constexpr,
    BI: tl.constexpr,
    SECOND: tl.constexpr,
    ORIGIN_OFFSET: tl.constexpr,
):
    row = tl.program_id(0)
    experts = tl.arange(0, BE)
    ends = tl.load(End + experts, experts < E, 0)
    expert = tl.sum(((row >= ends) & (experts < E)).to(tl.int32), 0)
    if SECOND:
        dest = row + tl.load(OtherEnd + expert)
    else:
        dest = row + tl.load(OtherEnd + expert - 1, expert > 0, 0)
    cols = tl.arange(0, BH)
    tl.store(
        MX + dest.to(tl.int64) * H + cols, tl.load(X + row.to(tl.int64) * H + cols, cols < H, 0), cols < H
    )
    ab_cols = tl.arange(0, BI)
    tl.store(
        MAB + dest.to(tl.int64) * 2 * I + ab_cols,
        tl.load(AB + row.to(tl.int64) * 2 * I + ab_cols, ab_cols < 2 * I, 0),
        ab_cols < 2 * I,
    )
    tl.store(MG + dest, tl.load(G + row))
    orig = tl.load(Origin + row)
    tl.store(MO + dest, tl.where(orig >= 0, orig + ORIGIN_OFFSET, -1))


class PipelinedMultiNodeRuntime(BlackwellRuntime):
    def __init__(
        self,
        group,
        experts,
        top_k,
        hidden,
        intermediate,
        capacity,
        stream,
        training,
        max_saved,
        nodes,
        nvfp4=False,
    ):
        self.training, self.max_saved = training, max_saved
        self.tapes = weakref.WeakSet()
        self.stream = (
            torch.cuda.Stream(device=stream.device)
            if not training and stream == torch.cuda.default_stream(stream.device)
            else stream
        )
        self.compute = torch.cuda.Stream(device=stream.device)
        compute_sms = (
            (torch.cuda.get_device_properties(stream.device).multi_processor_count - 4) // 4 * 4
        )
        self.split = (capacity + 1) // 2
        self.nvfp4 = nvfp4
        child_type = NVFP4MultiNodeRuntime if nvfp4 else MultiNodeRuntime
        self.windows = []
        for _ in range(2):
            worker = torch.cuda.Stream(device=stream.device)
            window = child_type(
                group, experts, top_k, hidden, intermediate, self.split, worker, training, max_saved, nodes,
                compute_sms=compute_sms,
            )
            window.rail.compact = True
            window.launch = self._serialize_compute(window.launch)
            self.windows.append(window)
        if training:
            self.buffer = self.windows[0].buffer
            self.peers = torch.cat([child.peers for child in self.windows])
            if nvfp4:
                self.dy_peers = torch.cat([child.dy_peers for child in self.windows])
                self.expert_gradients = self.windows[0].expert_gradients
            self.wgrad_stream = torch.cuda.Stream(device=stream.device)

    def _serialize_compute(self, launch):
        compute = self.compute

        def run(*args, **kwargs):
            transport = torch.cuda.current_stream()
            compute.wait_stream(transport)
            with torch.cuda.stream(compute):
                launch(*args, **kwargs)
            transport.wait_stream(compute)

        return run

    def _run(self, x, routes, gates, w1, w2, *, save=False, prepared=None, scales=None):
        if scales is None:
            scales = (self.scales.x, self.scales.intermediate) if self.nvfp4 else (1.0, 1.0)
        caller = torch.cuda.current_stream()
        if caller == self.stream:
            return self._run_windows(x, routes, gates, w1, w2, save, prepared, scales)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            result = self._run_windows(x, routes, gates, w1, w2, save, prepared, scales)
        caller.wait_stream(self.stream)
        (result.output if save else result).record_stream(caller)
        return result

    def _run_windows(self, x, routes, gates, w1, w2, save, prepared, scales):
        caller = torch.cuda.current_stream()
        if save and len(self.tapes) >= self.max_saved:
            raise RuntimeError("Blackwell training tape capacity exhausted")
        if self.nvfp4 and prepared is None:
            prepared = prepare_nvfp4_weights(w1, w2, weight_scales=(self.scales.w1, self.scales.w2))
        first, second = self.windows
        sections = (slice(None, self.split), slice(self.split, None))
        owners, outputs, saved = [], [], []
        for child, rows in zip(self.windows, sections):
            child.stream.wait_stream(caller)
            with torch.cuda.stream(child.stream):
                owners.append(child.stage_packet(x[rows], routes[rows], gates[rows], scales[0]))
        with torch.cuda.stream(first.stream):
            first.dispatch_packet()

        def dispatch_second(ready):
            # Queue both operations from the same ready event. The compute
            # grid reserves SMs for the remote put and its signal progress.
            second.stream.wait_event(ready)
            with torch.cuda.stream(second.stream):
                second.dispatch_packet()

        for child, rows, owner in zip(self.windows, sections, owners):
            with torch.cuda.stream(child.stream):
                result = child._run(
                    x[rows],
                    routes[rows],
                    gates[rows],
                    w1,
                    w2,
                    save=save,
                    owner=owner,
                    prepared=prepared,
                    scales=scales,
                    on_launch=dispatch_second if child is first else None,
                )
                if save:
                    saved.append(result)
                    result = result.output
                outputs.append(result)
        for child, output in zip(self.windows, outputs):
            caller.wait_stream(child.stream)
            output.record_stream(caller)
        output = torch.cat(outputs)
        if save:
            tape = Tape(output, tuple(saved))
            self.tapes.add(tape)
            return tape
        return output

    def backward_async(self, tape, dy):
        if tape not in self.tapes or tape.state is None:
            raise RuntimeError("Backward requires a live training tape from this runtime")
        from .blackwell_backward import expert_gradients

        if dy.shape != tape.output.shape or dy.dtype != torch.bfloat16 or dy.device != self.stream.device:
            raise ValueError("Backward requires BF16 dY matching the forward output")
        caller = torch.cuda.current_stream()
        states = [saved.state for saved in tape.state]
        pieces = (dy[: self.split], dy[self.split :])
        for child, piece in zip(self.windows, pieces):
            child.stream.wait_stream(caller)
            with torch.cuda.stream(child.stream):
                piece.record_stream(child.stream)
                child._stage_dy(piece)
        # Reassemble packed expert ranges once. Retain aligned zero padding
        # between windows, so all dW K reductions finish in a single GEMM.
        b = self.buffer
        alignment = self.windows[0].alignment
        ends = [((state[7] + alignment - 1) // alignment) * alignment for state in states]
        total = sum(len(state[3]) for state in states)
        merged = [t.new_empty((total, *t.shape[1:])) for t in states[0][3:7]]
        h, i = b.hidden, b.intermediate_hidden
        for index, state in enumerate(states):
            rows = len(state[3])
            if rows:
                _merge_checkpoint[(rows,)](
                    *state[3:7],
                    ends[index],
                    ends[1 - index],
                    *merged,
                    h,
                    i,
                    len(ends[index]),
                    triton.next_power_of_2(len(ends[index])),
                    triton.next_power_of_2(h),
                    triton.next_power_of_2(2 * i),
                    bool(index),
                    index * b.group.size() * b.num_max_tokens_per_rank * b.num_topk,
                )
        for child in self.windows:
            caller.wait_stream(child.stream)
        _, dw1, dw2 = getattr(self, "expert_gradients", expert_gradients)(
            self, *states[0][1:3], *merged, ends[0] + ends[1]
        )
        outputs = []
        for child, state, piece in zip(self.windows, states, pieces):
            child.stream.wait_stream(caller)
            with torch.cuda.stream(child.stream):
                local = child.buffer
                mailbox = local.buffer[-local.num_max_tokens_per_rank * local.num_topk * h * 2 :].view(
                    torch.bfloat16
                )
                outputs.append(child._return_gradients(state[0], state[8], state[9], mailbox, piece))
        caller.wait_stream(self.wgrad_stream)
        for child, saved, result in zip(self.windows, tape.state, outputs):
            caller.wait_stream(child.stream)
            for tensor in result:
                tensor.record_stream(caller)
            saved.state = None
            child.tapes.discard(saved)
        tape.state = None
        self.tapes.discard(tape)
        dx, dgate = zip(*outputs)
        return torch.cat(dx), torch.cat(dgate), dw1, dw2

    def usage(self):
        return dict(
            notify_slots_in_use=len(self.tapes),
            symmetric_bytes=sum(child.usage()["symmetric_bytes"] for child in self.windows),
            gin_bytes=sum(child.usage()["gin_bytes"] for child in self.windows),
            gin_backend=self.windows[0].rail.backend,
        )

    def retire_peer_mappings(self):
        for child in self.windows:
            child.retire_peer_mappings()
