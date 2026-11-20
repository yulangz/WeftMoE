"""Hierarchical Blackwell MoE. Each remote source row traverses the rail once.

NVLink delivers expert outputs into per-slot mailboxes. Pack only the slots
owned by this node for the return transfer, then reduce globally in expert order.
"""

import weakref

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .blackwell import BlackwellRuntime
from .nvfp4 import NVFP4Runtime, _encode_row, _pack_rows, prepare_nvfp4_weights, save_routed_x
from .blackwell_gin import RailWindow


@triton.jit
def _packet(
    X,
    Routes,
    Gates,
    Packet,
    Owner,
    Sizes,
    n,
    layer_scale,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    NODES: tl.constexpr,
    NODE: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    NVFP4: tl.constexpr,
):
    row = tl.program_id(0)
    payload: tl.constexpr = H // 2 + H // 16 if NVFP4 else 2 * H
    if row == 0:
        tl.store(Packet.to(tl.pointer_type(tl.int32)), n)
        tl.store(Sizes + NODE, 16 + n * (payload + 8 * K))
    if row < n:
        base = Packet + 16 + row.to(tl.int64) * (payload + 8 * K)
        col = tl.arange(0, BH)
        values = tl.load(X + row * H + col, col < H, 0)
        if NVFP4:
            quantized, block_sf = _encode_row(values, layer_scale, BH)
            c = tl.arange(0, BH // 2)
            g = tl.arange(0, BH // 16)
            tl.store(base + c, quantized, c < H // 2)
            tl.store(base + H // 2 + g, block_sf, g < H // 16)
        else:
            tl.store(base.to(tl.pointer_type(tl.bfloat16)) + col, values, col < H)
        slot = tl.arange(0, BK)
        ids = tl.load(Routes + row * K + slot, slot < K, -1)
        weights = tl.load(Gates + row * K + slot, slot < K, 0.0)
        keys = tl.sort(tl.where((ids >= 0) & (ids < E), ids, E) * BK + slot)
        expert = keys // BK
        tl.store((base + payload).to(tl.pointer_type(tl.int32)) + slot, expert, slot < K)
        tl.store(
            (base + payload + 4 * K).to(tl.pointer_type(tl.float32)) + slot,
            tl.gather(weights, keys % BK, 0),
            slot < K,
        )
        tl.store(Owner + row * K + slot, expert // (E // NODES), slot < K)


@triton.jit
def _unpack(
    Packet,
    X,
    SF,
    Ids,
    Gates,
    CAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    ELOCAL: tl.constexpr,
    NODE: tl.constexpr,
    STRIDE: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    NVFP4: tl.constexpr,
):
    row, source = tl.program_id(0), tl.program_id(1)
    payload: tl.constexpr = H // 2 + H // 16 if NVFP4 else 2 * H
    packet = Packet + source.to(tl.int64) * STRIDE
    n = tl.load(packet.to(tl.pointer_type(tl.int32)))
    base = packet + 16 + row.to(tl.int64) * (payload + 8 * K)
    dest = source * CAP + row
    col, slot = tl.arange(0, BH), tl.arange(0, BK)
    if row < n:
        if NVFP4:
            c, g = tl.arange(0, BH // 2), tl.arange(0, BH // 16)
            tl.store(X + dest * (H // 2) + c, tl.load(base + c, c < H // 2, 0), c < H // 2)
            tl.store(SF + dest * (H // 16) + g, tl.load(base + H // 2 + g, g < H // 16, 0), g < H // 16)
        else:
            tl.store(
                X + dest * H + col, tl.load(base.to(tl.pointer_type(tl.bfloat16)) + col, col < H, 0), col < H
            )
    ids = tl.load((base + payload).to(tl.pointer_type(tl.int32)) + slot, (row < n) & (slot < K), -1)
    ids -= NODE * ELOCAL
    tl.store(Ids + dest * K + slot, tl.where((ids >= 0) & (ids < ELOCAL), ids, -1), slot < K)
    weights = tl.load(
        (base + payload + 4 * K).to(tl.pointer_type(tl.float32)) + slot, (row < n) & (slot < K), 0.0
    )
    tl.store(Gates + dest * K + slot, weights, slot < K)


@triton.jit
def _pack_slots(
    Mail,
    Ids,
    Prefix,
    Tx,
    CAP: tl.constexpr,
    BCAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    NODE: tl.constexpr,
    BH: tl.constexpr,
):
    row, source = tl.program_id(0), tl.program_id(1)
    token = source * CAP + row
    if source != NODE:
        cols = tl.arange(0, BH)
        for slot in tl.static_range(K):
            index = token * K + slot
            if tl.load(Ids + index) >= 0:
                pos = tl.load(Prefix + index) - 1
                src = (slot * BCAP + token).to(tl.int64) * H
                dst = (source * CAP * K + pos).to(tl.int64) * H
                tl.store(Tx + dst + cols, tl.load(Mail + src + cols, cols < H, 0), cols < H)


@triton.jit
def _finish(
    Mail,
    Rx,
    Owner,
    RemotePrefix,
    Out,
    n,
    CAP: tl.constexpr,
    BCAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    NODE: tl.constexpr,
    NODES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(K):
        index = row * K + slot
        owner = tl.load(Owner + index)
        if owner == NODE:
            src = (slot * BCAP + NODE * CAP + row).to(tl.int64) * H
            acc += tl.load(Mail + src + cols, cols < H, 0).to(tl.float32)
        elif owner < NODES:
            pos = tl.load(RemotePrefix + owner * n * K + index) - 1
            src = (owner * CAP * K + pos).to(tl.int64) * H
            acc += tl.load(Rx + src + cols, cols < H, 0).to(tl.float32)
    tl.store(Out + row * H + cols, acc, cols < H)


@triton.jit
def _gradient_packet(
    Mail,
    Gate,
    Valid,
    Lengths,
    Tx,
    CAP: tl.constexpr,
    BCAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, source, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = tl.load(Lengths + source)
    if row < n:
        # Keeping node partials in FP32 limits wire bytes independently of K.
        # dGate belongs to exactly one expert and retains its sorted slot.
        cols = part * BLOCK + tl.arange(0, BLOCK)
        token = source * CAP + row
        packet = (Tx + source.to(tl.int64) * STRIDE).to(tl.pointer_type(tl.float32))
        acc = tl.full((BLOCK,), 0, tl.float32)
        for slot in tl.static_range(K):
            live = tl.load(Valid + token * K + slot)
            pos = (slot * BCAP + token).to(tl.int64) * H
            acc += tl.load(Mail + pos + cols, live & (cols < H), 0).to(tl.float32)
            if part == 0:
                g = tl.load(Gate + token * K + slot, live, 0.0)
                tl.store(packet + n * H + row * K + slot, g)
        tl.store(packet + row * H + cols, acc, cols < H)


@triton.jit
def _gradient_sum(
    Tx,
    Rx,
    Routes,
    DX,
    DG,
    n,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    NODE: tl.constexpr,
    NODES: tl.constexpr,
    STRIDE: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for node in tl.static_range(NODES):
        buf = Tx if node == NODE else Rx
        packet = (buf + node * STRIDE).to(tl.pointer_type(tl.float32))
        total += tl.load(packet + row * H + cols, cols < H, 0.0)
    tl.store(DX + row * H + cols, total, cols < H)
    if part == 0:
        slots = tl.arange(0, BK)
        ids = tl.load(Routes + row * K + slots, slots < K, -1)
        keys = tl.sort(tl.where((ids >= 0) & (ids < E), ids, E) * BK + slots)
        owner = (keys // BK) // (E // NODES)
        value = tl.full((BK,), 0, tl.float32)
        for node in tl.static_range(NODES):
            buf = Tx if node == NODE else Rx
            packet = (buf + node * STRIDE).to(tl.pointer_type(tl.float32))
            value += tl.load(packet + n * H + row * K + slots, (slots < K) & (owner == node), 0.0)
        tl.store(DG + row * K + keys % BK, value, slots < K)


_local_groups = weakref.WeakKeyDictionary()


def _local_team(group, lanes, node):
    # A single local team belongs to the parent EP group and can serve many
    # workspaces. Recreating hashed c10d subgroups after destroy can read the
    # previous NCCL bootstrap ID from a long-lived store.
    if group not in _local_groups:
        ranks = dist.get_process_group_ranks(group)
        _local_groups[group] = dist.new_group(
            ranks[node * lanes : (node + 1) * lanes], backend="nccl", use_local_synchronization=True
        )
    return _local_groups[group]


class MultiNodeRuntime(BlackwellRuntime):
    def __init__(
        self, group, experts, top_k, hidden, intermediate, capacity, stream, training, max_saved, nodes,
    ):
        self.nodes = nodes
        lanes = group.size() // nodes
        self.node = group.rank() // lanes
        self.local_group = _local_team(group, lanes, self.node)
        self.cap = max(1, capacity)
        self.experts = experts
        super().__init__(
            self.local_group,
            experts // nodes,
            top_k,
            hidden,
            intermediate,
            self.cap * nodes,
            stream,
            training,
            max_saved,
        )
        self.nvfp4 = self.buffer.fp4
        self.compact_nvfp4 = self.nvfp4 and not training
        payload = hidden // 2 + hidden // 16 if self.compact_nvfp4 else hidden * 2
        self.packet_stride = triton.cdiv(16 + self.cap * (payload + 8 * top_k), 128) * 128
        self.return_stride = self.cap * top_k * hidden * 2
        self.grad_stride = self.cap * (hidden + top_k) * 4
        region_bytes = max(self.return_stride, self.grad_stride if training else 0)
        self.send_offset = nodes * self.packet_stride
        self.recv_offset = self.send_offset + nodes * region_bytes
        self.rail = RailWindow(group, self.recv_offset + nodes * region_bytes, nodes)
        if self.nvfp4 and training:
            self.wgrad_stream = torch.cuda.Stream(device=stream.device)
        self.sizes = torch.empty(nodes, dtype=torch.int64, device=stream.device)
        self.owner_ids = torch.arange(nodes, device=stream.device, dtype=torch.int32).view(nodes, 1)
        self.dummy = torch.empty(self.cap * nodes, hidden, dtype=torch.bfloat16, device=stream.device)
        self.input_buffer = getattr(self, "dy_buffer", self.dummy) if self.nvfp4 else self.buffer.x
        self.stream.wait_stream(torch.cuda.current_stream())

    def stage_packet(self, x, routes, gates, x_scale=1 / 32):
        n, h = x.shape
        k = self.buffer.num_topk
        owner = torch.empty(n * k, device=x.device, dtype=torch.int32)
        _packet[(max(1, n),)](
            x,
            routes,
            gates,
            self.rail.data[self.node * self.packet_stride :],
            owner,
            self.sizes,
            n,
            x_scale,
            h,
            k,
            self.experts,
            self.nodes,
            self.node,
            triton.next_power_of_2(k),
            triton.next_power_of_2(h),
            self.compact_nvfp4,
        )
        return owner

    def dispatch_packet(self):
        self.rail.exchange(0, 0, self.packet_stride, self.sizes, broadcast=True)

    def _run(
        self,
        x,
        routes,
        gates,
        w1,
        w2,
        *,
        save=False,
        owner=None,
        on_launch=None,
        prepared=None,
        scales=None,
    ):
        if scales is None:
            scales = (self.scales.x, self.scales.intermediate) if self.nvfp4 else (1.0, 1.0)
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            b, n, h, k = self.buffer, len(x), self.buffer.hidden, self.buffer.num_topk
            cap, nodes, node = self.cap, self.nodes, self.node
            data = self.rail.data
            bh, bk = triton.next_power_of_2(h), triton.next_power_of_2(k)
            if owner is None:
                owner = self.stage_packet(x, routes, gates, scales[0])
                self.dispatch_packet()
            _unpack[(cap, nodes)](
                data,
                b.x if self.compact_nvfp4 else self.input_buffer,
                b.x_sf.view(torch.uint8) if self.compact_nvfp4 else b.x,
                b.topk_idx,
                b.topk_weights,
                cap,
                h,
                k,
                self.experts // nodes,
                node,
                self.packet_stride,
                bk,
                bh,
                self.compact_nvfp4,
            )
            ids = b.topk_idx[: cap * nodes].view(nodes, cap * k)
            prefix = (ids >= 0).cumsum(1, dtype=torch.int32)
            sizes = prefix[:, -1].to(torch.int64) * (h * 2)
            remote_prefix = (owner[None, :] == self.owner_ids).cumsum(1, dtype=torch.int32)
            if save:
                from .blackwell_backward import Tape, checkpoint

                if len(self.tapes) >= self.max_saved:
                    raise RuntimeError("Blackwell training tape capacity exhausted")
                buffers, ends = checkpoint(b, cap * nodes, x.device, self.alignment)
            options = dict(saved=buffers if save else ())
            if self.nvfp4:
                if prepared is None:
                    prepared = prepare_nvfp4_weights(w1, w2, weight_scales=(self.scales.w1, self.scales.w2))
                sx, sm = scales
                if not self.compact_nvfp4:
                    _pack_rows[(cap * nodes, triton.cdiv(h, 512))](
                        self.input_buffer, b.x, b.x_sf.view(torch.uint8), sx, 1, h, 512, False, False
                    )
                options["tensor_scales"] = (
                    sx * prepared.tensor_scales[0],
                    sm,
                    sm * prepared.tensor_scales[1],
                )
                a, c = prepared.fc1, prepared.fc2
            else:
                a, c = w1, w2
            ready = self.stream.record_event() if on_launch is not None else None
            self.launch(self.dummy, a, c, b, **options)
            if on_launch is not None:
                on_launch(ready)
            if self.nvfp4 and save:
                save_routed_x(self, buffers[2], buffers[5])
            bcap = b.num_max_tokens_per_rank
            mailbox = b.buffer[-bcap * k * h * 2 :].view(torch.bfloat16)
            tx = data[self.send_offset : self.recv_offset].view(torch.bfloat16)
            rx = data[self.recv_offset :].view(torch.bfloat16)
            _pack_slots[(cap, nodes)](mailbox, ids, prefix, tx, cap, bcap, h, k, node, bh)
            self.rail.exchange(self.send_offset, self.recv_offset, self.return_stride, sizes, broadcast=False)
            output = torch.empty_like(x)
            if n:
                _finish[(n, triton.cdiv(h, 512))](
                    mailbox, rx, owner, remote_prefix, output, n, cap, bcap, h, k, node, nodes, 512
                )
            parameters = (*a, *c) if self.nvfp4 else (w1, w2)
            for tensor in (x, routes, gates, *parameters):
                tensor.record_stream(self.stream)
            output.record_stream(caller)
            if save:
                lengths = (
                    data[: self.send_offset].view(nodes, self.packet_stride).view(torch.int32)[:, 0].clone()
                )
                tape = Tape(output, (routes.clone(), w1, w2, *buffers[2:], ends, ids.ge(0), lengths))
                self.tapes.add(tape)
        caller.wait_stream(self.stream)
        return tape if save else output

    def backward_async(self, tape, dy):
        from .blackwell_backward import expert_gradients

        if tape not in self.tapes or tape.state is None:
            raise RuntimeError("Backward requires a live training tape from this runtime")
        if dy.shape != tape.output.shape or dy.dtype != torch.bfloat16 or dy.device != self.stream.device:
            raise ValueError("Backward requires BF16 dY matching the forward output")
        routes, w1, w2, x, ab, gates, origins, ends, valid, lengths = tape.state
        self._stage_dy(dy)
        mailbox, dw1, dw2 = getattr(self, "expert_gradients", expert_gradients)(
            self, w1, w2, x, ab, gates, origins, ends
        )
        dx, dgate = self._return_gradients(routes, valid, lengths, mailbox, dy)
        torch.cuda.current_stream().wait_stream(self.wgrad_stream)
        tape.state = None
        self.tapes.discard(tape)
        return dx, dgate, dw1, dw2

    def _stage_dy(self, dy):
        cap, nodes, node = self.cap, self.nodes, self.node
        data = self.rail.data
        n, h = dy.shape
        packet_bytes = cap * h * 2
        packets = data[: nodes * packet_bytes].view(torch.bfloat16).view(nodes, cap, h)
        packets[node, :n].copy_(dy)
        self.sizes[node] = n * h * 2
        self.rail.exchange(0, 0, packet_bytes, self.sizes, broadcast=True)
        self.input_buffer[: nodes * cap].copy_(packets.view(-1, h))

    def _return_gradients(self, routes, valid, lengths, mailbox, dy):
        cap, nodes, node = self.cap, self.nodes, self.node
        b, data = self.buffer, self.rail.data
        n, h = dy.shape
        k = b.num_topk
        tx, rx = data[self.send_offset : self.recv_offset], data[self.recv_offset :]
        _gradient_packet[(cap, nodes, triton.cdiv(h, 512))](
            mailbox,
            b.topk_weights,
            valid,
            lengths,
            tx,
            cap,
            b.num_max_tokens_per_rank,
            h,
            k,
            self.grad_stride,
            512,
        )
        sizes = lengths.to(torch.int64) * ((h + k) * 4)
        self.rail.exchange(self.send_offset, self.recv_offset, self.grad_stride, sizes, broadcast=False)
        dx, dgate = torch.empty_like(dy), torch.empty_like(routes, dtype=torch.float32)
        if n:
            _gradient_sum[(n, triton.cdiv(h, 512))](
                tx,
                rx,
                routes,
                dx,
                dgate,
                n,
                h,
                k,
                self.experts,
                node,
                nodes,
                self.grad_stride,
                triton.next_power_of_2(k),
                512,
            )
        return dx, dgate

    def usage(self):
        return dict(**super().usage(), gin_bytes=self.rail.data.nbytes, gin_backend=self.rail.backend)

    def retire_peer_mappings(self):
        self.stream.synchronize()
        self.rail.close()
        super().retire_peer_mappings()
        self.local_group = None


class NVFP4MultiNodeRuntime(MultiNodeRuntime, NVFP4Runtime):
    """Use the NVFP4 local workspace and tensor cores with the GIN transport."""
