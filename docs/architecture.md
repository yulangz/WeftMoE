# Forward architecture

## Scope and invariants

The production data plane implements one fixed Hopper Forward shape:
`H=7168`, `I=3072`, and BF16 inputs/weights/outputs. FC1 accumulates in FP32,
rounds A/B to BF16, and evaluates SwiGLU from those exact rounded operands.
The epilogue then applies the route-local FP32 gate and rounds the resulting
FC2 input once to BF16. Returned routes are accumulated in FP32 without
another gate multiplication. Every expert row range is padded to an M-block
boundary of 128.
The same M-block identity is used by Notify, Dispatch, FC1, FC2, Return, and
Reduce; a phase cannot silently reinterpret a tail.

Topology is homogeneous. Rank identity is exactly `(node, local_rank)`, nodes
and local ranks are dense, and expert ownership is immutable for a generation.
All public inputs carry session, generation, stream, device, shape, and owner
provenance. A mismatch fails before launching the data plane.

## Control plane

Each rank starts with only its own selected experts and gates. Notify performs:

1. stable rank-local histogram and prefix construction;
2. fixed-width control/config agreement;
3. NCCL AllGather of expert counts and destination-node summaries;
4. GPU materialization of canonical destination rows, packet fan-out,
   M-block readiness geometry, reverse routes, and return locations.

The sort key includes the original flat route index, so duplicate experts have
a total order. The resulting `DeviceDispatchHandle` is immutable. Its backing
arena and collective completions stay alive through a shared lease; kernel
adapters receive a short-lived pointer POD only after acquiring a consumer
lease.

The resource pools preallocate device arenas, pinned terminal records, events,
NCCL workspaces, CUDA IPC mappings, and optional GIN windows. A steady
generation allocates none of these resources. Slots use monotonically
increasing epochs and are quarantined if completion cannot be proven.

## Split data plane

The optimized implementation launches independent transport and compute grids.
One CTA occupies one SM, and a CTA keeps one role for the entire generation.

```text
Notify metadata
      |
      v
RDMA issue/completion  ---> remote ingress readiness
      |                           |
      +---------------------------v
                            Relay/TMA copy
                                  |
                                  v
                            FC1-ready queue
                                  |
                                  v
                 FC1 + SwiGLU + route gate (TMA/WGMMA)
                                  |
                                  v
                         FC2-priority queue
                                  |
                                  v
                     FC2 direct return-layout store
                                  |
                    +-------------+-------------+
                    v                           v
             local Relay copy            remote GIN put
                    +-------------+-------------+
                                  v
                         token-ready Reduce queue
                                  |
                                  v
                 fixed expert-order unweighted FP32 sum
```

The grids never synchronize with a whole-grid cross-role barrier. They exchange
only generation-tagged counters, append-only descriptors, and bounded queues:

- a producer writes payload bytes, performs the required device/system release
  operation, then publishes readiness;
- a consumer acquires readiness before reading those bytes;
- each task has one owner and a monotonic claimed/completed record;
- terminal publication requires every expected task and local completion to be
  accounted for exactly once.

The system-scope fence orders GPU stores before a peer device or GIN operation
is allowed to observe the completion marker. It does not wait for every CTA and
is not equivalent to `grid.sync()`.

## CTA roles

### RDMA

For two-node GIN, eight CTAs are the default: one owner per physical rail. They
pack the dispatch payload, issue GIN puts/signals, track local completion, scan
ready FC2 return chunks, and issue FULL-return puts. They do not enter WGMMA or
perform node-local relay copies.

### Relay

Relay CTAs consume remote ingress and move rows into canonical local expert
storage through CUDA IPC/NVLink. The local path uses asynchronous TMA streams
when possible. Relay also handles local return movement and publishes reduce
readiness after all required local/remote contributions are visible.

### Compute and Reduce

All remaining CTAs are compute-only during FC1/FC2. The scheduler is
work-conserving: a ready FC2 M-block is selected before new FC1 work so return
traffic begins while other experts still compute. After compute and return
work drain, these CTAs consume the token reduce queue; there is no separate
global reduction launch.

For each canonical route row, the numeric contract is:

```text
A_bf16 = bf16(FC1_A(X))
B_bf16 = bf16(FC1_B(X))
S = SwiGLU(fp32(A_bf16), fp32(B_bf16))
U = bf16(fp32(gate) * S)
R = bf16(FC2(U))
Y[token] = bf16(sum_fp32_in_expert_order(R[route]))
```

Dispatch carries the FP32 gate beside the deduplicated activation packet and
publishes it into canonical expert-row order before the row's FC1-ready release.
This placement lets Backward form `dGate = <dU, S>` without retaining FC2
output, while `dW2` consumes the exact gated BF16 input `U` used by Forward.

The inference configuration aliases canonical X with the temporary FC2 output
plane after FC1 has consumed it. An opt-in Backward-state pool instead keeps X
immutable, writes plane-major BF16 A/B checkpoints, and selects a separate FC2
output plane. The returned Forward handle owns that generation-local storage.

## GEMM implementation

The default compile-time point uses:

| parameter | FC1 | FC2 |
| --- | ---: | ---: |
| tile M | 128 | 128 |
| tile N | 128 | 256 |
| tile K | 64 | 64 |
| stages | 4 | 4 |

Each compute CTA has one TMA producer warpgroup and two WGMMA math warpgroups.
Register reconfiguration assigns the producer and math roles independently;
shared storage is reused between FC1 and FC2 after its prior lifetime ends.
FC2 uses an N64 store atom and can use the TMA epilogue path. Cluster size,
tile shapes, stages, register budgets, wait depth, and store policy are
compile-time tuning parameters.

## Local and remote topology

World 8 uses the split architecture with `rdma_ctas=0`; Relay CTAs move local
traffic over CUDA IPC/NVLink while compute proceeds. No legacy combined local
selection remains.

World 16 uses same-rail dispatch to avoid duplicating cross-node activation
traffic. Return uses FULL connectivity because every source needs its own
contribution and the rail form cannot reduce return bytes. Fixed RDMA owners
issue cross-node operations; Relay owners handle the destination node.

The NCCL Socket implementation is an explicit correctness backend. Its payload
phases are host-enqueued and therefore it is excluded from overlap/performance
claims.

## Completion and failure semantics

Every production kernel reports a fixed-width status containing generation,
epoch, origin rank, error code, field, expected value, and actual value. GIN
and local split paths use device-visible completion dependencies; the Socket
path additionally performs collective terminal agreement. Cancellation and
deadlines are generation-wide, and malformed or stale state fails closed.

The cached-state Backward data plane is now qualified on the same world-8 and
world-16 topologies. It retains this Forward generation's compact X, BF16 A/B,
gate, and immutable Dispatch context, and reports availability only from
CUDA-backed binaries that contain the implementation. Its task dependencies,
compact/padded address contract, and independent performance evidence are in
[backward_design.md](backward_design.md).
