# Implementation roadmap and current state

The seven implementation phases are closed. This file records what each phase
delivered and separates remaining validation or optimization work from missing
production behavior. Detailed requirement mapping is in
`docs/design_compliance.md`; measured results are summarized in
`docs/performance_summary.md`.

## Phase 1 — deterministic rank-local foundation

Complete: strict homogeneous topology, rank-local histogram, immutable phase
handles, typed host storage, opaque device leases, canonical rows, reverse
routes, packet/fan-out descriptors, workspace identity, independent Python
oracle, and ASan/UBSan failure containment.

## Phase 2 — device Notify and NCCL agreement

Complete: device validation/stable mapping, fixed little-endian control wire,
exact config agreement, real NCCL AllGather counts/summaries, completion leases,
failure recovery, and local/two-host world 2/16 validation. Trusted single-node
release generations use one fused status/count/summary collective after static
session config is verified; debug and cross-host paths retain the full protocol.

## Phase 3 — all-device metadata materialization

Complete: count matrices remain on device; parallel source/expert prefix,
canonical row, M-tile, reverse route, node mask, packet/fan-out and expected
signal metadata are written into an immutable arena. Production performs no
bulk metadata D2H/materialize/H2D roundtrip and no steady allocation/event
creation.

## Phase 4 — activation transport

Complete: typed BF16 activation/gate provenance, CUDA IPC LocalCudaP2p,
node-dedup/same-rail descriptors, ring/credit/sequence protocol, deterministic
source gate-after combine, and real GIN/RDMA plus separately labelled Socket
correctness fallback. Simulation remains an oracle and never reports a physical
backend audit.

## Phase 5 — persistent Forward

Complete: one cooperative data-plane kernel contains dispatch, TMA/WGMMA FC1,
SwiGLU, TMA/WGMMA FC2, return and deterministic reduce for the frozen
`hidden=7168, intermediate=3072` specialization. Cache/no-cache lifetimes,
actual-device weight hashes, cross-EP bitwise fixtures and independent FP64
checks are closed. Standalone Forward has observed a two-CTA cluster, while the
current deferred F+B path observes one CTA; both audits are recorded with their
measurement and TMA multicast is disabled.

## Phase 6 — persistent Backward

Complete: one cooperative kernel contains dY dispatch, cache/recompute FC1,
dV/SwiGLU backward, dX, split-k=1 canonical-M dW1/dW2, dGate, return and fixed-
order dX combine. Pool=2 isolation, empty experts, gradient hashes, sanitizer
paths and production terminal error agreement are closed. The checked local
three-run comparison clears the 15% Backward and Notify-inclusive training-step
gates against Megatron-Core 0.17.0.

## Phase 7 — cross-host training data plane

Complete on the available two homogeneous nodes: NCCL 2.29.7 CUDA-12.9 GIN
device comm/window/context, same-rail device put/signal/local completion,
CUDA-IPC destination fan-out, complete Forward/Backward/dGate, deterministic
return, and production status agreement. A concrete host-enqueued NCCL Socket
backend supplies a correctness/lower-bound fallback and never claims GIN or a
single persistent cross-host payload path. Physical runs cover world 2 and 16.

## Final runtime-shape and code-organization pass

Complete: non-main positive H/I values now select one-kernel Forward and
Backward correctness fallbacks with BF16 inputs/outputs, FP32 accumulation,
cache/no-cache behavior and M/N/K tails. Recorded cases are 64/32, 96/48,
127/65 and 256/192 on world 1 plus 127/65 through real world-2 LocalCudaP2p.
The main shape continues to select the unchanged WGMMA/TMA cubin. Generic GIN
uses the real same-rail device data plane and is correctness-tested on the
available physical world-2/world-16 topology.

Internal Forward/Backward pointer contracts and test harnesses are separated
from installed headers while CUDA remains a single aggregation translation
unit to preserve compiler visibility and SASS. New tests are added through
focused `tests/cuda` entry points; legacy launcher modes remain compatible.

## Remaining work

- Runtime and performance validation on physical H200/SM90a hardware. Current
  runtime evidence is from machines reporting L20X with the documented virtual
  compute-capability behavior; SM90a is also compile-checked.
- Physical GIN validation on three or more nodes and on heterogeneous/missing-
  rail topologies. Heterogeneous nodes remain explicitly unsupported; 3x1,
  3x4 and 4x2 homogeneous descriptor/team construction is simulated.
- Performance specializations/autotuning for H/I shapes other than 7168/3072.
  The generic path is deliberately correctness-only.
- Optional graph-capture and further GIN/weight-multicast tuning, provided the
  frozen canonical, cast and accumulation invariants remain unchanged.
