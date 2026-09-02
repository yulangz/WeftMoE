# Design compliance

The table maps the current Forward implementation to the repository design
contract. Historical Backward and generic-shape claims are intentionally not
carried forward.

| Contract | Implementation | Validation |
| --- | --- | --- |
| Canonical, deterministic expert rows | GPU Notify sort/scan and immutable `DeviceDispatchHandle` | C++/Python reference cross-check, duplicate and capacity cases |
| M blocks padded to 128 | Notify M-tile metadata and compile-time `tile_m` constraint | metadata tests plus Forward output hashes |
| One CTA per SM with fixed roles | split RDMA, Relay, and Compute grids | launch audit and V3 protocol smoke |
| Communication overlaps FC1/FC2 | generation counters, ready queues, and direct FC2 return-layout stores | device timeline and end-to-end benchmark |
| FC2 is scheduler-priority work | work-conserving FC1/FC2 queue | tuning/resource tests and timeline events |
| No cross-role whole-grid barrier | independent grids exchange release/acquire state | source scan plus protocol stress tests |
| Device-issued cross-node transport | NCCL GIN put/signal/local completion | world-2/world-16 smoke and audit counters |
| Local transport uses CUDA IPC/NVLink | split local Relay path, optional asynchronous TMA row copies | world-2/world-8 smoke and output hashes |
| Return locations known before compute | immutable compact return descriptors from Notify | descriptor validation and repeated generations |
| Deterministic source combine | fixed expert order, FP32 accumulate, final BF16 cast | independent reference and repeated hashes |
| No steady-path allocation | preallocated Notify and activation pools | CUDA API audit and pool lifecycle tests |
| Explicit compile-time tuning | generated header from a bounded JSON plan | autotune unit tests and per-candidate digest |
| Unsupported work fails closed | fixed shape/topology/provenance checks | negative API and distributed protocol tests |

The Socket backend is retained for correctness comparison only and does not
satisfy the device-issued overlap requirement. Backward compliance is pending a
new design; `backward_pipeline_available()` reports that state explicitly.
