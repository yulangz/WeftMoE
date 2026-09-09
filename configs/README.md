# Autotune Configuration Reference

The configuration files define kernel hyperparameter search spaces for forward
and backward execution on single-node and multi-node systems. Input dimensions
and parallel topology are supplied separately by the caller. Compiler, library,
and network settings are not kernel hyperparameters.

Each parameter contains a nonempty array of candidate values. The first values
form the incumbent configuration; the Cartesian product defines the search
space. A single value fixes a parameter for that search. Supported values and
parameter combinations are checked by the autotuner before benchmarking.

FWD denotes forward execution; BWD denotes backward execution. A CTA is a CUDA
thread block. A warp contains 32 threads, and a warpgroup contains four warps.
An M-block is a block of routed rows belonging to one expert.

## GEMM Parameters (`1_gemm`)

Tile dimensions determine work granularity and data reuse. Pipeline stages
control how many input buffers can overlap data movement with computation.
Larger tiles or deeper pipelines can improve reuse, but also increase resource
requirements and reduce scheduling flexibility.

| Parameter | Phase | Purpose and guidance |
|---|---|---|
| `tile_m` | FWD/BWD | Row dimension of activation GEMM tiles. Padding and the number of available row blocks affect utilization. Keep the default unless evaluating another supported tile shape. |
| `tile_k` | FWD/BWD | Reduction-dimension step of the GEMM pipeline. The current implementation uses 64 to match its shared-memory layout. |
| `fc1_tile_n`, `fc2_tile_n` | FWD | Output-column dimensions of the two forward GEMMs. Wider tiles can improve reuse but increase register pressure and reduce the number of independently schedulable tasks. |
| `dab_tile_n`, `dx_tile_n` | BWD | Output-column dimensions of the activation-gradient and input-gradient GEMMs. Start with 256 for both; evaluate narrower tiles when dimensions or task parallelism warrant it. Narrower tiles do not necessarily reduce register usage. |
| `dw_tile_m`, `dw_tile_n` | BWD | Output-tile dimensions of the weight-gradient GEMMs. Larger tiles reuse more input data but require more accumulator storage. |
| `fc1_stages`, `fc2_stages` | FWD | Number of input-buffer stages for each forward GEMM. More stages can hide load latency at the cost of shared memory. Change only supported stage pairs. |
| `dab_stages`, `dx_stages` | BWD | Input-buffer stages for activation-gradient and input-gradient GEMMs. These values must match in the current implementation. |
| `dw1_stages`, `dw2_stages` | BWD | Input-buffer stages for the two weight-gradient GEMMs. These values must match in the current implementation. |
| `compute_threads` | FWD/BWD | Threads per compute CTA. The current implementation requires 384. |
| `math_warpgroups` | FWD/BWD | Warpgroups assigned to matrix computation. The current implementation requires two. |
| `producer_warpgroups` | FWD/BWD | Warpgroups assigned to staging operands. The current implementation requires one. |
| `math_registers`, `producer_registers` | FWD/BWD | Per-thread register budget requests for the corresponding roles. These are allocation requests, not measured register counts; inspect compiler resource usage when changing them. |
| `wgmma_wait_group` | FWD | Number of outstanding WGMMA groups permitted at a wait point. Controls the balance between pipeline overlap and operand or accumulator dependencies. |
| `fc2_store_atom_n` | FWD | Column width of the FC2 output TMA store atom. Determines the granularity of output stores. |
| `cluster_size` | FWD | Number of CTAs in a compute cluster. Affects cooperative scheduling, TMA multicast, and residency constraints. |
| `fc2_tma_store` | FWD | Enables FC2 output stores through shared-memory staging and TMA. Keep enabled as the starting point; its supported stage and cluster combinations are constrained. |

## Transport Parameters (`2_transport`)

Relay CTAs handle node-local data movement and associated progress work. RDMA
CTAs handle inter-node GIN communication. Increasing transport concurrency can
improve operand delivery and return progress, but leaves fewer SMs available
for compute workers.

| Parameter | Phase | Purpose and guidance |
|---|---|---|
| `rdma_ctas` | FWD/BWD | Number of inter-node GIN communication CTAs. Single-node execution sets this to 0. The current GIN implementation uses 8; this is not the number of nodes or the EP size. |
| `relay_ctas` | FWD/BWD | Number of CTAs assigned to node-local transport. For small hidden dimensions, include larger values such as 24, 32, and 40: transport progress can limit throughput even on one node. Stop increasing the count when reduced compute capacity outweighs faster transport. Counts remain subject to alignment and residency constraints. |
| `rdma_control_warps` | FWD | Warps assigned to RDMA issue and completion control. The current protocol requires two. |
| `local_dispatch_tma_warps` | FWD | Warps per relay CTA used for node-local dispatch copies. More copy streams increase concurrency but compete with progress and readiness publication work. |
| `local_return_tma_warps` | FWD | Warps used for node-local return TMA copies. A value of 0 selects vector copies. Comparing 0 and 8 is a useful initial search; more TMA streams are not always faster end to end. |
| `relay_scatter_warps` | FWD | Warps assigned to relay scatter copies. Controls parallelism within a scatter task. |
| `local_return_batch_mblocks` | FWD | Maximum M-blocks processed in one local return batch. Larger batches amortize scheduling overhead but can delay service of other ready work. |
| `dispatch_publish_batch_rows` | FWD | Row granularity for publishing dispatch arrival progress. Smaller batches expose compute work sooner; larger batches reduce publication overhead. |
| `return_chunk_rows` | FWD | Row granularity of inter-node return chunks. Controls transfer and completion-tracking granularity. Chunks must preserve the padded M-block layout. |
| `return_max_inflight` | FWD | Limit on the return in-flight window. Larger windows allow more outstanding work but consume more tracking resources. A value of 0 disables this window limit. |
| `registered_output` | FWD | Places FC2 output in the GIN-registered output region for direct transfer. Required by the GIN FC2 TMA store path. |
| `relay_local_return` | FWD | Assigns local return processing to relay CTAs. Keep enabled; the current supported path requires it. |
| `dispatch_batch_rows` | BWD | Routed rows claimed in one relay dispatch batch. Start with 16 or 32 for node-local execution. Larger batches can reduce scheduling overhead, while smaller batches improve responsiveness to return work. |
| `gin_max_inflight` | BWD | Maximum outstanding GIN requests. Relevant only to inter-node communication. Increase when insufficient outstanding work limits transfer throughput. |
| `setup_threads` | BWD | Threads per setup CTA. Controls parallelism in pipeline initialization. |
| `setup_idle_nanosleep` | BWD | Backoff hint while setup waits for progress. Keep the default unless initialization waiting is significant. |

## Scheduler Parameters (`3_scheduler`)

Scheduler parameters balance queue contention, weight reuse, and how quickly
newly ready work can execute. Larger batches or locality groups can reduce
scheduling overhead while increasing the time other tasks spend waiting.

| Parameter | Phase | Purpose and guidance |
|---|---|---|
| `local_route_batch_rows` | FWD | Number of local dispatch routes assigned per batch. Larger batches reduce claim overhead but can increase load imbalance. |
| `startup_mblocks` | FWD | Initial M-blocks per expert published without waiting for subsequent locality groups. Controls how quickly the first compute work becomes available. |
| `fc1_locality_group` | FWD | Groups adjacent M-blocks when publishing FC1 work to encourage reuse of the same weight tiles in L2. Larger groups can improve reuse but delay FC2 and return progress. |
| `fc1_locality_group4_begin`, `fc1_locality_group4_end` | FWD | Expert-local M-block interval `[begin, end)` that uses groups of four when the base locality group is two. These are row-block indices, not expert IDs or GPU groups. A bounded middle interval preserves finer scheduling near startup and completion. |
| `fc2_priority_burst` | FWD | Limit on consecutive priority FC2 task claims. A value of 0 leaves this priority burst unlimited. Smaller finite bursts allow FC1 work to receive service more frequently. |
| `task_claim_chunk` | FWD | Compute tasks claimed per queue operation. The current implementation supports one task per claim. |
| `task_idle_nanosleep` | FWD | Backoff hint for idle compute-task polling. |
| `transport_idle_nanosleep` | FWD | Backoff hint for idle transport polling. |
| `setup_nanosleep` | FWD | Backoff hint during setup waits. |
| `queue_idle_nanosleep` | BWD | Backoff hint when no compute-ready task is available. |
| `ready_word_stagger` | BWD | Starts different CTAs at different ready-bitmap words when enabled. Can reduce contention, but may weaken locality between related tasks. |
| `activation_owner_group` | BWD | Group size for distributing activation-gradient task claims among neighboring ready M-block owners. These are compute-work owners, not communication ranks. Start with 4 to balance contention and weight reuse. |
| `epilogue_rows_per_task` | BWD | Rows processed per activation epilogue task. Controls the granularity of non-GEMM activation-gradient work. |
| `dw2_claim_tiles`, `dw1_claim_tiles` | BWD | Weight-gradient tile claim granularity. Larger claims amortize scheduling overhead but can delay other ready work. |

Nanosleep values are GPU backoff hints, not guaranteed delays. Lower values may
improve responsiveness while increasing polling contention. Tune these after
transport concurrency and GEMM tile choices.

## Completion and Reduction Parameters (`4_completion_reduce`)

In FWD, return processing makes each routed contribution available at its
source token. Once all contributions for a token are ready, the token can enter
the reduction queue. Idle CTAs may consume ready tokens and accumulate their
contributions in a fixed top-k order. Inter-node return completion also checks
fragment markers before publishing route readiness.

BWD uses a separate input-gradient reduction kernel after persistent transport
completes. This reduction can overlap remaining weight-gradient computation.
Its thread-count setting does not change the main compute-worker allocation.

| Parameter | Phase | Purpose and guidance |
|---|---|---|
| `completion_chunks_per_pass` | FWD | Return completion chunks considered in one progress pass. Larger values can accelerate completion discovery but reduce time available for other transport work. |
| `completion_marker_warps` | FWD | Warps checking return-fragment completion markers. Increase only when marker scanning limits completion progress. |
| `reduce_scheduler` | FWD | Selects how ready reduction work is discovered and scheduled. Use `queue` as the starting point. |
| `reduce_warps` | FWD | Warps participating when a CTA executes reduction work. Controls parallelism within the reduction role. |
| `reduce_tokens_per_claim` | FWD | Tokens claimed in one reduction queue operation. Must not exceed `reduce_warps`. Larger claims reduce queue overhead but can reduce load balance near completion. |
| `reduce_registers` | FWD | Per-thread register budget request when a CTA enters its reduction role. Inspect compiled resource usage when changing it. |
| `return_idle_nanosleep` | FWD | Backoff hint while waiting for return progress. |
| `reduce_queue_nanosleep` | FWD | Backoff hint while waiting for reduction work. |
| `finalize_nanosleep` | FWD | Backoff hint while waiting for final completion. |
| `reduce_threads` | BWD | Threads per input-gradient reduction CTA. Each warp handles one token in the current implementation. Start with 256; larger CTAs may reduce scheduling flexibility for short reductions. |
| `reduce_idle_nanosleep` | BWD | Backoff hint while waiting for input-gradient route contributions. |
