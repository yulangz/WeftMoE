# Current PF and DT profiles

`main` retains one PF dataflow and the latest stable DT implementation. Older
kernels and profiles are available at `archive/pre-pf-dt-cleanup-20261012`.
Historical measurement IDs and reports are not rewritten.

| Profile | Scope |
| --- | --- |
| `pf_inference_flash_20261008` | Single-node PF, H4096/I2048/E256/top-k6; inference or training cache |
| `pf_network_l132_20261009` | Multi-node PF, same shape; borrowed warp, 132 network owners, D32/R128 |
| `dt_flash_probe_20261009` | Stable single-node DT, same shape; collective capacity <=64 |
| [`pf_qwen35_20261013`](pf_qwen35_20261013/README.md) | Tuned cached PF FWD+BWD, H2048/I512/E256/top-k8, EP8/EP16 |
| [`pf_qwen35_compact_20261013`](pf_qwen35_compact_20261013/README.md) | Compact local contributions and window-ordered network tape; Qwen M14 candidate and validation status |
| [`pf_flash_strategy_20261015`](pf_flash_strategy_20261015/README.md) | Flash transfer experiment with Qwen dataflow switches; mixed/negative results, **not a replacement default** |

PF keeps M128/N256/K64, three input stages, two-CTA multicast, FC1 register-vector
writeback and the retained FC2 epilogues. Both PF modes run standalone notify
then the same main kernel. Training allocates exact-size X/GU caches between
them; inference passes no activation cache. The network profile inherits the
local profile directly, not an experimental WG3 configuration.

Build with the existing CUDA/CUTLASS/NCCL environment:

```bash
export NVCCFLAGS="-lineinfo -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I${PWD}/configs/pf_network_l132_20261009"
python build_native.py lib --output build/current_pf/libweftmoe.so --jobs 4
python tools/build_tensor_runtime.py --library build/current_pf/libweftmoe.so
```

Select the printed extension with `WEFTMOE_TENSOR_EXTENSION`. Set ring capacity
before preparing a runtime (`WEFTMOE_STATIC_RING_TOKENS`,
`WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS`); the measured network configuration
uses 4096 tokens and two contribution windows. Network contributions retain the
complete invocation, while dispatch RX and X/S/return TX remain bounded rings.
`WEFTMOE_PF_MIN_RING_DEPTH=2` and `WEFTMOE_PF_CREDIT_WINDOWS=2` select the retained
network capacity settings. Worker mode, publication-plan and return-deduplication
switches have been removed: only the retained settings are compiled.

For PF inference timing, use `benchmarks/torchrun_megamoe_prefill.py --backend pf
--prepared-weights`. For cached FWD or FWD+BWD, use `tools/benchmark_training.py
--backend pf` (add `--forward-only` for cached FWD). Input generation is untimed;
normal training timing includes current-weight preparation. DT uses the same
inference benchmark with `--backend dt` and its own compiled profile.

`pf_training_autotune.json` searches current PF ring/FWD/BWD settings through
`tools/autotune_training.py`, with caller-supplied model dimensions and topology.
For current stream BWD GEMM tuning, prefer `pf_backward_stream_autotune.json`:
it holds FWD/communication fixed and varies the active activation/dW tiles,
stages and dW chunks. BWD now has no legacy kernel fallback; cluster-N2 shapes
must fit the selected N tile (use N128 for odd N256 extents).
`forward_v4_ring_autotune.json` and `forward_v4_decode_autotune.json` select PF
and DT respectively in the independent Forward tuner. Compatibility parameter
spellings remain in the generated-header format; obsolete scheduler axes are
no longer varied by the shipped Forward grids. The compatibility BWD grids also
keep the retired `dispatch_batch_rows` field fixed, avoiding duplicate builds.
Candidates and configuration
files are search definitions, not claims of measured optimality.

`pf_qwen35_training_autotune.json` is the current-path search for Qwen3.5-35B-A3B
(H2048/I512/E256/top-k8). It searches FWD N/stages, BWD activation N/K/stages,
dW geometry/stages/chunks, then runtime ring capacity. Run it separately for
EP8 and EP16 with `tools/autotune_training.py --tokens 8192 20480`; model shape
and topology remain CLI arguments. It does not search retired BWD return axes.
For network PF, contribution capacity covers the complete invocation and is
derived from token capacity; a D-only variant is redundant and is not searched.

The optional single-node `table.json` performs exact-capacity profile lookup,
not first-call compilation. Its recorded entries refer to immutable prepared
weights, inference, and the historical trace capacities. Re-export tuning
results after rebuilding; do not reuse compiled digests from an older checkout.
