# WeftMoE

> Weaving communication into computation.

WeftMoE implements fused Mixture-of-Experts forward and backward pipelines for
Hopper GPUs. It overlaps TMA/WGMMA computation with node-local data movement and
inter-node NCCL GIN communication. Routed contributions are reduced in a fixed
order. Compiled specializations are cached, and tuning results are saved for
inspection.

The static Forward implementations are named `weftmoe_dt` (single-node decode)
and `weftmoe_pf` (general prefill/ring pipeline, including multi-node inference
and retained-state Forward). Their CUDA modules live in `src/cuda/weftmoe_dt/`
and `src/cuda/weftmoe_pf/`; shared GEMM/worker code lives in `src/cuda/forward/`.
Runtime selection, tuning/environment names and the V3 fallback are unchanged.
`weftmoe_pf` targets long sequences on both single- and multi-node systems;
short multi-node requests retain their existing compatibility path.

## Installation

Build and install from the repository:

```bash
python3 -m pip install .
```

For development, use `python3 -m pip install -e .` and reinstall after modifying
native code. Environments managed by uv can use `uv pip install .` or
`uv pip install -e .`. To build a wheel and source distribution, install the
Python `build` package and run `python3 -m build`.

Source builds require a C++17 compiler, CUDA with SM90a support, CUTLASS headers,
and NCCL 2.31.2 or newer headers and libraries with GIN support. Configure toolchain discovery
with `CUDA_HOME`, `NVCC`, `CXX`, `CUTLASS_INCLUDE_DIR`, `NCCL_INCLUDE_DIR`, and
`NCCL_LIB_DIR`. CUDA and NCCL are external runtime dependencies; the package does
not bundle them. `MAX_JOBS` controls package-build parallelism.

## Python Usage

The unified repository entry point is [tools/run.py](tools/run.py). It compiles
and benchmarks a single configuration, or selects a configuration with
autotuning before running the benchmark. Run the following from the repository
root after installation:

```python
from weftmoe import ModelSpec
from tools.run import run

model = ModelSpec(hidden=4096, intermediate=1536, experts=64, top_k=8)

result = run(
    model,
    phase="forward",       # Use "backward" for the backward pipeline.
    tokens=8192,            # Input tokens per rank.
    nodes=1,
    gpus_per_node=8,
    autotune=False,         # Set True to compile and measure the search space.
)
print(result["config"])
print(result["benchmark"]["record"])
```

This entry point executes generated benchmark inputs through the native rank
launchers. Application tensors use the separate, explicitly prepared
[`TensorRuntime`](weftmoe/runtime.py) API. The Qwen3.5 AReaL integration and its
additional binding build are described in [the integration guide](docs/areal_qwen35.md).
Python-driven benchmark compilation and autotuning are supported; automatic
tuning on a tensor operator's first call is not.

Without autotuning, the entry point compiles one configuration and runs 30
measured iterations. The single-node defaults use 24 relay CTAs for forward
and 32 for backward. A supplied `config` file uses its first candidate values.
With `autotune=True`, it compiles the candidate set, measures each candidate,
confirms the finalists, and benchmarks the selected configuration at the
requested token count. An explicit autotune call performs a new measurement;
compiled binaries can be reused from cache.

Additional arguments configure the same workflow:

- `config`: optional hyperparameter search-space JSON path. Workload dimensions
  and parallel topology are supplied separately from the search space.
- `system`: optional system settings, including `remote`, `port`,
  `nccl_lib_dir`, and `runtime_library_path` for multi-node execution.
- `jobs`: global compilation concurrency budget, default 16.
- `output`: result JSON path; associated logs are stored alongside it.
- `generate_only=True`: validate and inspect the configurations without
  compiling or running GPU work.

`WEFTMOE_ROOT` selects the compile and tuning cache root, defaulting to
`~/.cache/weftmoe`. `WEFTMOE_AUTOTUNE_CONFIG` supplies a search-space file when
`config` is omitted. Parameter definitions and tuning recommendations are in
[configs/README.md](configs/README.md).

Tuning uses token buckets with representatives of 64, 8192, and 65536 tokens
per rank. The final benchmark uses the original token count without padding it
to the bucket size. The default search uses 15 measured iterations per candidate
and 30 per finalist confirmation pass. Each run has 10 additional warmup
iterations. Two confirmation passes reverse candidate order; the final selected
configuration is benchmarked for 30 measured iterations.
