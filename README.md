# WeftMoE

> Weaving communication into computation.

WeftMoE implements fused Mixture-of-Experts forward and backward pipelines for
Hopper GPUs. It overlaps TMA/WGMMA computation with node-local data movement and
inter-node NCCL GIN communication. Routed contributions are reduced in a fixed
order. Compiled specializations are cached, and tuning results are saved for
inspection.

The retained implementations are `weftmoe_pf` (single-/multi-node prefill,
inference and training) and `weftmoe_dt` (single-node decode). PF inference and
training share the same main kernel after standalone notify; training additionally
allocates and writes the activation tape. PF's streaming backward lives in
`src/cuda/weftmoe_pf/backward_kernel.cuh`: it reuses the PF planner and transport,
then runs static dQ/SwiGLU/dX work before dW. Shared backward math and the
compatibility schedule for other tile configurations live in `src/cuda/backward/`.
See [backward implementation and measurements](docs/pf_backward_20261012.md).

V3, earlier PF kernels, WG3 experiments and their frozen build profiles have
been removed from `main`. The complete pre-cleanup tree is preserved by tag
`archive/pre-pf-dt-cleanup-20261012` at commit `565037e`. Historical reports and
measurement files remain available; their old source paths refer to that tag.
There is no V3 fallback. Use a PF or DT specialization matching the model shape;
see [current profiles](configs/README.md).

## Installation
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
and NCCL 2.31.2 or newer headers and libraries with GIN support. Multi-node
execution requires the GDAKI backend; CPU Proxy, GPI, and EFA GDA device paths
are not compiled into the native library. Configure toolchain discovery
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
measured iterations. PF uses static compute ownership and the borrowed-warp multi-node pipeline. A supplied `config` file uses its first candidate values.
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
