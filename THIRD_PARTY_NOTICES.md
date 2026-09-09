# Third-party notices

WeftMoE's own source is licensed under Apache-2.0. The following permissively
licensed projects are build dependencies or documented design references.
Their upstream copyright and license terms remain applicable.

- DeepGEMM (DeepSeek, MIT): descriptor/address and accumulator-array adapters
  in `src/cuda/primitives/hopper.cuh` are derived from its header utilities.
  The upstream license is retained in `LICENSES/DeepGEMM-MIT.txt`.
  No DeepGEMM headers, binary, or Python package are required to build/run WeftMoE.
- CUTLASS/CuTe (NVIDIA, BSD-3-Clause): external header dependency providing
  WGMMA, TMA, matrix stores, tensor-map, barrier, and layout primitives.
  Configure its standalone include directory with `CUTLASS_INCLUDE_DIR`.
- CUB (NVIDIA, BSD-3-Clause): scan, reduction, and radix-sort APIs from the CUDA
  Toolkit are used by device Notify. CUB is not vendored.
- DeepEP (DeepSeek, MIT): inspected as a transport/ownership design reference;
  no DeepEP source is vendored or linked.
- Triton-distributed (ByteDance and contributors, MIT): inspected as a
  distributed transport design reference; no source is vendored or linked.
- NCCL and CUDA (NVIDIA): external proprietary SDK/runtime dependencies. They
  are dynamically linked or loaded from the user's installation and are not
  redistributed by this repository.

The exact inspected upstream files and the distinction between inspiration
and build-time use are listed in `docs/references.md`. Upstream license texts
must accompany any separately redistributed third-party binaries or headers.
