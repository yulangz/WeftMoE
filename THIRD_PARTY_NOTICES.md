# Third-party notices

MegaMOE's own source is licensed under Apache-2.0. The following permissively
licensed projects are build dependencies or documented design references.
Their upstream copyright and license terms remain applicable.

- DeepGEMM (DeepSeek, MIT): selected header-only TMA, SM90 WGMMA, load/store,
  and math utilities are vendored under `src/cuda/vendor/deep_gemm`; the
  upstream MIT license is retained alongside those files. No DeepGEMM binary
  is redistributed.
- CUTLASS/CuTe (NVIDIA, BSD-3-Clause): CUDA tensor-map, barrier, layout, and
  device primitives are included through the external DeepGEMM/CUTLASS
  installation. CUTLASS is not vendored.
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
