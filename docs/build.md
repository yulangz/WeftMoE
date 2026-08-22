# Build and developer workflow

The phase-1 library is deliberately dependency-free C++17. The checked build
environment did not provide CMake, so the repository uses a small Makefile.

```bash
make -j                     # C++ test and benchmark binaries
make test                   # Python reference tests, then C++ production tests
make bench                  # 8 ranks × 8192 tokens × top-k 6 host benchmark
```

Build artifacts stay under `build/`. Override `CXX` or `CXXFLAGS` normally:

```bash
make clean
make CXX=clang++ CXXFLAGS='-O2 -g -std=c++17 -Wall -Wextra -Werror'
```

CUDA 12 and GPUs are not needed for Notify Dispatch host-control validation.
The available development GPUs may differ from H200. Later SM90 kernels should
be compiled for `sm_90a` on H200 and tested separately from this portable
metadata suite; they must compare every emitted metadata array against the
Python reference before performance work begins.
