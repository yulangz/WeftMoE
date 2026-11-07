// Shape-independent FP8 helpers, usable before a model runtime is prepared.
#include "python/fp8.hpp"
PYBIND11_MODULE(_runtime, m) { weftmoe::python_fp8::bind(m); }
