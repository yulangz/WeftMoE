#pragma once

#include "cuda/fp8/api.hpp"
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include <string>
#include <vector>

namespace weftmoe::python_fp8 {
namespace py = pybind11;
using Shape = std::vector<std::uint64_t>;

inline Shape shape(const py::object &x) { return x.attr("shape").cast<Shape>(); }
inline void *data(const py::object &x) {
  return reinterpret_cast<void *>(x.attr("data_ptr")().cast<std::uintptr_t>());
}
inline void check(const py::object &x, const char *dtype, const Shape &expected) {
  int device = -1;
  const auto status = cudaGetDevice(&device);
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
  if (!x.attr("is_cuda").cast<bool>() ||
      x.attr("get_device")().cast<int>() != device ||
      !x.attr("is_contiguous")().cast<bool>() || shape(x) != expected ||
      py::str(x.attr("dtype")).cast<std::string>() != dtype)
    throw std::invalid_argument(std::string("FP8 expected contiguous CUDA ") +
                                dtype + " with matching shape on the current device");
}
inline unsigned dimension(std::uint64_t n) {
  if (n > std::numeric_limits<unsigned>::max() - 128U)
    throw std::invalid_argument("FP8 dimension is too large");
  return static_cast<unsigned>(n);
}
inline void disjoint(const py::object &output, const py::object &input) {
  const auto a = reinterpret_cast<std::uintptr_t>(data(output));
  const auto b = reinterpret_cast<std::uintptr_t>(data(input));
  const auto na = output.attr("numel")().cast<std::uint64_t>() *
                  output.attr("element_size")().cast<unsigned>();
  const auto nb = input.attr("numel")().cast<std::uint64_t>() *
                  input.attr("element_size")().cast<unsigned>();
  if (na && nb && (a <= b ? b - a < na : a - b < nb))
    throw std::invalid_argument("FP8 output must not alias another operand");
}

inline cuda::fp8::Matrix matrix(const py::object &q, const py::object &s,
                                unsigned columns, unsigned block_rows) {
  const auto qs = shape(q);
  if (qs.size() != 2 || (block_rows != 1U && block_rows != 128U))
    throw std::invalid_argument("FP8 GEMM requires a matrix and block_rows 1 or 128");
  const auto rows = dimension(qs[0]);
  const auto padded_k = (dimension(columns) + 127U) / 128U * 128U;
  check(q, "torch.float8_e4m3fn", {rows, padded_k});
  check(s, "torch.float32", {(rows + block_rows - 1U) / block_rows,
                             (columns + 127U) / 128U});
  return {static_cast<const std::uint8_t *>(data(q)),
          static_cast<const float *>(data(s)), rows, columns, padded_k, block_rows};
}

inline void bind(py::module_ &m) {
  m.def("fp8_quantize_weight", [](py::object x, py::object q, py::object s,
                                   py::object qt, py::object st,
                                   std::uintptr_t stream) {
    const auto xs = shape(x);
    if (xs.size() != 2 && xs.size() != 3)
      throw std::invalid_argument("FP8 weight quantization requires rank 2/3");
    const auto rows = dimension(xs[xs.size() - 2]);
    const auto columns = dimension(xs.back());
    const auto batches = xs.size() == 3 ? dimension(xs[0]) : 1U;
    if (batches > 65535U)
      throw std::invalid_argument("FP8 quantization batch exceeds CUDA grid capacity");
    const auto dtype = py::str(x.attr("dtype")).cast<std::string>();
    if (dtype != "torch.bfloat16" && dtype != "torch.float32")
      throw std::invalid_argument("FP8 quantization input must be BF16 or FP32");
    check(x, dtype.c_str(), xs);
    Shape qs{rows, (columns + 127U) / 128U * 128U};
    Shape ss{(rows + 127U) / 128U, (columns + 127U) / 128U};
    Shape qts{columns, (rows + 127U) / 128U * 128U};
    Shape sts{(columns + 127U) / 128U, (rows + 127U) / 128U};
    if (xs.size() == 3)
      for (auto *value : {&qs, &ss, &qts, &sts}) value->insert(value->begin(), batches);
    check(q, "torch.float8_e4m3fn", qs); check(s, "torch.float32", ss);
    check(qt, "torch.float8_e4m3fn", qts); check(st, "torch.float32", sts);
    const py::object operands[] = {x, q, s, qt, st};
    for (unsigned i = 1; i < 5U; ++i)
      for (unsigned j = 0; j < i; ++j) disjoint(operands[i], operands[j]);
    if (reinterpret_cast<std::uintptr_t>(data(q)) % 4U ||
        reinterpret_cast<std::uintptr_t>(data(qt)) % 4U)
      throw std::invalid_argument("FP8 weight output data must be 4-byte aligned");
    cuda::fp8::quantize_weight(data(x), dtype == "torch.bfloat16"
        ? cuda::fp8::InputType::BFloat16 : cuda::fp8::InputType::Float32,
        static_cast<std::uint8_t *>(data(q)), static_cast<float *>(data(s)),
        static_cast<std::uint8_t *>(data(qt)), static_cast<float *>(data(st)),
        batches, rows, columns, reinterpret_cast<cudaStream_t>(stream));
  });
  m.def("fp8_quantize", [](py::object x, py::object q, py::object s,
                            unsigned block_rows, bool transpose,
                            std::uintptr_t stream) {
    const auto xs = shape(x);
    if ((xs.size() != 2 && xs.size() != 3) ||
        (block_rows != 1U && block_rows != 128U))
      throw std::invalid_argument("FP8 quantization requires rank 2/3 and block_rows 1/128");
    const auto rows = dimension(xs[xs.size() - 2]);
    const auto columns = dimension(xs.back());
    const auto batches = xs.size() == 3 ? dimension(xs[0]) : 1U;
    if (batches > 65535U)
      throw std::invalid_argument("FP8 quantization batch exceeds CUDA grid capacity");
    const auto dtype = py::str(x.attr("dtype")).cast<std::string>();
    if (dtype != "torch.bfloat16" && dtype != "torch.float32")
      throw std::invalid_argument("FP8 quantization input must be BF16 or FP32");
    check(x, dtype.c_str(), xs);
    const auto out_rows = transpose ? columns : rows;
    const auto out_cols = transpose ? rows : columns;
    Shape qs{out_rows, (out_cols + 127U) / 128U * 128U};
    Shape ss{(out_rows + block_rows - 1U) / block_rows, (out_cols + 127U) / 128U};
    if (xs.size() == 3) {
      qs.insert(qs.begin(), batches);
      ss.insert(ss.begin(), batches);
    }
    check(q, "torch.float8_e4m3fn", qs);
    check(s, "torch.float32", ss);
    disjoint(q, x); disjoint(s, x); disjoint(q, s);
    cuda::fp8::quantize(data(x), dtype == "torch.bfloat16"
        ? cuda::fp8::InputType::BFloat16 : cuda::fp8::InputType::Float32,
        static_cast<std::uint8_t *>(data(q)), static_cast<float *>(data(s)),
        batches, rows, columns, block_rows, transpose,
        reinterpret_cast<cudaStream_t>(stream));
  });
  m.def("fp8_transpose_weight", [](py::object q, py::object s,
                                    py::object qt, py::object st,
                                    unsigned columns, std::uintptr_t stream) {
    const auto qs = shape(q);
    if (qs.size() != 2 && qs.size() != 3)
      throw std::invalid_argument("FP8 weight must have rank 2 or 3");
    const auto rows = dimension(qs[qs.size() - 2]);
    const auto batches = qs.size() == 3 ? dimension(qs[0]) : 1U;
    if (batches > 65535U)
      throw std::invalid_argument("FP8 weight batch exceeds CUDA grid capacity");
    Shape expected_q{rows, (dimension(columns) + 127U) / 128U * 128U};
    Shape expected_s{(rows + 127U) / 128U, (columns + 127U) / 128U};
    Shape expected_qt{columns, (rows + 127U) / 128U * 128U};
    Shape expected_st{(columns + 127U) / 128U, (rows + 127U) / 128U};
    if (qs.size() == 3)
      for (auto *value : {&expected_q, &expected_s, &expected_qt, &expected_st})
        value->insert(value->begin(), batches);
    check(q, "torch.float8_e4m3fn", expected_q);
    check(s, "torch.float32", expected_s);
    check(qt, "torch.float8_e4m3fn", expected_qt);
    check(st, "torch.float32", expected_st);
    disjoint(qt, q); disjoint(qt, s); disjoint(st, q); disjoint(st, s); disjoint(qt, st);
    cuda::fp8::Matrix input{static_cast<const std::uint8_t *>(data(q)),
        static_cast<const float *>(data(s)), rows, columns,
        dimension(qs.back()), 128U};
    cuda::fp8::transpose_weight(input, static_cast<std::uint8_t *>(data(qt)),
        static_cast<float *>(data(st)), batches, reinterpret_cast<cudaStream_t>(stream));
  });
  m.def("fp8_gemm", [](py::object aq, py::object as, py::object bq,
                        py::object bs, py::object output, unsigned columns,
                        unsigned a_block_rows, unsigned b_block_rows,
                        bool accumulate, std::uintptr_t stream) {
    const auto a = matrix(aq, as, columns, a_block_rows);
    const auto b = matrix(bq, bs, columns, b_block_rows);
    const auto dtype = py::str(output.attr("dtype")).cast<std::string>();
    if (dtype != "torch.bfloat16" && dtype != "torch.float32")
      throw std::invalid_argument("FP8 GEMM output must be BF16 or FP32");
    check(output, dtype.c_str(), {a.rows, b.rows});
    for (const auto *input : {&aq, &as, &bq, &bs})
      disjoint(output, *input);
    cuda::fp8::gemm(a, b, data(output), dtype == "torch.float32", accumulate,
                    reinterpret_cast<cudaStream_t>(stream));
  });
}
} // namespace weftmoe::python_fp8
