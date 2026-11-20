// Host ABI for the routed CUTLASS/CuTe kernel. Requires the CUDA driver only.
#include <algorithm>
#include <array>
#include <cmath>
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>

#define WEFT_HOST_ASSERT(condition)                                                                               \
    do {                                                                                                          \
        if (!(condition))                                                                                         \
            throw std::invalid_argument(#condition);                                                              \
    } while (0)
#include "fused/scheduler/mega_moe.cuh"

namespace py = pybind11;
namespace weftmoe::sm100 {
using Ptr = uintptr_t;

static void check(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char *msg = nullptr;
        cuGetErrorString(status, &msg);
        throw std::runtime_error(msg ? msg : "CUDA driver error");
    }
}

static int align_up(int n, int a) { return (n + a - 1) / a * a; }

struct Workspace {
    int ranks, experts, capacity, topk, h, i, ring = 0, sf_ring = 0;
    bool fp4;
    int64_t bytes;
    std::array<Ptr, 8> offsets;
    Workspace(int ranks, int experts, int cap, int topk, int h, int i, int sms, bool fp4)
        : ranks(ranks), experts(experts), capacity(align_up(cap, layout::kLCMCandidateBlockM)), topk(topk), h(h),
          i(i), fp4(fp4) {
        for (int m : layout::kCandidateBlockM) {
            int blocks = math::ceil_div(capacity * ranks * topk, m) + experts / ranks;
            ring = std::max(ring, m * sched::get_num_max_live_pool_blocks(blocks, sms, h, i));
        }
        ring = align_up(ring, layout::kLCMCandidateBlockM);
        if (fp4)
            for (int m : layout::kCandidateBlockM)
                sf_ring = std::max(sf_ring, layout::get_num_sf_ring_tokens(ring, m));
        layout::MegaMoEBuffer b(nullptr, h, i, ranks, experts, capacity, topk, ring, sf_ring, fp4, 0, true, fp4);
        bytes = b.get_num_bytes();
        offsets = {(Ptr)b.input_token_buffer.base,    (Ptr)b.input_sf_buffer.base,
                   (Ptr)b.input_topk_idx_buffer.base, (Ptr)b.input_topk_weights_buffer.base,
                   (Ptr)b.l1_token_buffer.base,       (Ptr)b.l1_sf_buffer.base,
                   (Ptr)b.l2_token_buffer.base,       (Ptr)b.l2_sf_buffer.base};
    }
};

static CUtensorMap tensor_map(Ptr ptr, CUtensorMapDataType type, std::vector<cuuint64_t> dims,
                              std::vector<cuuint64_t> strides, std::vector<cuuint32_t> box, int swizzle) {
    CUtensorMap map;
    cuuint32_t steps[] = {1, 1, 1, 1, 1};
    auto mode = swizzle == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                : swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                : swizzle == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                                : CU_TENSOR_MAP_SWIZZLE_NONE;
    check(cuTensorMapEncodeTiled(&map, type, dims.size(), (void *)ptr, dims.data(), strides.data(), box.data(),
                                 steps, CU_TENSOR_MAP_INTERLEAVE_NONE, mode, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

struct Kernel {
    CUmodule module;
    CUfunction fn;
    int smem, sms;
    Kernel(const std::string &path, int smem, int sms) : smem(smem), sms(sms) {
        check(cuModuleLoad(&module, path.c_str()));
        try {
            unsigned count;
            check(cuModuleGetFunctionCount(&count, module));
            if (count != 1)
                throw std::runtime_error("Expected one routed kernel per specialization");
            check(cuModuleEnumerateFunctions(&fn, 1, module));
            check(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
        } catch (...) {
            cuModuleUnload(module);
            throw;
        }
    }
    ~Kernel() { cuModuleUnload(module); }

    void run(const Workspace &b, const std::vector<int64_t> &peers, int rank, int rows, int m, int k, int store_m,
             Ptr output, std::array<Ptr, 4> weights, std::array<Ptr, 2> sf_strides, std::array<Ptr, 6> saved,
             Ptr domains, std::array<float, 3> scales, Ptr stream) const {
        using U = cuuint64_t;
        using B = cuuint32_t;
        const U h = b.h, i = b.i, e = b.experts / b.ranks;
        const bool fp4 = b.fp4;
        const U pack = fp4 ? 2 : 1, size = fp4 ? 1 : 2;
        const auto type = fp4 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        const B tile = k / pack, sm = store_m, load = m / 2;
        const int swizzle = tile * size;
        const Ptr base = peers.at(rank);
        auto acts = [&](int idx, U cols, B inner, B outer, int sw) {
            return tensor_map(base + b.offsets[idx], type, {cols / pack, U(b.ring)}, {cols / pack * size},
                              {inner, outer}, sw);
        };
        auto weight = [&](Ptr ptr, U n, U cols) {
            const U width = cols / pack, stride = width * size;
            return tensor_map(ptr, type, {width, n / 2, 2, e}, {stride, n / 2 * stride, n * stride},
                              {tile, 128, 1, 1}, swizzle);
        };
        auto sf = [&](Ptr ptr, U n, U cols, U groups, B block, U stride) {
            return tensor_map(ptr, CU_TENSOR_MAP_DATA_TYPE_INT32, {n, cols / 64 * groups}, {stride * 4},
                              {block, B(k / 64)}, 0);
        };
        auto a1 = acts(4, h, tile, load, swizzle);
        auto w1 = fp4 ? weight(weights[0], 2 * i, h)
                      : tensor_map(weights[0], type, {h, 8, 2, i / 8, e},
                                   {h * 2, h * i * 2, 8 * h * 2, 2 * i * h * 2}, {64, 8, 2, 8, 1}, 128);
        auto mid = acts(6, i, B(64 / pack), sm, int(64 / pack * size));
        auto a2 = acts(6, i, tile, load, swizzle);
        auto w2 = weight(weights[2], h, i);
        CUtensorMap sa1{}, sw1{}, sa2{}, sw2{};
        std::vector<void *> maps;
        if (fp4) {
            auto sf_block = B(align_up(m, 128));
            sa1 = sf(base + b.offsets[5], b.sf_ring, h, 1, sf_block, b.sf_ring);
            sa2 = sf(base + b.offsets[7], b.sf_ring, i, 1, sf_block, b.sf_ring);
            sw1 = sf(weights[1], 2 * i, h, e, 128, sf_strides[0]);
            sw2 = sf(weights[3], h, i, e, 128, sf_strides[1]);
            maps = {&a1, &sa1, &w1, &sw1, &mid, &a2, &sa2, &w2, &sw2};
        } else {
            maps = {&a1, &w1, &mid, &a2, &w2};
        }
        layout::SymBuffer<> sym(peers, rank);
        layout::BF16Checkpoint checkpoint{(int *)saved[0],      (int *)saved[1],   (uint16_t *)saved[2],
                                          (uint16_t *)saved[3], (float *)saved[4], (int *)saved[5]};
        std::vector<void *> args{&output, &rows, &sym};
        args.insert(args.end(), maps.begin(), maps.end());
        args.push_back(&domains);
        float inverse = 1.f / scales[1];
        if (fp4)
            args.insert(args.end(), {&scales[0], &scales[1], &inverse, &scales[2]});
        args.push_back(&checkpoint);
        CUlaunchAttribute attr{};
        attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attr.value.clusterDim = {2, 1, 1};
        CUlaunchConfig cfg{};
        cfg.gridDimX = sms;
        cfg.gridDimY = cfg.gridDimZ = 1;
        cfg.blockDimX = 512;
        cfg.blockDimY = cfg.blockDimZ = 1;
        cfg.sharedMemBytes = smem;
        cfg.hStream = (CUstream)stream;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        check(cuLaunchKernelEx(&cfg, fn, args.data(), nullptr));
    }
};
} // namespace weftmoe::sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    using namespace weftmoe::sm100;
    py::class_<Workspace>(mod, "Workspace")
        .def(py::init<int, int, int, int, int, int, int, bool>())
        .def_readonly("capacity", &Workspace::capacity)
        .def_readonly("ring", &Workspace::ring)
        .def_readonly("sf_ring", &Workspace::sf_ring)
        .def_readonly("bytes", &Workspace::bytes)
        .def_readonly("offsets", &Workspace::offsets);
    mod.attr("task_bytes") = sizeof(sched::TaskInfo<true>);
    py::class_<Kernel>(mod, "Kernel").def(py::init<const std::string &, int, int>()).def("run", &Kernel::run);
}
