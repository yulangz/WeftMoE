// Grouped NVFP4 NT GEMM for M-grouped activation gradients and K-grouped weight gradients.
// Expert boundaries are computed on device. Each output tile has one owner.
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <stdexcept>
#include "nvfp4_quantize.cuh"

namespace weft_nvfp4 {
using namespace cute;
using F4 = cutlass::float_e2m1_t;
using SF = cutlass::float_ue4m3_t;
using BF16 = cutlass::bfloat16_t;
using Shape3 = Shape<int, int, int>;
using Tile = Shape<_128, _128, _256>;
using Cluster = Shape<_1, _1, _1>;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Arch = cutlass::arch::Sm100;
using Ops = cutlass::arch::OpClassBlockScaledTensorOp;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, Ops, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float, BF16, Row *, 8,
    BF16, Row *, 8, cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, Ops, cutlass::nv_float4_t<F4>, Row *, 32, cutlass::nv_float4_t<F4>, Col *, 32, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;
using Kernel =
    cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<Shape3>, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
constexpr int kMaxExperts = 512;

struct alignas(256) Groups {
    Shape3 shapes[kMaxExperts];
    F4 const *a[kMaxExperts];
    F4 const *b[kMaxExperts];
    SF const *sa[kMaxExperts];
    SF const *sb[kMaxExperts];
    BF16 *d[kMaxExperts];
    BF16 const *c[kMaxExperts];
    typename Kernel::InternalStrideA da[kMaxExperts];
    typename Kernel::InternalStrideB db[kMaxExperts];
    typename Kernel::InternalStrideD dd[kMaxExperts];
    typename Mainloop::InternalLayoutSFA la[kMaxExperts];
    typename Mainloop::InternalLayoutSFB lb[kMaxExperts];
};

__global__ void descriptors(Groups *p, int const *ends, int groups, int outer, int cols, bool k_grouped,
                            uint8_t const *a, uint8_t const *sa, uint8_t const *b, uint8_t const *sb, BF16 *d) {
    int e = threadIdx.x + blockIdx.x * blockDim.x;
    if (e >= groups)
        return;
    int64_t start = e ? ends[e - 1] : 0;
    int extent = ends[e] - start;
    int m = k_grouped ? outer : extent, n = cols, k = k_grouped ? extent : outer;
    int64_t oa = start * outer;
    int64_t ob = k_grouped ? start * cols : int64_t(e) * n * k;
    p->shapes[e] = make_shape(m, n, k);
    p->a[e] = reinterpret_cast<F4 const *>(a + oa / 2);
    p->b[e] = reinterpret_cast<F4 const *>(b + ob / 2);
    p->sa[e] = reinterpret_cast<SF const *>(sa + oa / 16);
    p->sb[e] = reinterpret_cast<SF const *>(sb + ob / 16);
    p->d[e] = d + (k_grouped ? int64_t(e) * m * n : start * n);
    p->c[e] = p->d[e];
    p->da[e] = {int64_t(k), _1{}, _0{}};
    p->db[e] = {int64_t(k), _1{}, _0{}};
    p->dd[e] = {int64_t(n), _1{}, _0{}};
    p->la[e] = Mainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    p->lb[e] = Mainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
}

Gemm::Arguments arguments(Groups *p, int count, float alpha, int device, int sms) {
    cutlass::KernelHardwareInfo info{};
    info.device_id = device;
    info.sm_count = sms;
    Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                         {count, p->shapes, nullptr},
                         {p->a, p->da, p->b, p->db, p->sa, p->la, p->sb, p->lb},
                         {{}, p->c, p->dd, p->d, p->dd},
                         info};
    args.epilogue.thread.alpha = alpha;
    args.epilogue.thread.beta = 0.f;
    return args;
}

size_t workspace_bytes(int groups, int device, int sms) {
    if (groups < 1 || groups > kMaxExperts)
        throw std::invalid_argument("invalid local expert count");
    // No device pointers are dereferenced when CUTLASS sizes descriptor workspace.
    static Groups sizing{};
    return sizeof(Groups) + Gemm::get_workspace_size(arguments(&sizing, groups, 1.f, device, sms));
}

void run(uintptr_t a, uintptr_t sa, uintptr_t b, uintptr_t sb, uintptr_t out, uintptr_t ends, int groups,
         int outer, int cols, bool k_grouped, float alpha, int device, int sms, uintptr_t storage,
         size_t capacity, uintptr_t stream_ptr) {
    auto *p = reinterpret_cast<Groups *>(storage);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (capacity < workspace_bytes(groups, device, sms))
        throw std::invalid_argument("undersized NVFP4 GEMM workspace");
    descriptors<<<(groups + 63) / 64, 64, 0, stream>>>(
        p, reinterpret_cast<int const *>(ends), groups, outer, cols, k_grouped,
        reinterpret_cast<uint8_t const *>(a), reinterpret_cast<uint8_t const *>(sa),
        reinterpret_cast<uint8_t const *>(b), reinterpret_cast<uint8_t const *>(sb),
        reinterpret_cast<BF16 *>(out));
    Gemm gemm;
    auto args = arguments(p, groups, alpha, device, sms);
    auto status = gemm.initialize(args, p + 1, stream);
    if (status == cutlass::Status::kSuccess)
        status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(cutlassGetStatusString(status));
}
} // namespace weft_nvfp4

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_with_layer_scale", &weft_nvfp4::quantize);
    m.def("workspace_bytes", &weft_nvfp4::workspace_bytes);
    m.def("run", &weft_nvfp4::run);
}
