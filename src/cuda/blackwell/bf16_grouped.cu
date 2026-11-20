// CUTLASS grouped NN activation gradients and TN weight gradients.
// Each pair of SMs computes one output tile. Dgrad writes the transposed GEMM
// through a column-major view of the original row-major output.
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <pybind11/pybind11.h>

namespace weftmoe::sm100 {
using namespace cute;
using BF = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Shape3 = Shape<int, int, int>;
template <bool Wgrad> struct Plan {
    using Tile = Shape<_256, _256, _64>;
    using Cluster = Shape<_2, _1, _1>;
    using OutputLayout = std::conditional_t<Wgrad, Row, Col>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, OutputLayout *, 8, BF, OutputLayout *,
        8, cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, BF, Col *, 8, BF,
        std::conditional_t<Wgrad, Row, Col> *, 8, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<Shape3>, Main, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    struct alignas(256) Groups {
        Shape3 shape[512];
        BF const *a[512], *b[512];
        BF *d[512];
        typename Kernel::InternalStrideA da[512];
        typename Kernel::InternalStrideB db[512];
        typename Kernel::InternalStrideD dd[512];
    };
    static auto args(Groups *p, int count, int device, int sms) {
        cutlass::KernelHardwareInfo info{};
        info.device_id = device;
        info.sm_count = sms;
        typename Gemm::Arguments a{cutlass::gemm::GemmUniversalMode::kGrouped,
                                   {count, p->shape, nullptr},
                                   {p->a, p->da, p->b, p->db},
                                   {{}, nullptr, p->dd, p->d, p->dd},
                                   info};
        a.epilogue.thread.alpha = 1.f;
        a.epilogue.thread.beta = 0.f;
        return a;
    }
};

template <bool Wgrad>
__global__ void describe(typename Plan<Wgrad>::Groups *p, BF const *a, BF const *b, BF *d, int const *ends,
                         int groups, int outer, int n) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= groups)
        return;
    int64_t start = e ? ends[e - 1] : 0;
    int length = ends[e] - start;
    if constexpr (Wgrad) {
        p->shape[e] = make_shape(outer, n, length);
        p->a[e] = a + start * outer;
        p->b[e] = b + start * n;
        p->d[e] = d + int64_t(e) * outer * n;
        p->da[e] = {_1{}, int64_t(outer), _0{}};
        p->db[e] = {_1{}, int64_t(n), _0{}};
        p->dd[e] = {int64_t(n), _1{}, _0{}};
    } else {
        p->shape[e] = make_shape(n, length, outer);
        p->a[e] = b + int64_t(e) * outer * n;
        p->b[e] = a + start * outer;
        p->d[e] = d + start * n;
        p->da[e] = {_1{}, int64_t(n), _0{}};
        p->db[e] = {int64_t(outer), _1{}, _0{}};
        p->dd[e] = {_1{}, int64_t(n), _0{}};
    }
}

template <bool Wgrad> size_t workspace(int groups, int device, int sms) {
    if (groups <= 0 || groups > 512)
        throw std::invalid_argument("Invalid local expert count");
    using P = Plan<Wgrad>;
    static typename P::Groups sizing{};
    return sizeof(sizing) + P::Gemm::get_workspace_size(P::args(&sizing, groups, device, sms));
}

template <bool Wgrad>
void run(uintptr_t a, uintptr_t b, uintptr_t d, uintptr_t ends, int groups, int outer, int n, int device, int sms,
         uintptr_t storage, size_t capacity, uintptr_t stream_ptr) {
    using P = Plan<Wgrad>;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto p = reinterpret_cast<typename P::Groups *>(storage);
    if (capacity < workspace<Wgrad>(groups, device, sms))
        throw std::invalid_argument("Undersized grouped workspace");
    describe<Wgrad><<<(groups + 63) / 64, 64, 0, stream>>>(p, (BF const *)a, (BF const *)b, (BF *)d,
                                                           (int const *)ends, groups, outer, n);
    typename P::Gemm gemm;
    auto status = gemm.initialize(P::args(p, groups, device, sms), p + 1, stream);
    if (status == cutlass::Status::kSuccess)
        status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(cutlassGetStatusString(status));
}
} // namespace weftmoe::sm100
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace weftmoe::sm100;
    m.def("dgrad_bytes", &workspace<false>);
    m.def("wgrad_bytes", &workspace<true>);
    m.def("dgrad", &run<false>);
    m.def("wgrad", &run<true>);
}
