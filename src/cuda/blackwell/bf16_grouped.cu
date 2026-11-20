// CUTLASS grouped NN activation gradients and TN weight gradients.
// Each expert's padded row interval is contiguous. One CTA owns each output tile.
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
using Tile = Shape<_128, _128, _64>;
using Cluster = Shape<_1, _1, _1>;
using Arch = cutlass::arch::Sm100;
using Op = cutlass::arch::OpClassTensorOp;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, Op, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float, BF, Row *, 8, BF,
    Row *, 8, cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;

template <bool Wgrad> struct Plan {
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, Op, BF, std::conditional_t<Wgrad, Col, Row> *, 8, BF, Row *, 8, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(Epilogue::SharedStorage)>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<Shape3>, Main, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    struct alignas(256) Groups {
        Shape3 shape[512];
        BF const *a[512], *b[512], *c[512];
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
                                   {{}, p->c, p->dd, p->d, p->dd},
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
    p->shape[e] = make_shape(Wgrad ? outer : length, n, Wgrad ? length : outer);
    p->a[e] = a + start * outer;
    p->b[e] = b + (Wgrad ? start : int64_t(e) * outer) * n;
    p->d[e] = d + (Wgrad ? int64_t(e) * outer : start) * n;
    p->c[e] = p->d[e];
    if constexpr (Wgrad)
        p->da[e] = {_1{}, int64_t(outer), _0{}};
    else
        p->da[e] = {int64_t(outer), _1{}, _0{}};
    p->db[e] = {_1{}, int64_t(n), _0{}};
    p->dd[e] = {int64_t(n), _1{}, _0{}};
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
