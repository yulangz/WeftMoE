// Each NVLink lane exchanges registered, rank-local slices with the same
// lane of every other node. Each stripe advances its own GIN signal sequence.
#include <cuda_runtime.h>
#include <nccl_device.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <string>

namespace py = pybind11;

static void check(ncclResult_t rc, const char* call) {
    if (rc != ncclSuccess)
        throw std::runtime_error(std::string(call) + ": " + ncclGetErrorString(rc));
}

static int rail_backend(uintptr_t ptr, int nodes) {
    auto host = reinterpret_cast<ncclComm_t>(ptr);
    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    check(ncclCommQueryProperties(host, &properties), "ncclCommQueryProperties");
    if (!properties.deviceApiSupport || ncclTeamRail(host).nRanks != nodes)
        return NCCL_GIN_TYPE_NONE;
    return properties.railedGinType;
}

__global__ void exchange_slices(ncclDevComm_t comm, ncclWindow_t window,
                                size_t source, size_t dest, size_t stride,
                                const int64_t* sizes, bool broadcast, int stripes) {
    if (threadIdx.x % 32 != 0) return;
    const int stripe = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (stripe >= stripes) return;
    ncclGin gin(comm, stripe);
    const ncclTeam_t team = ncclTeamRail(comm);
    for (int step = 1; step < team.nRanks; ++step) {
        const int to = (team.rank + step) % team.nRanks;
        const int from = (team.rank + team.nRanks - step) % team.nRanks;
        // A receiver returns credit only after its preceding stream work has
        // finished reading the slice. Distinct signals survive graph replay.
        gin.signal(team, from, ncclGin_SignalInc{2u * team.rank});
        gin.increaseSignalShadow(2 * to, 1);
        gin.waitSignalMeetShadow(ncclCoopThread{}, 2 * to);
        size_t bytes = sizes[broadcast ? team.rank : to];
        assert(bytes <= stride);
        size_t chunk = ((bytes + stripes - 1) / stripes + 127) & ~size_t(127);
        size_t first = stripe * chunk;
        if (first < bytes) {
            size_t length = min(chunk, bytes - first);
            gin.put(team, to, window, dest + team.rank * stride + first,
                    window, source + (broadcast ? team.rank : to) * stride + first, length,
                    ncclGin_SignalInc{2u * team.rank + 1});
        } else {
            gin.signal(team, to, ncclGin_SignalInc{2u * team.rank + 1});
        }
        gin.increaseSignalShadow(2 * from + 1, 1);
        gin.waitSignalMeetShadow(ncclCoopThread{}, 2 * from + 1);
    }
    gin.flush(ncclCoopThread{});
}

class RailWindow {
    ncclComm_t host;
    ncclDevComm_t dev{};
    ncclWindow_t window = nullptr;
    void* allocation = nullptr;
    bool created = false;
    size_t extent;
    int nodes;
    int contexts;

public:
    RailWindow(uintptr_t ptr, size_t bytes, int expected_nodes, int stripes)
        : host(reinterpret_cast<ncclComm_t>(ptr)), extent(bytes), nodes(expected_nodes) {
        ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        req.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
        req.ginContextCount = stripes;
        req.ginExclusiveContexts = false;
        req.ginSignalCount = 2 * nodes;
        try {
            check(ncclDevCommCreate(host, &req, &dev), "ncclDevCommCreate(RAIL)");
            created = true;
            contexts = std::min(stripes, static_cast<int>(dev.ginContextCount));
            if (contexts < 1) throw std::runtime_error("NCCL created zero GIN contexts");
            check(ncclMemAlloc(&allocation, bytes), "ncclMemAlloc");
            check(ncclCommWindowRegister(host, allocation, bytes, &window,
                                         NCCL_WIN_STRICT_ORDERING), "ncclCommWindowRegister");
        } catch (...) {
            close();
            throw;
        }
    }
    uintptr_t data() const { return reinterpret_cast<uintptr_t>(allocation); }
    int stripes() const { return contexts; }
    void exchange(size_t src, size_t dst, size_t stride, uintptr_t sizes,
                  bool broadcast, uintptr_t raw_stream, bool compact) {
        if (!window || stride > extent / nodes || src > extent - stride * nodes ||
            dst > extent - stride * nodes)
            throw std::out_of_range("GIN window slice exceeds allocation");
        const int warps = compact ? std::min(contexts, 16) : 1;
        exchange_slices<<<(contexts + warps - 1) / warps, warps * 32, 0, reinterpret_cast<cudaStream_t>(raw_stream)>>>(
            dev, window, src, dst, stride, reinterpret_cast<const int64_t*>(sizes), broadcast, contexts);
        auto rc = cudaGetLastError();
        if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
    }
    void close() {
        if (window) { check(ncclCommWindowDeregister(host, window), "ncclCommWindowDeregister"); window = nullptr; }
        if (allocation) { check(ncclMemFree(allocation), "ncclMemFree"); allocation = nullptr; }
        if (created) { check(ncclDevCommDestroy(host, &dev), "ncclDevCommDestroy"); created = false; }
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rail_backend", &rail_backend);
    py::class_<RailWindow>(m, "RailWindow")
        .def(py::init<uintptr_t, size_t, int, int>(), py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("address", &RailWindow::data)
        .def_property_readonly("contexts", &RailWindow::stripes)
        .def("exchange", &RailWindow::exchange)
        .def("close", &RailWindow::close, py::call_guard<py::gil_scoped_release>());
}
