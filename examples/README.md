# Minimal lifecycle examples

These snippets show ownership and call order. Application bootstrap supplies
the rank-local `NcclBootstrapId`; the router supplies typed device-resident
route, gate and activation leases. None of the public handles exposes a native
CUDA, IPC or GIN pointer.

## Single-host Forward

```cpp
using namespace weftmoe;

auto base = cuda::make_nccl_session("ep-team", world, rank, 1, device,
                                    bootstrap_id);
auto exec = cuda::make_cuda_execution("forward", device);
auto notify_pool = cuda::make_device_notify_resource_pool(
    base, exec, notify_pool_config);
auto data_pool = cuda::make_activation_transport_pool(
    base.session(), exec, transport_pool_config);

for (std::uint64_t generation = 1; generation != steps + 1; ++generation) {
  auto session = cuda::make_nccl_generation(base, "ep-team", generation);
  cuda::CudaNotifyGenerationCoordinator notify(session, exec, notify_pool);
  auto notified = notify.run_device(
      {runtime_config, router.routes(generation), router.gates(generation),
       max_tokens_per_rank});

  cuda::LocalCudaP2pForwardMegaKernel forward(
      session, exec, data_pool);
  auto fwd = forward.run(notified.dispatch(), router.activations(generation),
                         weights, forward_config);
  consumer.consume(std::move(fwd));
}
```

Destroy result/consumer handles before their pools when deterministic teardown
latency matters. The pools nevertheless own their backend state strongly and
drain live leases safely if application objects are destroyed in another order.

## Two-host GIN

Create the same Notify objects, but enable `enable_gin_rdma` in the transport
pool and construct the explicitly typed GIN adapters:

```cpp
cuda::GinRdmaForwardMegaKernel forward(session, exec, data_pool);
auto fwd = forward.run(notified.dispatch(), router.activations(generation),
                       weights, forward_config);
```

All ranks use a common raw stable session identity and runtime homogeneous
topology. GIN is available only when the capability report says
`gin_usable=true`; otherwise construction fails with `BackendUnavailable`.
Select `NcclSocketFallbackForwardMegaKernel` explicitly for the host-enqueued
correctness fallback—there is no enum that silently
changes a requested GIN backend into Socket or simulation.

The complete executable bootstrap/launcher reference is
`tests/run_multinode_nccl.py`; build and repository-external NCCL instructions
are in `docs/build.md`.
