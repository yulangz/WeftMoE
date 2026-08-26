# Public API v2 migration

Version 2 is an intentional breaking cleanup for this new repository. The
installed surface contains production session/topology, Notify, typed input
leases, resource pools, backend capability/selection, Forward/Backward owning
handles and production errors/status only.

The following declarations are no longer installed:

- `megamoe::cuda::testing::*` and `megamoe::testing::*`;
- synthetic transport handles/backends and `run_synthetic`;
- `*_for_validation`, download/oracle and mutation helpers;
- unpooled coordinators, fault injection and runtime-shape harnesses;
- raw/private kernel views, IPC/GIN factories and backend handles.

Repository tests include `src/cuda/testing/api.hpp` or
`src/metadata/testing_api.hpp` explicitly. Applications must not add `src/` to
their include path. Replace host test uploads/downloads with the typed
router/autograd producer leases and owning result/consumer handles. Perform
one-time Backward weight preparation with
`prepare_forward_weights_for_backward`, then destroy results before pools and
sessions unless a retained handle intentionally extends pool lifetime.

Run `make install-test` to validate the exact installed manifest and compile
the minimal lifecycle example from an isolated prefix.
