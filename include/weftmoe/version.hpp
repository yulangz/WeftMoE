#pragma once

#include <cstdint>

#if defined(_WIN32)
#if defined(WEFTMOE_BUILD_SHARED)
#define WEFTMOE_API __declspec(dllexport)
#else
#define WEFTMOE_API __declspec(dllimport)
#endif
#define WEFTMOE_PUBLIC_API_BEGIN
#define WEFTMOE_PUBLIC_API_END
#else
#define WEFTMOE_API __attribute__((visibility("default")))
#define WEFTMOE_PUBLIC_API_BEGIN _Pragma("GCC visibility push(default)")
#define WEFTMOE_PUBLIC_API_END _Pragma("GCC visibility pop")
#endif

WEFTMOE_PUBLIC_API_BEGIN

namespace weftmoe {

// Build-time switch. Building with WEFTMOE_ENABLE_GIN_RDMA=0 produces a
// single-node-only binary: the RDMA/GIN device-communicator requirements and
// window allocations, the GinRdma forward kernel, and the GIN branches and
// system-scope fences inside the notify exchange are all compiled out, and the
// runtime rejects EP teams spanning more than one node. The default keeps the
// full multi-node implementation.
#ifndef WEFTMOE_ENABLE_GIN_RDMA
#define WEFTMOE_ENABLE_GIN_RDMA 1
#endif

// Version 2 keeps repository-only diagnostics outside the installed API. It
// is intentionally a
// breaking boundary: this repository has no external compatibility contract.
inline constexpr std::uint32_t kPublicApiVersionMajor = 2;
inline constexpr std::uint32_t kPublicApiVersionMinor = 0;
inline constexpr std::uint32_t kPublicApiVersionPatch = 0;

} // namespace weftmoe

WEFTMOE_PUBLIC_API_END
