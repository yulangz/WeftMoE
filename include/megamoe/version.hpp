#pragma once

#include <cstdint>

#if defined(_WIN32)
#if defined(MEGAMOE_BUILD_SHARED)
#define MEGAMOE_API __declspec(dllexport)
#else
#define MEGAMOE_API __declspec(dllimport)
#endif
#define MEGAMOE_PUBLIC_API_BEGIN
#define MEGAMOE_PUBLIC_API_END
#else
#define MEGAMOE_API __attribute__((visibility("default")))
#define MEGAMOE_PUBLIC_API_BEGIN _Pragma("GCC visibility push(default)")
#define MEGAMOE_PUBLIC_API_END _Pragma("GCC visibility pop")
#endif

MEGAMOE_PUBLIC_API_BEGIN

namespace megamoe {

// Version 2 keeps repository-only diagnostics outside the installed API. It
// is intentionally a
// breaking boundary: this repository has no external compatibility contract.
inline constexpr std::uint32_t kPublicApiVersionMajor = 2;
inline constexpr std::uint32_t kPublicApiVersionMinor = 0;
inline constexpr std::uint32_t kPublicApiVersionPatch = 0;

} // namespace megamoe

MEGAMOE_PUBLIC_API_END
