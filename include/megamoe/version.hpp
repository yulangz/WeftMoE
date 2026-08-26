#pragma once

#include <cstdint>

namespace megamoe {

// Version 2 removes validation, synthetic transport, fault-injection and raw
// kernel-view declarations from the installed API.  It is intentionally a
// breaking boundary: this repository has no external compatibility contract.
inline constexpr std::uint32_t kPublicApiVersionMajor = 2;
inline constexpr std::uint32_t kPublicApiVersionMinor = 0;
inline constexpr std::uint32_t kPublicApiVersionPatch = 0;

} // namespace megamoe
