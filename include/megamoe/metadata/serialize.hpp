#pragma once

#include <string>

#include "megamoe/metadata/dispatch_handle.hpp"

namespace megamoe {

// Stable machine-readable serialization of every deterministic metadata field.
// Runtime-only provenance tokens are intentionally excluded.
std::string serialize_dispatch_metadata(const DispatchHandle& handle);

}  // namespace megamoe
