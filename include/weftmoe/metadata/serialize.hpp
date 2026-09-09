#pragma once

#include <string>

#include "weftmoe/metadata/dispatch_handle.hpp"

namespace weftmoe {

// Stable machine-readable serialization of every deterministic metadata field.
// Runtime-only provenance tokens are intentionally excluded.
std::string serialize_dispatch_metadata(const DispatchHandle& handle);

}  // namespace weftmoe
