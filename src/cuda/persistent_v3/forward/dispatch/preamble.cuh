#pragma once

// Included by activation_transport.cu after the private Forward launch
// contract, GIN wire records, completion protocol, and peer helpers are
// visible.  This file contains only the fixed 8-RDMA/8-Relay Dispatch state
// machine; WGMMA lives in compute_kernel.cuh and cannot inherit these live
// ranges.

