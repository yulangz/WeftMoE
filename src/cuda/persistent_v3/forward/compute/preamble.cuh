#pragma once

// Hopper-only Forward compute grid.  Transport owns generation setup and the
// immutable Dispatch/Return directories; this grid owns only FC1/FC2.  Keeping
// the WGMMA role in a separate kernel and an out-of-line device function is
// intentional: ptxas can now assign the math warpgroup and the TMA
// producer independent dynamic register quotas without carrying any GIN or
// relay live ranges through the WGMMA pipeline.

