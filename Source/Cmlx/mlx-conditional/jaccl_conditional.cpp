// Copyright © 2026 Apple Inc.

#include <TargetConditionals.h>

// JACCL (RDMA over Thunderbolt 5) needs <infiniband/verbs.h>, which ships with
// the macOS 26.2 SDK and later. Build the backend and its library when the
// header is present, and the stub otherwise. librdma is loaded at runtime, so
// on an older macOS MLXDistributed.isAvailable(.jaccl) reports false.
#if TARGET_OS_OSX && __has_include(<infiniband/verbs.h>)
#include "../mlx/mlx/distributed/jaccl/jaccl.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/jaccl.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/mesh.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/rdma.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/ring.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/tcp.cpp"
#else
#include "../mlx/mlx/distributed/jaccl/no_jaccl.cpp"
#endif
