// Copyright © 2024 Apple Inc.

#include <TargetConditionals.h>

// select the correct cpu compile system based on TARGET_OS
#if TARGET_OS_IOS || TARGET_OS_VISION
#include "../mlx/mlx/backend/common/compiled_nocpu.cpp"
#else
#include "../mlx/mlx/backend/common/compiled_cpu.cpp"
#endif

