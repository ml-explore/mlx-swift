// Copyright © 2024 Apple Inc.

#include <TargetConditionals.h>

// select the correct cpu compile system based on TARGET_OS
#if TARGET_OS_IOS || TARGET_OS_VISION
#include "../mlx/mlx/backend/no_cpu/compiled.cpp"
#else
#include "../mlx/mlx/backend/cpu/compiled.cpp"
#endif

