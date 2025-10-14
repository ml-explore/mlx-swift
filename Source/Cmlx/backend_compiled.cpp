// Backend compiled selector
// Copyright © 2024 Apple Inc.
// This file includes the appropriate backend based on build configuration

#ifdef MLX_BUILD_CUDA
    // Include CUDA backend
    #include "mlx/mlx/backend/cuda/compiled.cpp"
    #include "mlx/mlx/backend/no_cpu/compiled.cpp"
#else
    // Include CPU backend (default)
    #include "mlx/mlx/backend/cpu/compiled.cpp"
    #include "mlx/mlx/backend/cuda/no_cuda.cpp"
#endif