/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_CUDA_H
#define MLX_CUDA_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <Cmlx/mlx-c-array.h>
#include <Cmlx/mlx-c-closure.h>
#include <Cmlx/mlx-c-distributed_group.h>
#include <Cmlx/mlx-c-io_types.h>
#include <Cmlx/mlx-c-map.h>
#include <Cmlx/mlx-c-stream.h>
#include <Cmlx/mlx-c-string.h>
#include <Cmlx/mlx-c-vector.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup cuda Cuda specific operations
 */
/**@{*/

int mlx_cuda_is_available(bool* res);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
