/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_METAL_H
#define MLX_METAL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup metal Metal specific operations
 */
/**@{*/

/**
 * @deprecated Use mlx_device_info instead.
 * This struct is kept for backwards compatibility.
 */
typedef struct mlx_metal_device_info_t_ {
  char architecture[256];
  size_t max_buffer_length;
  size_t max_recommended_working_set_size;
  size_t memory_size;
} mlx_metal_device_info_t;

/**
 * @deprecated Use mlx_device_info_get() instead.
 * Get Metal device information (deprecated).
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((deprecated("Use mlx_device_info_get() instead")))
#elif defined(_MSC_VER)
__declspec(deprecated("Use mlx_device_info_get() instead"))
#endif
mlx_metal_device_info_t mlx_metal_device_info(void);

int mlx_metal_is_available(bool* res);
int mlx_metal_start_capture(const char* path);
int mlx_metal_stop_capture(void);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
