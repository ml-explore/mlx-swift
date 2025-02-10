/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_METAL_H
#define MLX_METAL_H

#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
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
int mlx_metal_clear_cache();

typedef struct mlx_metal_device_info_t_ {
  char architecture[256];
  size_t max_buffer_length;
  size_t max_recommended_working_set_size;
  size_t memory_size;
} mlx_metal_device_info_t;
mlx_metal_device_info_t mlx_metal_device_info();

int mlx_metal_get_active_memory(size_t* res);
int mlx_metal_get_cache_memory(size_t* res);
int mlx_metal_get_peak_memory(size_t* res);
int mlx_metal_is_available(bool* res);
int mlx_metal_reset_peak_memory();
int mlx_metal_set_cache_limit(size_t* res, size_t limit);
int mlx_metal_set_memory_limit(size_t* res, size_t limit, bool relaxed);
int mlx_metal_set_wired_limit(size_t* res, size_t limit);
int mlx_metal_start_capture(const char* path);
int mlx_metal_stop_capture();
/**@}*/

#ifdef __cplusplus
}
#endif

#endif
