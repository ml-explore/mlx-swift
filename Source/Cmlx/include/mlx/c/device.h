/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_DEVICE_H
#define MLX_DEVICE_H

#include <stdbool.h>

#include "mlx/c/string.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup mlx_device Device
 * MLX device object.
 */
/**@{*/

/**
 * A MLX device object.
 */
typedef struct mlx_device_ {
  void* ctx;
} mlx_device;

/**
 * Device type.
 */
typedef enum mlx_device_type_ { MLX_CPU, MLX_GPU } mlx_device_type;

/**
 * Returns a new empty device.
 */
mlx_device mlx_device_new();

/**
 * Returns a new device of specified `type`, with specified `index`.
 */
mlx_device mlx_device_new_type(mlx_device_type type, int index);
/**
 * Free a device.
 */
int mlx_device_free(mlx_device dev);
/**
 * Set device to provided src device.
 */
int mlx_device_set(mlx_device* dev, const mlx_device src);
/**
 * Get device description.
 */
int mlx_device_tostring(mlx_string* str, mlx_device dev);
/**
 * Check if devices are the same.
 */
bool mlx_device_equal(mlx_device lhs, mlx_device rhs);
/**
 * Returns the index of the device.
 */
int mlx_device_get_index(int* index, mlx_device dev);
/**
 * Returns the type of the device.
 */
int mlx_device_get_type(mlx_device_type* type, mlx_device dev);
/**
 * Returns the default MLX device.
 */
int mlx_get_default_device(mlx_device* dev);
/**
 * Set the default MLX device.
 */
int mlx_set_default_device(mlx_device dev);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
