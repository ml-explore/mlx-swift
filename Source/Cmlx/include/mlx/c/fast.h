/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_FAST_H
#define MLX_FAST_H

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
 * \defgroup fast Fast custom operations
 */
/**@{*/
int mlx_fast_affine_dequantize(
    mlx_array* res,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases,
    int group_size,
    int bits,
    const mlx_stream s);
int mlx_fast_affine_quantize(
    mlx_array* res_0,
    mlx_array* res_1,
    mlx_array* res_2,
    const mlx_array w,
    int group_size,
    int bits,
    const mlx_stream s);
int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);

typedef struct mlx_fast_metal_kernel_ {
  void* ctx;
} mlx_fast_metal_kernel;
mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const char* source,
    const char* header);
void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);
int mlx_fast_metal_kernel_add_input_name(
    mlx_fast_metal_kernel cls,
    const char* name);
int mlx_fast_metal_kernel_set_input_names(
    mlx_fast_metal_kernel cls,
    int num,
    ...);
int mlx_fast_metal_kernel_add_output_name(
    mlx_fast_metal_kernel cls,
    const char* name);
int mlx_fast_metal_kernel_set_output_names(
    mlx_fast_metal_kernel cls,
    int num,
    ...);
int mlx_fast_metal_kernel_set_contiguous_rows(
    mlx_fast_metal_kernel cls,
    bool flag);
int mlx_fast_metal_kernel_set_atomic_outputs(
    mlx_fast_metal_kernel cls,
    bool flag);

int mlx_fast_metal_kernel_add_output_arg(
    mlx_fast_metal_kernel cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_set_grid(
    mlx_fast_metal_kernel cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_metal_kernel_set_thread_group(
    mlx_fast_metal_kernel cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_metal_kernel_set_init_value(
    mlx_fast_metal_kernel cls,
    float value);
int mlx_fast_metal_kernel_set_verbose(mlx_fast_metal_kernel cls, bool verbose);
int mlx_fast_metal_kernel_add_template_arg_dtype(
    mlx_fast_metal_kernel cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_add_template_arg_int(
    mlx_fast_metal_kernel cls,
    const char* name,
    int value);
int mlx_fast_metal_kernel_add_template_arg_bool(
    mlx_fast_metal_kernel cls,
    const char* name,
    bool value);

int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_stream stream);

int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight,
    float eps,
    const mlx_stream s);
int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const mlx_array mask /* may be null */,
    mlx_optional_int memory_efficient_threshold,
    const mlx_stream s);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif
