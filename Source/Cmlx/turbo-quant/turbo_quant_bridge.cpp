// TurboQuant C bridge for Swift bindings
#include "mlx/c/fast.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "turbo_quant_decl.h"

extern "C" int mlx_fast_turbo_encode(
    mlx_array* res_polar_k,
    mlx_array* res_polar_v,
    mlx_array* res_residual_k,
    mlx_array* res_residual_v,
    const mlx_array keys,
    const mlx_array values,
    int k_bits,
    const mlx_stream s) {
    try {
        mlx_array_set_(
            *res_polar_k,
            mlx::core::fast::turbo_encode_k(
                mlx_array_get_(keys),
                mlx_stream_get_(s)));
        mlx_array_set_(
            *res_polar_v,
            mlx::core::fast::turbo_encode_v(
                mlx_array_get_(values),
                mlx_stream_get_(s)));
        *res_residual_k = mlx_array_new();
        *res_residual_v = mlx_array_new();
    } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
    return 0;
}

extern "C" int mlx_fast_turbo_decode_k(
    mlx_array* res,
    const mlx_array packed,
    const mlx_stream s) {
    try {
        mlx_array_set_(
            *res,
            mlx::core::fast::turbo_decode_k(
                mlx_array_get_(packed),
                mlx_stream_get_(s)));
    } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
    return 0;
}

extern "C" int mlx_fast_turbo_decode_v(
    mlx_array* res,
    const mlx_array packed,
    const mlx_stream s) {
    try {
        mlx_array_set_(
            *res,
            mlx::core::fast::turbo_decode_v(
                mlx_array_get_(packed),
                mlx_stream_get_(s)));
    } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
    return 0;
}
