// TurboQuant function declarations for C++ namespace
// These extend mlx::core::fast with TurboQuant operations
#pragma once

#include "mlx/mlx.h"

namespace mlx::core::fast {

array turbo_encode_k(const array& keys, StreamOrDevice s = {});
array turbo_encode_v(const array& values, StreamOrDevice s = {});
array turbo_decode_k(const array& packed, StreamOrDevice s = {});
array turbo_decode_v(const array& packed, StreamOrDevice s = {});

} // namespace mlx::core::fast
