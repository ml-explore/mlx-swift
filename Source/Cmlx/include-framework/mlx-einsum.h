#ifdef __cplusplus
// Copyright © 2024 Apple Inc.
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <Cmlx/mlx-api.h>
#include <Cmlx/mlx-array.h>
#include <Cmlx/mlx-utils.h>

namespace mlx::core {

MLX_API std::pair<std::vector<std::vector<int>>, std::string> einsum_path(
    const std::string& subscripts,
    const std::vector<array>& operands);

MLX_API array einsum(
    const std::string& subscripts,
    const std::vector<array>& operands,
    StreamOrDevice s = {});

} // namespace mlx::core
#endif
