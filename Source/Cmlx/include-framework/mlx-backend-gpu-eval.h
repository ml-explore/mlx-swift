#ifdef __cplusplus
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>

#include <Cmlx/mlx-array.h>
#include <Cmlx/mlx-stream.h>

namespace mlx::core::gpu {

void new_stream(Stream stream);
void eval(array& arr);
void finalize(Stream s);
void synchronize(Stream s);

} // namespace mlx::core::gpu
#endif
