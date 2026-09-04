#ifdef __cplusplus
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>

#include <Cmlx/mlx-array.h>
#include <Cmlx/mlx-stream.h>

namespace mlx::core::gpu {

void init();
void new_stream(Stream s);
void new_thread_unsafe_stream(Stream s);
void eval(array& arr);
void finalize(Stream s);
void synchronize(Stream s);
void clear_streams();

} // namespace mlx::core::gpu
#endif
