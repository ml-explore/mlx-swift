namespace mlx::core::metal {

const char* arange() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/arange.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/arange.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/arange.h"
// Copyright © 2023-2024 Apple Inc.
template <typename T>
[[kernel]] void arange(
    constant const T& start,
    constant const T& step,
    device T* out,
    uint index [[thread_position_in_grid]]) {
  out[index] = start + index * step;
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
