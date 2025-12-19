namespace mlx::core::metal {

const char* ternary_ops() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/ternary_ops.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/ternary_ops.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/ternary_ops.h"
// Copyright © 2023-2024 Apple Inc.


struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
