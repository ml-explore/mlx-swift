namespace mlx::core::metal {

const char* gather_front() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/indexing/gather_front.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/indexing/indexing.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/indexing/indexing.h"
// Copyright © 2023-2024 Apple Inc.


#include <metal_stdlib>

template <typename IdxT, int NIDX>
struct Indices {
  const array<const device IdxT*, NIDX> buffers;
  const constant int* shapes;
  const constant int64_t* strides;
  const constant bool* row_contiguous;
  const int ndim;
};

template <typename IdxT>
METAL_FUNC size_t offset_neg_idx(IdxT idx, int size) {
  if (is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return (idx < 0) ? idx + size : idx;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/indexing/gather_front.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/indexing/gather_front.h"
// Copyright © 2025 Apple Inc.



template <typename T, typename IdxT, typename LocT, int N>
[[kernel]] void gather_front(
    const device T* src,
    const device IdxT* indices,
    device T* out,
    const constant int64_t& stride,
    const constant int& size,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto idx = offset_neg_idx(indices[index.y], size);
  LocT src_idx = static_cast<LocT>(stride) * idx;
  LocT out_idx = static_cast<LocT>(stride) * index.y;

  int s_idx = N * index.x;
  for (int i = 0; i < N && s_idx < stride; ++i, ++s_idx) {
    out[out_idx + s_idx] = src[src_idx + s_idx];
  }
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
