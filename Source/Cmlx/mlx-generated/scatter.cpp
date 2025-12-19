namespace mlx::core::metal {

const char* scatter() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/indexing/scatter.h

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
// Contents from "mlx/backend/metal/kernels/indexing/scatter.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/indexing/scatter.h"
// Copyright © 2024 Apple Inc.



template <
    typename T,
    typename IdxT,
    typename Op,
    int NIDX,
    bool UPD_ROW_CONTIG,
    int NWORK,
    typename LocT>
METAL_FUNC void scatter_impl(
    const device T* updates,
    device mlx_atomic<T>* out,
    const constant int* upd_shape,
    const constant int64_t* upd_strides,
    const constant size_t& upd_ndim,
    const constant size_t& upd_size,
    const constant int* out_shape,
    const constant int64_t* out_strides,
    const constant size_t& out_ndim,
    const constant int* axes,
    const constant size_t& idx_size,
    const thread Indices<IdxT, NIDX>& indices,
    uint2 gid [[thread_position_in_grid]]) {
  Op op;

  auto ind_idx = gid.y * NWORK;
  LocT out_offset = 0;
  if (upd_size > 1) {
    out_offset = elem_to_loc<LocT>(
        gid.x, upd_shape + indices.ndim, out_strides, out_ndim);
  }

  for (int j = 0; j < NWORK && ind_idx < idx_size; ++j, ind_idx++) {
    LocT out_idx = out_offset;
    for (int i = 0; i < NIDX; ++i) {
      auto idx_loc = indices.row_contiguous[i]
          ? ind_idx
          : elem_to_loc<LocT>(
                ind_idx,
                &indices.shapes[indices.ndim * i],
                &indices.strides[indices.ndim * i],
                indices.ndim);
      auto ax = axes[i];
      auto idx_val = offset_neg_idx(indices.buffers[i][idx_loc], out_shape[ax]);
      out_idx +=
          static_cast<LocT>(idx_val) * static_cast<LocT>(out_strides[ax]);
    }
    auto upd_idx = ind_idx * static_cast<LocT>(upd_size) + gid.x;
    if constexpr (!UPD_ROW_CONTIG) {
      upd_idx = elem_to_loc<LocT>(upd_idx, upd_shape, upd_strides, upd_ndim);
    }
    op.atomic_update(out, updates[upd_idx], out_idx);
  }
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
