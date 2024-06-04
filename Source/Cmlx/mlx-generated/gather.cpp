namespace mlx::core::metal {

const char* gather() {
  return R"preamble(
template <typename IdxT, int NIDX>
struct Indices {
  const array<const device IdxT*, NIDX> buffers;
  const constant int* shapes;
  const constant size_t* strides;
  const int ndim;
};
template <typename IdxT>
METAL_FUNC size_t offset_neg_idx(IdxT idx, size_t size) {
  if (is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return (idx < 0) ? idx + size : idx;
  }
}

template <typename T, typename IdxT, int NIDX, int IDX_NDIM>
METAL_FUNC void gather_impl(
    const device T* src [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant int* src_shape [[buffer(2)]],
    const constant size_t* src_strides [[buffer(3)]],
    const constant size_t& src_ndim [[buffer(4)]],
    const constant int* slice_sizes [[buffer(5)]],
    const constant int* axes [[buffer(6)]],
    const thread Indices<IdxT, NIDX>& indices,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto ind_idx = index.x;
  auto ind_offset = index.y;
  size_t src_idx = 0;
  for (int i = 0; i < NIDX; ++i) {
    size_t idx_loc;
    if (IDX_NDIM == 0) {
      idx_loc = 0;
    } else if (IDX_NDIM == 1) {
      idx_loc = ind_idx * indices.strides[indices.ndim * i];
    } else {
      idx_loc = elem_to_loc(
          ind_idx,
          &indices.shapes[indices.ndim * i],
          &indices.strides[indices.ndim * i],
          indices.ndim);
    }
    auto ax = axes[i];
    auto idx_val = offset_neg_idx(indices.buffers[i][idx_loc], src_shape[ax]);
    src_idx += idx_val * src_strides[ax];
  }
  auto src_offset = elem_to_loc(ind_offset, slice_sizes, src_strides, src_ndim);
  size_t out_idx = index.y + static_cast<size_t>(grid_dim.y) * index.x;
  out[out_idx] = src[src_offset + src_idx];
}
)preamble";
}

} // namespace mlx::core::metal
