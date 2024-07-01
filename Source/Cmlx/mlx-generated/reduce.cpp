namespace mlx::core::metal {

const char* reduce() {
  return R"preamble(

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U per_thread_all_reduce(
    const device T* in,
    const device size_t& in_size,
    uint gid,
    uint grid_size) {
  Op op;
  U total_val = Op::init;
  if (gid * N_READS < in_size) {
    in += gid * N_READS;
    int r = 0;
    for (; r < (int)ceildiv(in_size, grid_size * N_READS) - 1; r++) {
      U vals[N_READS] = {op.init};
      for (int i = 0; i < N_READS; i++) {
        vals[i] = static_cast<U>(in[i]);
      }
      for (int i = 0; i < N_READS; i++) {
        total_val = op(vals[i], total_val);
      }
      in += grid_size * N_READS;
    }
    size_t curr_idx = (gid + r * (size_t)grid_size) * N_READS;
    if (curr_idx < in_size) {
      int max_reads = in_size - curr_idx;
      T vals[N_READS];
      for (int i = 0, idx = 0; i < N_READS; i++, idx++) {
        idx = idx < max_reads ? idx : max_reads - 1;
        vals[i] = in[idx];
      }
      for (int i = 0; i < N_READS; i++) {
        U val = i < max_reads ? vals[i] : Op::init;
        total_val = op(static_cast<U>(val), total_val);
      }
    }
  }
  return total_val;
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce(
    const device T* in [[buffer(0)]],
    device mlx_atomic<U>* out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U local_vals[simd_size];
  U total_val =
      per_thread_all_reduce<T, U, Op, N_READS>(in, in_size, gid, grid_size);
  total_val = op.simd_reduce(total_val);
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total_val = lid < simd_per_group ? local_vals[lid] : op.init;
  total_val = op.simd_reduce(total_val);
  if (lid == 0) {
    op.atomic_update(out, total_val);
  }
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint thread_group_id [[threadgroup_position_in_grid]]) {
  Op op;
  threadgroup U local_vals[simd_size];
  U total_val =
      per_thread_all_reduce<T, U, Op, N_READS>(in, in_size, gid, grid_size);
  for (uint16_t lane_offset = simd_size / 2; lane_offset > 0;
       lane_offset /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, lane_offset));
  }
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total_val = lid < simd_per_group ? local_vals[lid] : op.init;
  for (uint16_t lane_offset = simd_size / 2; lane_offset > 0;
       lane_offset /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, lane_offset));
  }
  if (lid == 0) {
    out[thread_group_id] = total_val;
  }
}
template <typename T, typename U, typename Op>
[[kernel]] void col_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    const constant size_t& non_col_reductions [[buffer(8)]],
    const constant int* non_col_shapes [[buffer(9)]],
    const constant size_t* non_col_strides [[buffer(10)]],
    const constant int& non_col_ndim [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  (void)out_size;
  Op op;
  U total_val = Op::init;
  auto out_idx = tid;
  in += elem_to_loc(
      out_idx,
      shape + non_col_ndim,
      strides + non_col_ndim,
      ndim - non_col_ndim);
  for (uint i = 0; i < non_col_reductions; i++) {
    size_t in_idx =
        elem_to_loc(i, non_col_shapes, non_col_strides, non_col_ndim);
    for (uint j = 0; j < reduction_size; j++, in_idx += reduction_stride) {
      U val = static_cast<U>(in[in_idx]);
      total_val = op(total_val, val);
    }
  }
  out[out_idx] = total_val;
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U _contiguous_strided_reduce(
    const device T* in,
    threadgroup U* local_data,
    uint in_idx,
    uint reduction_size,
    uint reduction_stride,
    uint2 tid,
    uint2 lid,
    uint2 lsize) {
  Op op;
  U total_val = Op::init;
  uint base_offset = (tid.y * lsize.y + lid.y) * N_READS;
  for (uint r = 0; r < N_READS && (base_offset + r) < reduction_size; r++) {
    uint offset = base_offset + r;
    total_val =
        op(static_cast<U>(total_val), in[in_idx + offset * reduction_stride]);
  }
  local_data[lsize.y * lid.x + lid.y] = total_val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U val = Op::init;
  if (lid.y == 0) {
    for (uint i = 0; i < lsize.y; i++) {
      val = op(val, local_data[lsize.y * lid.x + i]);
    }
  }
  return val;
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_general(
    const device T* in [[buffer(0)]],
    device mlx_atomic<U>* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    threadgroup U* local_data [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  auto out_idx = tid.x * lsize.x + lid.x;
  auto in_idx = elem_to_loc(out_idx + tid.z * out_size, shape, strides, ndim);
  Op op;
  if (out_idx < out_size) {
    U val = _contiguous_strided_reduce<T, U, Op, N_READS>(
        in,
        local_data,
        in_idx,
        reduction_size,
        reduction_stride,
        tid.xy,
        lid.xy,
        lsize.xy);
    if (lid.y == 0) {
      op.atomic_update(out, val, out_idx);
    }
  }
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_general_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    threadgroup U* local_data [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]]) {
  auto out_idx = tid.x * lsize.x + lid.x;
  auto in_idx = elem_to_loc(out_idx + tid.z * out_size, shape, strides, ndim);
  if (out_idx < out_size) {
    U val = _contiguous_strided_reduce<T, U, Op, N_READS>(
        in,
        local_data,
        in_idx,
        reduction_size,
        reduction_stride,
        tid.xy,
        lid.xy,
        lsize.xy);
    if (lid.y == 0) {
      uint tgsize_y = ceildiv(gsize.y, lsize.y);
      uint tgsize_z = ceildiv(gsize.z, lsize.z);
      out[tgsize_y * tgsize_z * gid.x + tgsize_y * tid.z + tid.y] = val;
    }
  }
}
template <typename T, typename U, typename Op>
[[kernel]] void row_reduce_general_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint lid [[thread_position_in_grid]]) {
  Op op;
  uint out_idx = lid;
  if (out_idx >= out_size) {
    return;
  }
  U total_val = Op::init;
  for (short r = 0; r < short(non_row_reductions); r++) {
    uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
    const device T* in_row = in + in_idx;
    for (short i = 0; i < short(reduction_size); i++) {
      total_val = op(static_cast<U>(in_row[i]), total_val);
    }
  }
  out[out_idx] = total_val;
}
template <typename T, typename U, typename Op>
[[kernel]] void row_reduce_general_med(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[dispatch_simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  uint out_idx = simd_per_group * tid + simd_group_id;
  if (out_idx >= out_size) {
    return;
  }
  U total_val = Op::init;
  if (short(non_row_reductions) == 1) {
    uint in_idx = elem_to_loc(out_idx, shape, strides, ndim);
    const device T* in_row = in + in_idx;
    for (short i = simd_lane_id; i < short(reduction_size); i += 32) {
      total_val = op(static_cast<U>(in_row[i]), total_val);
    }
  }
  else if (short(non_row_reductions) >= 32) {
    for (short r = simd_lane_id; r < short(non_row_reductions); r += 32) {
      uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
      const device T* in_row = in + in_idx;
      for (short i = 0; i < short(reduction_size); i++) {
        total_val = op(static_cast<U>(in_row[i]), total_val);
      }
    }
  }
  else {
    const short n_reductions =
        short(reduction_size) * short(non_row_reductions);
    const short reductions_per_thread =
        (n_reductions + simd_size - 1) / simd_size;
    const short r_st = simd_lane_id / reductions_per_thread;
    const short r_ed = short(non_row_reductions);
    const short r_jump = simd_size / reductions_per_thread;
    const short i_st = simd_lane_id % reductions_per_thread;
    const short i_ed = short(reduction_size);
    const short i_jump = reductions_per_thread;
    if (r_st < r_jump) {
      for (short r = r_st; r < r_ed; r += r_jump) {
        uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
        const device T* in_row = in + in_idx;
        for (short i = i_st; i < i_ed; i += i_jump) {
          total_val = op(static_cast<U>(in_row[i]), total_val);
        }
      }
    }
  }
  total_val = op.simd_reduce(total_val);
  if (simd_lane_id == 0) {
    out[out_idx] = total_val;
  }
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U per_thread_row_reduce(
    const device T* in,
    const constant size_t& reduction_size,
    const constant size_t& out_size,
    const constant int* shape,
    const constant size_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x,
    uint2 tid) {
  Op op;
  int idx = tid.y * out_size + tid.x;
  int extra_offset = elem_to_loc(idx, shape, strides, ndim);
  in += extra_offset + lid_x * N_READS;
  U total_val = Op::init;
  int r = 0;
  for (; r < (int)ceildiv(reduction_size, N_READS * lsize_x) - 1; r++) {
    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      vals[i] = in[i];
    }
    for (int i = 0; i < N_READS; i++) {
      total_val = op(static_cast<U>(vals[i]), total_val);
    }
    in += lsize_x * N_READS;
  }
  size_t reduction_index = (lid_x + (size_t)lsize_x * r) * N_READS;
  if (reduction_index < reduction_size) {
    int max_reads = reduction_size - reduction_index;
    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      int idx = min(i, max_reads - 1);
      vals[i] = static_cast<U>(in[idx]);
    }
    for (int i = 0; i < N_READS; i++) {
      T val = i < max_reads ? vals[i] : Op::init;
      total_val = op(static_cast<U>(val), total_val);
    }
  }
  return total_val;
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_general(
    const device T* in [[buffer(0)]],
    device mlx_atomic<U>* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  (void)non_row_reductions;
  Op op;
  threadgroup U local_vals[simd_size];
  U total_val = per_thread_row_reduce<T, U, Op, N_READS>(
      in,
      reduction_size,
      out_size,
      shape,
      strides,
      ndim,
      lsize.x,
      lid.x,
      tid.xy);
  total_val = op.simd_reduce(total_val);
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (reduction_size > simd_size) {
    total_val = lid.x < simd_per_group ? local_vals[lid.x] : op.init;
    total_val = op.simd_reduce(total_val);
  }
  if (lid.x == 0) {
    op.atomic_update(out, total_val, tid.x);
  }
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_general_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  (void)non_row_reductions;
  Op op;
  threadgroup U local_vals[simd_size];
  U total_val = per_thread_row_reduce<T, U, Op, N_READS>(
      in,
      reduction_size,
      out_size,
      shape,
      strides,
      ndim,
      lsize.x,
      lid.x,
      tid.xy);
  for (uint16_t i = simd_size / 2; i > 0; i /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, i));
  }
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ceildiv(reduction_size, N_READS) > simd_size) {
    total_val = lid.x < simd_per_group ? local_vals[lid.x] : op.init;
    for (uint16_t i = simd_size / 2; i > 0; i /= 2) {
      total_val = op(total_val, simd_shuffle_down(total_val, i));
    }
  }
  if (lid.x == 0) {
    out[(ceildiv(gsize.y, lsize.y) * tid.x) + tid.y] = total_val;
  }
}
)preamble";
}

} // namespace mlx::core::metal
