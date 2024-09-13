namespace mlx::core::metal {

const char* reduce() {
  return R"preamble(

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& in_size [[buffer(2)]],
    const constant size_t& row_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];
  U total = Op::init;
  int64_t start_idx = gid.y * row_size;
  int64_t actual_row =
      (start_idx + row_size <= in_size) ? row_size : in_size - start_idx;
  int64_t blocks = actual_row / (lsize.x * N_READS);
  int extra = actual_row - blocks * (lsize.x * N_READS);
  extra -= lid.x * N_READS;
  start_idx += lid.x * N_READS;
  in += start_idx;
  if (extra >= N_READS) {
    blocks++;
    extra = 0;
  }
  for (int64_t b = 0; b < blocks; b++) {
    for (int i = 0; i < N_READS; i++) {
      total = op(static_cast<U>(in[i]), total);
    }
    in += lsize.x * N_READS;
  }
  if (extra > 0) {
    for (int i = 0; i < extra; i++) {
      total = op(static_cast<U>(in[i]), total);
    }
  }
  total = op.simd_reduce(total);
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      shared_vals[simd_group_id] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = lid.x < simd_per_group ? shared_vals[lid.x] : op.init;
    total = op.simd_reduce(total);
  }
  if (lid.x == 0) {
    out[gid.y] = total;
  }
}
template <
    typename T,
    typename U,
    typename Op,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tsize [[threads_per_grid]]) {
  Op op;
  looped_elem_to_loc<NDIMS> loop;
  const device T* row;
  if (reduction_size * non_col_reductions < 64 && reduction_stride < 32) {
    U totals[31];
    for (int i = 0; i < 31; i++) {
      totals[i] = Op::init;
    }
    short stride = reduction_stride;
    short size = reduction_size;
    short blocks = stride / N_READS;
    short extra = stride - blocks * N_READS;
    size_t out_idx = tid.x + tsize.y * size_t(tid.y);
    in += elem_to_loc(out_idx, shape, strides, ndim);
    for (uint r = 0; r < non_col_reductions; r++) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);
      for (short i = 0; i < size; i++) {
        for (short j = 0; j < blocks; j++) {
          for (short k = 0; k < N_READS; k++) {
            totals[j * N_READS + k] =
                op(totals[j * N_READS + k],
                   static_cast<U>(row[i * stride + j * N_READS + k]));
          }
        }
        for (short k = 0; k < extra; k++) {
          totals[blocks * N_READS + k] =
              op(totals[blocks * N_READS + k],
                 static_cast<U>(row[i * stride + blocks * N_READS + k]));
        }
      }
      loop.next(reduce_shape, reduce_strides);
    }
    out += out_idx * reduction_stride;
    for (short j = 0; j < stride; j++) {
      out[j] = totals[j];
    }
  }
  else if (reduction_size * non_col_reductions < 32) {
    U totals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      totals[i] = Op::init;
    }
    short size = reduction_size;
    size_t offset = size_t(tid.x) * N_READS;
    bool safe = offset + N_READS <= reduction_stride;
    short extra = reduction_stride - offset;
    size_t out_idx = tid.y + tsize.z * size_t(tid.z);
    in += elem_to_loc(out_idx, shape, strides, ndim) + offset;
    for (uint r = 0; r < non_col_reductions; r++) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);
      if (safe) {
        for (short i = 0; i < size; i++) {
          for (short j = 0; j < N_READS; j++) {
            totals[j] =
                op(static_cast<U>(row[i * reduction_stride + j]), totals[j]);
          }
        }
      } else {
        for (short i = 0; i < size; i++) {
          for (short j = 0; j < extra; j++) {
            totals[j] =
                op(static_cast<U>(row[i * reduction_stride + j]), totals[j]);
          }
        }
      }
      loop.next(reduce_shape, reduce_strides);
    }
    out += out_idx * reduction_stride + offset;
    if (safe) {
      for (short i = 0; i < N_READS; i++) {
        out[i] = totals[i];
      }
    } else {
      for (short i = 0; i < extra; i++) {
        out[i] = totals[i];
      }
    }
  }
  else {
    threadgroup U shared_vals[1024];
    U totals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      totals[i] = Op::init;
    }
    short stride = reduction_stride;
    short lid = simd_group_id * simd_size + simd_lane_id;
    short2 tile((stride + N_READS - 1) / N_READS, 32);
    short2 offset((lid % tile.x) * N_READS, lid / tile.x);
    short sm_stride = tile.x * N_READS;
    bool safe = offset.x + N_READS <= stride;
    size_t out_idx = gid.y + gsize.y * size_t(gid.z);
    in += elem_to_loc(out_idx, shape, strides, ndim) + offset.x;
    size_t total = non_col_reductions * reduction_size;
    loop.next(offset.y, reduce_shape, reduce_strides);
    for (size_t r = offset.y; r < total; r += simd_size) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);
      if (safe) {
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(static_cast<U>(row[i]), totals[i]);
        }
      } else {
        U vals[N_READS];
        for (int i = 0; i < N_READS; i++) {
          vals[i] = (offset.x + i < stride) ? static_cast<U>(row[i]) : op.init;
        }
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(vals[i], totals[i]);
        }
      }
      loop.next(simd_size, reduce_shape, reduce_strides);
    }
    for (int i = 0; i < N_READS; i++) {
      shared_vals[offset.y * sm_stride + offset.x + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      totals[i] = op.simd_reduce(
          shared_vals[simd_lane_id * sm_stride + simd_group_id * N_READS + i]);
    }
    if (simd_lane_id == 0) {
      short column = simd_group_id * N_READS;
      out += out_idx * reduction_stride + column;
      if (column + N_READS <= stride) {
        for (int i = 0; i < N_READS; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; column + i < stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }
}
template <typename T, typename U, typename Op, int NDIMS, int BM, int BN>
[[kernel]] void col_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 4;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;
  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  looped_elem_to_loc<NDIMS> loop;
  const device T* row;
  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }
  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  size_t column = BN * gid.x + offset.x;
  bool safe = column + n_reads <= reduction_stride;
  size_t out_idx = gid.y + gsize.y * size_t(gid.z);
  size_t in_idx = elem_to_loc(out_idx, shape, strides, ndim);
  in += in_idx + column;
  size_t total = non_col_reductions * reduction_size;
  loop.next(offset.y, reduce_shape, reduce_strides);
  for (size_t r = offset.y; r < total; r += BM) {
    row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);
    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(static_cast<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? static_cast<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }
    loop.next(BM, reduce_shape, reduce_strides);
  }
  if (BM == 32) {
    constexpr int n_outputs = BN / n_simdgroups;
    static_assert(
        BM != 32 || n_outputs == n_reads,
        "The tile should be selected such that n_outputs == n_reads");
    for (int i = 0; i < n_reads; i++) {
      shared_vals[offset.y * BN + offset.x + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    short2 out_offset(simd_group_id * n_outputs, simd_lane_id);
    for (int i = 0; i < n_outputs; i++) {
      totals[i] =
          op.simd_reduce(shared_vals[out_offset.y * BN + out_offset.x + i]);
    }
    if (simd_lane_id == 0) {
      size_t out_column = BN * gid.x + out_offset.x;
      out += out_idx * reduction_stride + out_column;
      if (out_column + n_outputs <= reduction_stride) {
        for (int i = 0; i < n_outputs; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; out_column + i < reduction_stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }
  else {
    short x_block = offset.x / n_reads;
    for (int i = 0; i < n_reads; i++) {
      shared_vals[x_block * BM * n_reads + i * BM + offset.y] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (offset.y == 0) {
      for (int i = 0; i < n_reads; i++) {
        for (int j = 1; j < BM; j++) {
          totals[i] =
              op(shared_vals[x_block * BM * n_reads + i * BM + j], totals[i]);
        }
      }
    }
    if (offset.y == 0) {
      out += out_idx * reduction_stride + column;
      if (safe) {
        for (int i = 0; i < n_reads; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; column + i < reduction_stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }
}
template <typename T, typename Op>
[[kernel]] void init_reduce(
    device T* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::init;
}
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* inputs[N_WRITES],
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  Op op;
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = Op::init;
  }
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
      inputs[j] += lsize_x * N_READS;
    }
  }
  int index = lid_x * N_READS;
  if (index + N_READS <= extra) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
    }
  } else {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; index + i < extra; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
    }
  }
}
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const constant size_t& reduction_size,
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  const device T* inputs[N_WRITES];
  inputs[0] = in + lid_x * N_READS;
  for (int i = 1; i < N_READS; i++) {
    inputs[i] = inputs[i - 1] + reduction_size;
  }
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const size_t row_idx,
    int blocks,
    int extra,
    const constant int* shape,
    const constant size_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x) {
  const device T* inputs[N_WRITES];
  in += lid_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    inputs[i] = in + elem_to_loc(row_idx + i, shape, strides, ndim);
  }
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void threadgroup_reduce(
    thread U totals[N_WRITES],
    threadgroup U* shared_vals,
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = op.simd_reduce(totals[i]);
  }
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      for (int i = 0; i < N_WRITES; i++) {
        shared_vals[simd_group_id * N_WRITES + i] = totals[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U values[N_WRITES];
    for (int i = 0; i < N_WRITES; i++) {
      values[i] = (lid.x < simd_per_group) ? shared_vals[lid.x * N_WRITES + i]
                                           : op.init;
    }
    for (int i = 0; i < N_WRITES; i++) {
      totals[i] = op.simd_reduce(values[i]);
    }
  }
}
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC void
thread_reduce(thread U& total, const device T* row, int blocks, int extra) {
  Op op;
  for (int i = 0; i < blocks; i++) {
    U vals[N_READS];
    for (int j = 0; j < N_READS; j++) {
      vals[j] = row[j];
    }
    for (int j = 0; j < N_READS; j++) {
      total = op(vals[j], total);
    }
    row += N_READS;
  }
  for (int i = 0; i < extra; i++) {
    total = op(*row++, total);
  }
}
template <
    typename T,
    typename U,
    typename Op,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& row_size [[buffer(2)]],
    const constant size_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tsize [[threads_per_grid]]) {
  Op op;
  U total_val = Op::init;
  looped_elem_to_loc<NDIMS> loop;
  const device T* row;
  int blocks = row_size / N_READS;
  int extra = row_size % N_READS;
  if ((non_row_reductions < 32 && row_size <= 8) || non_row_reductions <= 8) {
    size_t out_idx = tid.x + tsize.y * size_t(tid.y);
    in += elem_to_loc(out_idx, shape, strides, ndim);
    for (uint r = 0; r < non_row_reductions; r++) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(reduce_shape, reduce_strides);
    }
    out[out_idx] = total_val;
  } else {
    size_t out_idx = gid.y + gsize.y * size_t(gid.z);
    in += elem_to_loc(out_idx, shape, strides, ndim);
    loop.next(simd_lane_id, reduce_shape, reduce_strides);
    for (uint r = simd_lane_id; r < non_row_reductions; r += simd_size) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(simd_size, reduce_shape, reduce_strides);
    }
    total_val = op.simd_reduce(total_val);
    if (simd_lane_id == 0) {
      out[out_idx] = total_val;
    }
  }
}
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
[[kernel]] void row_reduce_simple(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup U shared_vals[simd_size * N_WRITES];
  U totals[N_WRITES];
  size_t out_idx = N_WRITES * (gid.y + gsize.y * size_t(gid.z));
  if (out_idx + N_WRITES > out_size) {
    out_idx = out_size - N_WRITES;
  }
  in += out_idx * reduction_size;
  out += out_idx;
  int blocks = reduction_size / (lsize.x * N_READS);
  int extra = reduction_size - blocks * (lsize.x * N_READS);
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, in, reduction_size, blocks, extra, lsize.x, lid.x);
  threadgroup_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);
  if (lid.x == 0) {
    for (int i = 0; i < N_WRITES; i++) {
      out[i] = totals[i];
    }
  }
}
template <
    typename T,
    typename U,
    typename Op,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& row_size [[buffer(2)]],
    const constant size_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];
  U total = Op::init;
  size_t out_idx = gid.y + gsize.y * size_t(gid.z);
  in += elem_to_loc(out_idx, shape, strides, ndim) + lid.x * N_READS;
  looped_elem_to_loc<NDIMS> loop;
  const device T* row;
  int blocks = row_size / (lsize.x * N_READS);
  int extra = row_size - blocks * (lsize.x * N_READS);
  for (size_t i = 0; i < non_row_reductions; i++) {
    row = in + loop.location(i, reduce_shape, reduce_strides, reduce_ndim);
    U row_total;
    per_thread_row_reduce<T, U, Op, N_READS, 1>(
        &row_total, &row, blocks, extra, lsize.x, lid.x);
    total = op(total, row_total);
    loop.next(reduce_shape, reduce_strides);
  }
  threadgroup_reduce<T, U, Op, N_READS, 1>(
      &total, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);
  if (lid.x == 0) {
    out[out_idx] = total;
  }
}
)preamble";
}

} // namespace mlx::core::metal
