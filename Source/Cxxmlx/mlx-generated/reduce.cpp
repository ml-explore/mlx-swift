namespace mlx::core::metal {

const char* reduce() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/reduce.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/reduction/reduce_all.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/reduction/reduce_all.h"
// Copyright © 2023-2024 Apple Inc.

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS>
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
  IdxT start_idx = gid.y * IdxT(row_size);
  IdxT actual_row =
      (start_idx + row_size <= in_size) ? row_size : in_size - start_idx;
  IdxT blocks = actual_row / (lsize.x * N_READS);
  int extra = actual_row - blocks * (lsize.x * N_READS);
  extra -= lid.x * N_READS;
  start_idx += lid.x * N_READS;
  in += start_idx;

  if (extra >= N_READS) {
    blocks++;
    extra = 0;
  }

  for (IdxT b = 0; b < blocks; b++) {
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

  // Reduction within simd group
  total = op.simd_reduce(total);
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      shared_vals[simd_group_id] = total;
    }

    // Reduction within thread group
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = lid.x < simd_per_group ? shared_vals[lid.x] : op.init;
    total = op.simd_reduce(total);
  }

  if (lid.x == 0) {
    out[gid.y] = total;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/reduction/reduce_col.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/reduction/reduce_col.h"
// Copyright © 2023-2024 Apple Inc.

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void col_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  constexpr int n_reads = 4;
  Op op;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  U totals[n_reads];
  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  IdxT column = IdxT(gid.x) * lsize.x * n_reads + lid.x * n_reads;
  if (column >= reduction_stride) {
    return;
  }
  bool safe = column + n_reads <= reduction_stride;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total_rows = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(lid.y, reduce_shape, reduce_strides);
  for (IdxT r = lid.y; r < total_rows; r += lsize.y) {
    row = in + loop.location();
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
    loop.next(lsize.y, reduce_shape, reduce_strides);
  }

  if (lsize.y > 1) {
    // lsize.y should be <= 8
    threadgroup U shared_vals[32 * 8 * n_reads];
    for (int i = 0; i < n_reads; i++) {
      shared_vals[lid.y * lsize.x * n_reads + lid.x * n_reads + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.y == 0) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = shared_vals[lid.x * n_reads + i];
      }
      for (uint j = 1; j < lsize.y; j++) {
        for (int i = 0; i < n_reads; i++) {
          totals[i] =
              op(shared_vals[j * lsize.x * n_reads + lid.x * n_reads + i],
                 totals[i]);
        }
      }
    }
  }

  if (lid.y == 0) {
    out += out_idx * IdxT(reduction_stride) + column;
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

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void col_reduce_longcolumn(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant size_t& out_size [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  Op op;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  IdxT out_idx = gid.x + gsize.x * IdxT(gid.y);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + lid.x;

  U total = Op::init;
  IdxT total_rows = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(gid.z * lsize.y + lid.y, reduce_shape, reduce_strides);
  for (IdxT r = gid.z * lsize.y + lid.y; r < total_rows;
       r += lsize.y * gsize.z) {
    row = in + loop.location();
    total = op(static_cast<U>(*row), total);
    loop.next(lsize.y * gsize.z, reduce_shape, reduce_strides);
  }

  threadgroup U shared_vals[32 * 32];
  shared_vals[lid.y * lsize.x + lid.x] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0) {
    for (uint i = 1; i < lsize.y; i++) {
      total = op(total, shared_vals[i * lsize.x + lid.x]);
    }
    out[gid.z * IdxT(out_size) + out_idx * IdxT(reduction_stride) + lid.x] =
        total;
  }
}

/**
 * Our approach is the following simple looped approach:
 *  1. Each thread keeps running totals for BN / n_simdgroups outputs.
 *  2. Load a tile BM, BN in registers and accumulate in the running totals
 *  3. Move ahead by BM steps until the column axis and the non column
 *     reductions are exhausted.
 *  6. If BM == 32 then transpose in SM and simd reduce the running totals.
 *     Otherwise write in shared memory and BN threads accumulate the running
 *     totals with a loop.
 *  7. Write them to the output
 */
template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int BM,
    int BN>
[[kernel]] void col_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 8;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  IdxT column = BN * gid.x + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(offset.y, reduce_shape, reduce_strides);
  for (IdxT r = offset.y; r < total; r += BM) {
    row = in + loop.location();

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

  // We can use a simd reduction to accumulate across BM so each thread writes
  // the partial output to SM and then each simdgroup does BN / n_simdgroups
  // accumulations.
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

    // Write the output.
    if (simd_lane_id == 0) {
      IdxT out_column = BN * gid.x + out_offset.x;
      out += out_idx * IdxT(reduction_stride) + out_column;
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

  // Each thread holds n_reads partial results. We write them all out to shared
  // memory and threads with offset.y == 0 aggregate the columns and write the
  // outputs.
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

    // Write the output.
    if (offset.y == 0) {
      out += out_idx * IdxT(reduction_stride) + column;
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

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int BM,
    int BN>
[[kernel]] void col_reduce_2pass(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant size_t& out_size [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 8;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;
  constexpr int n_outputs = BN / n_simdgroups;
  constexpr short outer_blocks = 32;
  static_assert(BM == 32, "BM should be equal to 32");

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  IdxT column = BN * gid.x + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  IdxT full_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT block_idx = full_idx / IdxT(out_size);
  IdxT out_idx = full_idx % IdxT(out_size);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(offset.y + block_idx * BM, reduce_shape, reduce_strides);
  for (IdxT r = offset.y + block_idx * BM; r < total; r += outer_blocks * BM) {
    row = in + loop.location();

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

    loop.next(outer_blocks * BM, reduce_shape, reduce_strides);
  }

  // We can use a simd reduction to accumulate across BM so each thread writes
  // the partial output to SM and then each simdgroup does BN / n_simdgroups
  // accumulations.
  for (int i = 0; i < n_reads; i++) {
    shared_vals[offset.y * BN + offset.x + i] = totals[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  short2 out_offset(simd_group_id * n_outputs, simd_lane_id);
  for (int i = 0; i < n_outputs; i++) {
    totals[i] =
        op.simd_reduce(shared_vals[out_offset.y * BN + out_offset.x + i]);
  }

  // Write the output.
  if (simd_lane_id == 0) {
    IdxT out_column = BN * gid.x + out_offset.x;
    out += full_idx * IdxT(reduction_stride) + out_column;
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

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/reduction/reduce_init.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/reduction/reduce_init.h"
// Copyright © 2023-2024 Apple Inc.

template <typename T, typename Op>
[[kernel]] void init_reduce(
    device T* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::init;
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/reduction/reduce_row.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/reduction/reduce_row.h"
// Copyright © 2023-2024 Apple Inc.

// Row reduction utilities
// - `per_thread_row_reduce` collaborative partial reduction in the threadgroup
// - `threadgroup_reduce` collaborative reduction in the threadgroup such that
//   lid.x == 0 holds the reduced value
// - `thread_reduce` simple loop and reduce the row

/**
 * The thread group collaboratively reduces across the rows with bounds
 * checking. In the end each thread holds a part of the reduction.
 */
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

  // Set up the accumulator registers
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = Op::init;
  }

  // Loop over the reduction size within thread group
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }

      inputs[j] += lsize_x * N_READS;
    }
  }

  // Separate case for the last set as we close the reduction size
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

/**
 * Consecutive rows in a contiguous array.
 */
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
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  inputs[0] = in + lid_x * N_READS;
  for (int i = 1; i < N_READS; i++) {
    inputs[i] = inputs[i - 1] + reduction_size;
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

/**
 * Consecutive rows in an arbitrarily ordered array.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const int64_t row_idx,
    int blocks,
    int extra,
    const constant int* shape,
    const constant int64_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x) {
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  in += lid_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    inputs[i] = in + elem_to_loc(row_idx + i, shape, strides, ndim);
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

/**
 * Reduce within the threadgroup.
 */
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

  // Simdgroup first
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = op.simd_reduce(totals[i]);
  }

  // Across simdgroups
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

// Reduction kernels
// - `row_reduce_small` depending on the non-row reductions and row size it
//   either just loops over everything or a simd collaboratively reduces the
//   non_row reductions. In the first case one thread is responsible for one
//   output on the 2nd one simd is responsible for one output.
// - `row_reduce_simple` simple contiguous row reduction
// - `row_reduce_looped` simply loop and reduce each row for each non-row
//   reduction. One threadgroup is responsible for one output.

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& row_size [[buffer(2)]],
    const constant int64_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tsize [[threads_per_grid]]) {
  Op op;

  U total_val = Op::init;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);

  // Precompute some row reduction numbers
  const device T* row;
  int blocks = IdxT(row_size) / N_READS;
  int extra = IdxT(row_size) % N_READS;

  if ((non_row_reductions < 32 && row_size <= 8) || non_row_reductions <= 8) {
    // Simple loop over non_row_reductions and reduce the row in the thread.
    IdxT out_idx = tid.x + tsize.x * IdxT(tid.y);
    in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

    for (uint r = 0; r < non_row_reductions; r++) {
      row = in + loop.location();
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(reduce_shape, reduce_strides);
    }

    out[out_idx] = total_val;
  } else {
    // Collaboratively reduce over non_row_reductions in the simdgroup. Each
    // thread reduces every 32nd row and then a simple simd reduce.
    IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
    in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

    loop.next(simd_lane_id, reduce_shape, reduce_strides);

    for (uint r = simd_lane_id; r < non_row_reductions; r += simd_size) {
      row = in + loop.location();
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
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
[[kernel]] void row_reduce_simple(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& out_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup U shared_vals[simd_size * N_WRITES];
  U totals[N_WRITES];

  // Move to the row
  IdxT out_idx = N_WRITES * (gid.y + gsize.y * IdxT(gid.z));
  if (out_idx + N_WRITES > out_size) {
    out_idx = out_size - N_WRITES;
  }
  in += out_idx * IdxT(reduction_size);
  out += out_idx;

  // Each thread reduces across the row
  int blocks = IdxT(reduction_size) / (lsize.x * N_READS);
  int extra = reduction_size - blocks * (lsize.x * N_READS);
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, in, reduction_size, blocks, extra, lsize.x, lid.x);

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
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
    typename IdxT,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& row_size [[buffer(2)]],
    const constant int64_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
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

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);

  // lid.x * N_READS breaks the per_thread_row_reduce interface a bit. Maybe it
  // needs a small refactor.
  in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim) + lid.x * N_READS;

  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;
  int blocks = IdxT(row_size) / (lsize.x * N_READS);
  int extra = row_size - blocks * (lsize.x * N_READS);

  for (IdxT i = 0; i < non_row_reductions; i++) {
    row = in + loop.location();

    // Each thread reduces across the row
    U row_total;
    per_thread_row_reduce<T, U, Op, N_READS, 1>(
        &row_total, &row, blocks, extra, lsize.x, lid.x);

    // Aggregate across rows
    total = op(total, row_total);

    loop.next(reduce_shape, reduce_strides);
  }

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, 1>(
      &total, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
  if (lid.x == 0) {
    out[out_idx] = total;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/reduce.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/reduce.h"

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
