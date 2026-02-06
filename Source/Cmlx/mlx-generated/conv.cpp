namespace mlx::core::metal {

const char* conv() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/steel/conv/conv.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/defines.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/defines.h"
// Copyright © 2024 Apple Inc.


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/utils.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/utils.h"
// Copyright © 2024 Apple Inc.


#include <metal_stdlib>

METAL_FUNC ulong2 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
  }
  return ulong2(loc_a, loc_b);
}

METAL_FUNC ulong3 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  ulong loc_c{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
    loc_c += pos_in_dim * c_strides[i];
  }
  return ulong3(loc_a, loc_b, loc_c);
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/conv/params.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/conv/params.h"
// Copyright © 2024 Apple Inc.


template <int NDIM>
struct MLXConvParams {
  const int N; // Batch size
  const int C; // In channels
  const int O; // Out channels
  const int iS[NDIM]; // Input spatial dim
  const int wS[NDIM]; // Weight spatial dim
  const int oS[NDIM]; // Output spatial dim
  const int str[NDIM]; // Kernel strides
  const int pad[NDIM]; // Input padding
  const int kdil[NDIM]; // Kernel dilation
  const int idil[NDIM]; // Input dilation
  const int64_t in_strides[NDIM + 2]; // In strides
  const int64_t wt_strides[NDIM + 2]; // Wt strides
  const int64_t out_strides[NDIM + 2]; // Out strides
  const int groups; // Input channel groups
  const bool flip;
};

namespace mlx {
namespace steel {

struct ImplicitGemmConv2DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_c;

  const int tiles_n;
  const int tiles_m;
  const int swizzle_log;
};

struct Conv2DGeneralJumpParams {
  const int f_wgt_jump_h;
  const int f_wgt_jump_w;

  const int f_out_jump_h;
  const int f_out_jump_w;

  const int adj_out_h;
  const int adj_out_w;
  const int adj_out_hw;
  const int adj_implicit_m;
};

struct Conv2DGeneralBaseInfo {
  int weight_base;
  int weight_size;
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/conv/loaders/loader_channel_l.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/conv/loaders/loader_channel_l.h"
// Copyright © 2024 Apple Inc.




///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderLargeFilter {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;

  short weight_h;
  short weight_w;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderLargeFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Adjust for flip
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2] + bj;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];
      int ih = read_ih[i] + weight_h * params->kdil[0];
      int iw = read_iw[i] + weight_w * params->kdil[1];

      // Read from input if in bounds
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    if (++weight_w < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
    }
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderSmallFilter {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  using mask_t = short;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;

  short weight_h;
  short weight_w;

  const device T* src[n_rows];

  mask_t mask_h[n_rows];
  mask_t mask_w[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderSmallFilter(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_h(0),
        weight_w(0) {
    int out_n_pixels = params->oS[0] * params->oS[1];

    int read_n[n_rows];
    int read_ih[n_rows];
    int read_iw[n_rows];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;

      // Adjust for flip
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2] + bj;
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      mask_h[i] = 0;
      mask_w[i] = 0;
    }

    for (short kh = 0; kh < params->wS[0]; kh++) {
      short flip_h = params->flip ? params->wS[0] - kh - 1 : kh;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int n = read_n[i];
        int ih = read_ih[i] + flip_h * params->kdil[0];

        bool in_bounds = n < params->N && ih >= 0 && ih < params->iS[0];

        mask_h[i] |= (in_bounds << kh);
      }
    }

    for (short kw = 0; kw < params->wS[1]; kw++) {
      short flip_w = params->flip ? params->wS[1] - kw - 1 : kw;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; ++i) {
        int iw = read_iw[i] + flip_w * params->kdil[1];

        bool in_bounds = iw >= 0 && iw < params->iS[1];

        mask_w[i] |= (in_bounds << kw);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    mask_t h_mask = mask_t(1) << weight_h;
    mask_t w_mask = mask_t(1) << weight_w;

    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Read from input if in bounds
      if ((mask_h[i] & h_mask) && (mask_w[i] & w_mask)) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    if (++weight_w < params->wS[1]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }

      return;
    }

    weight_w = 0;

    if (++weight_h < params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }

      return;
    }

    weight_h = 0;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; i++) {
      src[i] += gemm_params->inp_jump_c;
    }
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DWeightBlockLoader {
  // Destination dimensions
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size =
      (BN == 8) ? 1 : (tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4);

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Leading dimension for src
  const int src_ld;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<2>* params;

  int weight_hw;
  int weight_step;

  const int read_n;
  const bool do_read;

  /* Constructor */
  METAL_FUNC Conv2DWeightBlockLoader(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        weight_hw(0),
        weight_step(params->C / params->groups),
        read_n(offsets.y + bi),
        do_read(read_n + n_rows * TROWS <= gemm_params_->N) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BN; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((read_n + i) < params->O) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = src[i * src_ld + j];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    if (++weight_hw < (params->wS[1] * params->wS[0])) {
      src += weight_step;
      return;
    }

    weight_hw = 0;

    src += BK - (params->wS[1] * params->wS[0] - 1) * weight_step;
  }
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/conv/loaders/loader_channel_n.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/conv/loaders/loader_channel_n.h"
// Copyright © 2024 Apple Inc.




///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <short n_channels_>
struct ChannelHelper {
  STEEL_CONST short n_channels = n_channels_;
  STEEL_CONST short vec_size = n_channels_ <= 4 ? 4 : 8;
  STEEL_CONST short excess = vec_size - n_channels_;
};

template <>
struct ChannelHelper<1> {
  STEEL_CONST short n_channels = 1;
  STEEL_CONST short vec_size = 1;
  STEEL_CONST short excess = 0;
};

template <>
struct ChannelHelper<2> {
  STEEL_CONST short n_channels = 2;
  STEEL_CONST short vec_size = 2;
  STEEL_CONST short excess = 0;
};

template <>
struct ChannelHelper<3> {
  STEEL_CONST short n_channels = 3;
  STEEL_CONST short vec_size = 4;
  STEEL_CONST short excess = 1;
};

template <>
struct ChannelHelper<4> {
  STEEL_CONST short n_channels = 4;
  STEEL_CONST short vec_size = 4;
  STEEL_CONST short excess = 0;
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short n_channels,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderSmallChannels {
  // Destination dimensions
  STEEL_CONST short BROWS = BM;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = ChannelHelper<n_channels>::vec_size;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;

  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;

  int weight_hw;

  const device T* src[n_rows];

  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];

  /* Constructor */
  METAL_FUNC Conv2DInputBlockLoaderSmallChannels(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        gemm_params(gemm_params_),
        weight_hw(thread_idx % TCOLS) {
    int out_n_pixels = params->oS[0] * params->oS[1];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];

      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];

      // Read from input if in bounds
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2];

      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    if (weight_hw >= params->wS[1] * params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    int wh = (weight_hw / params->wS[1]);
    int ww = (weight_hw % params->wS[1]);

    int flip_h = params->flip ? params->wS[0] - wh - 1 : wh;
    int flip_w = params->flip ? params->wS[1] - ww - 1 : ww;

    int weight_h = flip_h * params->kdil[0];
    int weight_w = flip_w * params->kdil[1];

    STEEL_PRAGMA_UNROLL
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      // Find bounds
      int n = read_n[i];
      int ih = read_ih[i] + weight_h;
      int iw = read_iw[i] + weight_w;

      // Read from input if in bounds
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
        const device T* curr_src = src[i] + weight_h * params->in_strides[1] +
            weight_w * params->in_strides[2];

        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < n_channels; ++j) {
          dst[is * dst_ld + j] = curr_src[j];
        }

        STEEL_PRAGMA_UNROLL
        for (short j = n_channels; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }

      // Zero pad otherwise
      else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    weight_hw += TCOLS;
  }
};

template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short n_channels,
    short tgp_padding = 0>
struct Conv2DWeightBlockLoaderSmallChannels {
  // Destination dimensions
  STEEL_CONST short BROWS = BN;
  STEEL_CONST short BCOLS = BK;

  // Read dimensions
  STEEL_CONST short dst_ld = BCOLS + tgp_padding;
  STEEL_CONST short vec_size = ChannelHelper<n_channels>::vec_size;

  // Thread read shape
  STEEL_CONST short TCOLS = BCOLS / vec_size;
  STEEL_CONST short TROWS = tgp_size / TCOLS;

  // Rows / strided reads within the block
  STEEL_CONST short n_rows = BROWS / TROWS;

  // Leading dimension for src
  const int src_ld;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  const constant MLXConvParams<2>* params;

  int weight_hw;

  const int read_n;
  const bool do_read;

  /* Constructor */
  METAL_FUNC Conv2DWeightBlockLoaderSmallChannels(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant ImplicitGemmConv2DParams* gemm_params_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld),
        params(params_),
        weight_hw(thread_idx % TCOLS),
        read_n(offsets.y + bi),
        do_read(read_n + BN <= gemm_params_->N) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    if (bi >= BROWS || bj >= BCOLS)
      return;

    if (read_n >= params->O || weight_hw >= params->wS[1] * params->wS[0]) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }

      return;
    }

    const device T* curr_src = src + weight_hw * (params->C / params->groups);

    if (BN != 8 || do_read) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < n_channels; j++) {
          dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        }

        STEEL_PRAGMA_UNROLL
        for (short j = n_channels; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
    } else {
      for (short i = 0; i < BROWS; i += TROWS) {
        if (((read_n + i) < params->O)) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < n_channels; j++) {
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
          }

          STEEL_PRAGMA_UNROLL
          for (short j = n_channels; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    weight_hw += TCOLS;
  }
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/conv/loader.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/conv/loader.h"
// Copyright © 2024 Apple Inc.



///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/transforms.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/transforms.h"
// Copyright © 2024 Apple Inc.



///////////////////////////////////////////////////////////////////////////////
// Transforms and Epilogues
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT) {
    return static_cast<OutT>(x);
  }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(
        x * static_cast<InT>(alpha) + (static_cast<OutT>(beta) * c));
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

struct BlockSwizzle {
  static METAL_FUNC int2
  swizzle(uint3 tid [[threadgroup_position_in_grid]], const int swizzle_log) {
    const int tid_x = (tid.x) >> swizzle_log;
    const int tid_y =
        ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
  }
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/utils/type_traits.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/utils/type_traits.h"
// Copyright © 2024 Apple Inc.


#include <metal_stdlib>

#pragma METAL internals : enable

namespace metal {

template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T>
constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class T>
struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

#pragma METAL internals : disable

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
// Copyright © 2024 Apple Inc.


#include <metal_stdlib>

#pragma METAL internals : enable

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;

  METAL_FUNC constexpr operator value_type() const noexcept {
    return value;
  }

  // METAL_FUNC constexpr value_type operator()() const noexcept {
  //   return value;
  // }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_integral : bool_constant<metal::is_integral<T>::value> {};

template <class T, T v>
struct is_integral<integral_constant<T, v>>
    : bool_constant<metal::is_integral<T>::value> {};

template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <int val>
using Int = integral_constant<int, val>;

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(res), res>{};         \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(true_type, T) {
  return true_type{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(T, true_type) {
  return true_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(false_type, T) {
  return false_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(T, false_type) {
  return false_type{};
}

// Dispatch utilities
template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/mma.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/mma.h"
// Copyright © 2024 Apple Inc.


#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(
      kFragRows_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
  static_assert(
      kFragCols_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;

  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;

  METAL_FUNC static constexpr short2 get_coord(
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  load(thread frag_type& dst, SrcPtrType src, StrX str_x, StrY str_y) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x + j * str_y]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[(off_x + i) * str_x + (off_x + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  store(const thread frag_type& src, DstPtrType dst, StrX str_x, StrY str_y) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_slice(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < stop_x && (off_x + i) >= start_x &&
            (off_y + j) < stop_y && (off_y + j) >= start_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread frag_type& A,
      thread frag_type& B,
      thread frag_type& C) {
    mat_type D_mat;
    mat_type A_mat;
    mat_type B_mat;
    mat_type C_mat;

    reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  METAL_FUNC static constexpr void mma(
      thread mat_type& D,
      thread mat_type& A,
      thread mat_type& B,
      thread mat_type& C) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  STEEL_CONST int kTileRows = kTileRows_;
  STEEL_CONST int kTileCols = kTileCols_;

  STEEL_CONST int kRows = kTileRows * kFragRows;
  STEEL_CONST int kCols = kTileCols * kFragCols;

  STEEL_CONST int kNumFrags = kTileRows * kTileCols;
  STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags] = {frag_type(0)};

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC mat_type mat_at(const short i, const short j) {
    mat_type val_mat;
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < kElemsPerFrag; ++ii) {
      val_mat.thread_elements()[ii] = frag_at(i, j)[ii];
    }
    return val_mat;
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(
                dst[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void load(const device U* src, const int ld) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(src[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load_safe(
            frag_at(i, j),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_slice(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }
};

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void tile_matmad(
    thread MMATile<T, M, N>& D,
    thread MMATile<U, M, K>& A,
    thread MMATile<U, K, N>& B,
    thread MMATile<T, M, N>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short n_serp = (m % 2) ? (N - 1 - n) : n;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMATile<T, M, N>::MMAFrag_t::mma(
            D.frag_at(m, n_serp),
            A.frag_at(m, k),
            B.frag_at(k, n_serp),
            C.frag_at(m, n_serp));
      }
    }
  }
}

template <typename InT>
struct TransformNone<complex64_t, InT> {
  static METAL_FUNC complex64_t apply(complex64_t x) {
    return x;
  }
  static METAL_FUNC complex64_t apply(complex64_t x, complex64_t) {
    return x;
  }
};

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  // MMAFrag size
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = kFragSize * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = kFragSize * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / (kFragSize * WM);
  // Warp tile size along N
  STEEL_CONST short TN = BN / (kFragSize * WN);

  // Threadgroup A strides
  STEEL_CONST short A_str_m = transpose_a ? 1 : lda_tgp; // M
  STEEL_CONST short A_str_k = transpose_a ? lda_tgp : 1; // K

  // Threadgroup B strides
  STEEL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp; // K
  STEEL_CONST short B_str_n = transpose_b ? ldb_tgp : 1; // N

  // Threadgroup strides along K
  STEEL_CONST short tile_stride_a = kFragSize * A_str_k;
  STEEL_CONST short tile_stride_b = kFragSize * B_str_k;

  // Simdgroup matrices
  MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;

  // Offsets within threadgroup
  short sm;
  short sn;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    // Determine thread position in simdgroup matrix
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);

    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    // Determine thread and simdgroup offset
    As_offset = (tm + sm) * A_str_m + (sn)*A_str_k; // M, K
    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n; // K, N

    sm += tm;
    sn += tn;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of kFragSize
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);

      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);

      simdgroup_barrier(mem_flags::mem_none);

      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Ctile, Atile, Btile, Ctile);

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* D, const int ldd) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    Ctile.template store<U, WM, WN>(D, ldd);
  }

  METAL_FUNC void
  store_result_slice(device U* D, const int ldd, short2 start, short2 stop) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    D += sm * ldd + sn;
    start -= short2(sn, sm);
    stop -= short2(sn, sm);

    // TODO: Check the start as well
    if (stop.y <= 0 || stop.x <= 0) {
      return;
    }

    Ctile.template store_slice<U, WM, WN>(D, ldd, start, stop);
  }

  METAL_FUNC void
  store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
  }

  /* Apply epilogue */
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue& epilogue_op) {
    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = epilogue_op.apply(Ctile.elems()[i]);
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          accum[k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

        // Read C
        U c_elems[kelems] = {0};

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          if ((j * TN_stride + k) < dst_tile_dims.x) {
            c_elems[k] = C[offset_c + k * fdc];
          }
        }

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          accum[k] = epilogue_op.apply(accum[k], c_elems[k]);
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;

    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          D[offset_d + k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = Ctile.frag_at(i, j);
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < kelems; k++) {
            if ((j * TN_stride + k) < dst_tile_dims.x) {
              D[offset_d + k] =
                  epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
            }
          }
        }
      }
    }
  }
};

template <
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType,
    typename Epilogue>
struct BlockMMA<
    complex64_t,
    U,
    BM,
    BN,
    BK,
    WM,
    WN,
    transpose_a,
    transpose_b,
    lda_tgp,
    ldb_tgp,
    AccumType,
    Epilogue> {
  static_assert(
      metal::is_same_v<AccumType, float>,
      "BlockMMA<complex64_t,...> expects float accumulators");
  static_assert(
      metal::is_same_v<U, complex64_t>,
      "For complex BlockMMA, U must be complex64_t; use a different epilogue for projections");
  // MMAFrag size
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = kFragSize * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = kFragSize * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / (kFragSize * WM);
  // Warp tile size along N
  STEEL_CONST short TN = BN / (kFragSize * WN);

  // Threadgroup A strides
  STEEL_CONST short A_str_m = transpose_a ? 1 : lda_tgp; // M
  STEEL_CONST short A_str_k = transpose_a ? lda_tgp : 1; // K

  // Threadgroup B strides
  STEEL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp; // K
  STEEL_CONST short B_str_n = transpose_b ? ldb_tgp : 1; // N

  // Threadgroup strides along K
  STEEL_CONST short tile_stride_a = kFragSize * A_str_k;
  STEEL_CONST short tile_stride_b = kFragSize * B_str_k;

  // When indexing complex as float[2]
  STEEL_CONST short A_str_m_f = A_str_m * 2;
  STEEL_CONST short A_str_k_f = A_str_k * 2;
  STEEL_CONST short B_str_k_f = B_str_k * 2;
  STEEL_CONST short B_str_n_f = B_str_n * 2;
  STEEL_CONST short tile_stride_a_f = tile_stride_a * 2;
  STEEL_CONST short tile_stride_b_f = tile_stride_b * 2;

  // Accumulators (real/imag)
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile_r;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile_i;

  // Offsets within threadgroup
  short sm, sn;
  short As_offset, Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    // Determine thread position in simdgroup matrix
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);

    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    // Determine thread and simdgroup offset
    As_offset = (tm + sm) * A_str_m + (sn)*A_str_k; // (M,K)
    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n; // (K,N)

    sm += tm;
    sn += tn;
  }

  /* Karatsuba MMA: 3 real MMAs per K-chunk */
  METAL_FUNC void mma(
      const threadgroup complex64_t* As,
      const threadgroup complex64_t* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;
    threadgroup const float* As_f =
        reinterpret_cast<threadgroup const float*>(As);
    threadgroup const float* Bs_f =
        reinterpret_cast<threadgroup const float*>(Bs);

    // Iterate over BK in blocks of kFragSize
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);

      MMATile<AccumType, TM, 1, MMAFrag_acc_t> Ar, Ai;
      Ar.template load<float, WM, 1, A_str_m_f, A_str_k_f>(As_f + 0);
      Ai.template load<float, WM, 1, A_str_m_f, A_str_k_f>(As_f + 1);

      simdgroup_barrier(mem_flags::mem_none);

      MMATile<AccumType, 1, TN, MMAFrag_acc_t> Br, Bi;
      Br.template load<float, 1, WN, B_str_k_f, B_str_n_f>(Bs_f + 0);
      Bi.template load<float, 1, WN, B_str_k_f, B_str_n_f>(Bs_f + 1);

      simdgroup_barrier(mem_flags::mem_none);

      // P = Ar*Br ; Q = Ai*Bi ; R = (Ar+Ai)*(Br+Bi)
      MMATile<AccumType, TM, TN, MMAFrag_acc_t> P, Q, R;

      tile_matmad(P, Ar, Br, P);
      tile_matmad(Q, Ai, Bi, Q);

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(Ar)::kElemsPerTile; ++i)
        Ar.elems()[i] += Ai.elems()[i];
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(Br)::kElemsPerTile; ++i)
        Br.elems()[i] += Bi.elems()[i];

      tile_matmad(R, Ar, Br, R);

      // C_r += P - Q ; C_i -= Q
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(Ctile_r)::kElemsPerTile; ++i) {
        const auto p = P.elems()[i];
        const auto q = Q.elems()[i];
        const auto r = R.elems()[i];
        Ctile_r.elems()[i] += (p - q);
        Ctile_i.elems()[i] += (r - p - q);
      }

      // Progress to next simdgroup tile
      As_f += tile_stride_a_f;
      Bs_f += tile_stride_b_f;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* D, const int ldd) {
    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        thread const auto& r = Ctile_r.frag_at(i, j);
        thread const auto& im = Ctile_i.frag_at(i, j);
        int off = (i * TM_stride) * ldd + (j * TN_stride);
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile_r)::kElemsPerFrag; k++) {
          D[off + k] = Epilogue::apply(complex64_t(r[k], im[k]));
        }
      }
    }
  }

  METAL_FUNC void
  store_result_slice(device U* D, const int ldd, short2 start, short2 stop) {
    D += sm * ldd + sn;
    start -= short2(sn, sm);
    stop -= short2(sn, sm);

    if (stop.y <= 0 || stop.x <= 0)
      return;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; ++i) {
      const int row = i * TM_stride;
      if (row >= start.y && row < stop.y) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; ++j) {
          const int off = row * ldd + (j * TN_stride);
          thread const auto& r = Ctile_r.frag_at(i, j);
          thread const auto& im = Ctile_i.frag_at(i, j);

          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < decltype(Ctile_r)::kElemsPerFrag; ++k) {
            const int col = j * TN_stride + k;
            if (col >= start.x && col < stop.x) {
              D[off + k] = Epilogue::apply(complex64_t(r[k], im[k]));
            }
          }
        }
      }
    }
  }

  METAL_FUNC void
  store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) {
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          int off = (i * TM_stride) * ldd + (j * TN_stride);
          thread const auto& r = Ctile_r.frag_at(i, j);
          thread const auto& im = Ctile_i.frag_at(i, j);
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < decltype(Ctile_r)::kElemsPerFrag; k++) {
            if ((j * TN_stride + k) < dst_tile_dims.x) {
              D[off + k] = Epilogue::apply(complex64_t(r[k], im[k]));
            }
          }
        }
      }
    }
  }

  /* Apply epilogue */
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue& epilogue_op) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile_r)::kElemsPerTile; i++) {
      complex64_t out = epilogue_op.apply(
          complex64_t(Ctile_r.elems()[i], Ctile_i.elems()[i]));
      Ctile_r.elems()[i] = out.real;
      Ctile_i.elems()[i] = out.imag;
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in Cr, Ci
        thread auto& r = Ctile_r.frag_at(i, j);
        thread auto& im = Ctile_i.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile_r)::kElemsPerFrag; k++) {
          complex64_t out = epilogue_op.apply(
              complex64_t(r[k], im[k]), C[offset_c + k * fdc]);
          r[k] = out.real;
          im[k] = out.imag;
        }
      }
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const BinaryEpilogue& epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in Cr, Ci
        thread auto& r = Ctile_r.frag_at(i, j);
        thread auto& im = Ctile_i.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        constexpr short kelems = decltype(Ctile_r)::kElemsPerFrag;
        complex64_t tmp[kelems];

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          if ((j * TN_stride + k) < dst_tile_dims.x &&
              (i * TM_stride) < dst_tile_dims.y) {
            tmp[k] = C[offset_c + k * fdc];
          } else {
            tmp[k] = complex64_t(0.0f, 0.0f);
          }
        }

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          complex64_t out = epilogue_op.apply(complex64_t(r[k], im[k]), tmp[k]);
          r[k] = out.real;
          im[k] = out.imag;
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;

    constexpr short kelems = decltype(Ctile_r)::kElemsPerFrag;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in Cr, Ci
        thread const auto& r = Ctile_r.frag_at(i, j);
        thread const auto& im = Ctile_i.frag_at(i, j);
        int off_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int off_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          D[off_d + k] =
              epilogue_op.apply(complex64_t(r[k], im[k]), C[off_c + k * fdc]);
        }
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    constexpr short kelems = decltype(Ctile_r)::kElemsPerFrag;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in Cr, Ci
          thread const auto& r = Ctile_r.frag_at(i, j);
          thread const auto& im = Ctile_i.frag_at(i, j);
          int off_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int off_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < kelems; k++) {
            if ((j * TN_stride + k) < dst_tile_dims.x) {
              D[off_d + k] = epilogue_op.apply(
                  complex64_t(r[k], im[k]), C[off_c + k * fdc]);
            }
          }
        }
      }
    }
  }
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/conv/conv.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/conv/conv.h"
// Copyright © 2024 Apple Inc.




using namespace metal;
using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
