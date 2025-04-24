namespace mlx::core::metal {

const char* conv() {
  return R"preamble(
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

template <int NDIM>
struct MLXConvParams {
  const int N;
  const int C;
  const int O;
  const int iS[NDIM];
  const int wS[NDIM];
  const int oS[NDIM];
  const int str[NDIM];
  const int pad[NDIM];
  const int kdil[NDIM];
  const int idil[NDIM];
  const int64_t in_strides[NDIM + 2];
  const int64_t wt_strides[NDIM + 2];
  const int64_t out_strides[NDIM + 2];
  const int groups;
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
}
}
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
  static constant constexpr const short BROWS = BM;
  static constant constexpr const short BCOLS = BK;
  static constant constexpr const short dst_ld = BCOLS + tgp_padding;
  static constant constexpr const short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;
  static constant constexpr const short TCOLS = BCOLS / vec_size;
  static constant constexpr const short TROWS = tgp_size / TCOLS;
  static constant constexpr const short n_rows = BROWS / TROWS;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;
  short weight_h;
  short weight_w;
  const device T* src[n_rows];
  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];
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
#pragma clang loop unroll(full)
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
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2] + bj;
    }
  }
  METAL_FUNC void load_unsafe() const {
#pragma clang loop unroll(full)
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      int n = read_n[i];
      int ih = read_ih[i] + weight_h * params->kdil[0];
      int iw = read_iw[i] + weight_w * params->kdil[1];
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
        }
      }
      else {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }
  METAL_FUNC void next() {
    if (++weight_w < params->wS[1]) {
#pragma clang loop unroll(full)
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }
      return;
    }
    weight_w = 0;
    if (++weight_h < params->wS[0]) {
#pragma clang loop unroll(full)
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }
      return;
    }
    weight_h = 0;
#pragma clang loop unroll(full)
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
  static constant constexpr const short BROWS = BM;
  static constant constexpr const short BCOLS = BK;
  static constant constexpr const short dst_ld = BCOLS + tgp_padding;
  static constant constexpr const short vec_size = tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4;
  static constant constexpr const short TCOLS = BCOLS / vec_size;
  static constant constexpr const short TROWS = tgp_size / TCOLS;
  static constant constexpr const short n_rows = BROWS / TROWS;
  using mask_t = short;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;
  short weight_h;
  short weight_w;
  const device T* src[n_rows];
  mask_t mask_h[n_rows];
  mask_t mask_w[n_rows];
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
#pragma clang loop unroll(full)
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
      if (params->flip) {
        ih += (params->wS[0] - 1) * params->kdil[0];
        iw += (params->wS[1] - 1) * params->kdil[1];
      }
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2] + bj;
    }
#pragma clang loop unroll(full)
    for (short i = 0; i < n_rows; ++i) {
      mask_h[i] = 0;
      mask_w[i] = 0;
    }
    for (short kh = 0; kh < params->wS[0]; kh++) {
      short flip_h = params->flip ? params->wS[0] - kh - 1 : kh;
#pragma clang loop unroll(full)
      for (short i = 0; i < n_rows; ++i) {
        int n = read_n[i];
        int ih = read_ih[i] + flip_h * params->kdil[0];
        bool in_bounds = n < params->N && ih >= 0 && ih < params->iS[0];
        mask_h[i] |= (in_bounds << kh);
      }
    }
    for (short kw = 0; kw < params->wS[1]; kw++) {
      short flip_w = params->flip ? params->wS[1] - kw - 1 : kw;
#pragma clang loop unroll(full)
      for (short i = 0; i < n_rows; ++i) {
        int iw = read_iw[i] + flip_w * params->kdil[1];
        bool in_bounds = iw >= 0 && iw < params->iS[1];
        mask_w[i] |= (in_bounds << kw);
      }
    }
  }
  METAL_FUNC void load_unsafe() const {
    mask_t h_mask = mask_t(1) << weight_h;
    mask_t w_mask = mask_t(1) << weight_w;
#pragma clang loop unroll(full)
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      if ((mask_h[i] & h_mask) && (mask_w[i] & w_mask)) {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = src[i][j];
        }
      }
      else {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }
  METAL_FUNC void next() {
    if (++weight_w < params->wS[1]) {
#pragma clang loop unroll(full)
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_w;
      }
      return;
    }
    weight_w = 0;
    if (++weight_h < params->wS[0]) {
#pragma clang loop unroll(full)
      for (short i = 0; i < n_rows; i++) {
        src[i] += gemm_params->inp_jump_h;
      }
      return;
    }
    weight_h = 0;
#pragma clang loop unroll(full)
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
  static constant constexpr const short BROWS = BN;
  static constant constexpr const short BCOLS = BK;
  static constant constexpr const short dst_ld = BCOLS + tgp_padding;
  static constant constexpr const short vec_size =
      (BN == 8) ? 1 : (tgp_size / (BROWS * BCOLS) >= 8 ? 8 : 4);
  static constant constexpr const short TCOLS = BCOLS / vec_size;
  static constant constexpr const short TROWS = tgp_size / TCOLS;
  static constant constexpr const short n_rows = BROWS / TROWS;
  const int src_ld;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;
  const constant MLXConvParams<2>* params;
  int weight_hw;
  const int read_n;
  const bool do_read;
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
        read_n(offsets.y + bi),
        do_read(read_n + n_rows * TROWS <= gemm_params_->N) {}
  METAL_FUNC void load_unsafe() const {
    if (BN != 8 || do_read) {
#pragma clang loop unroll(full)
      for (short i = 0; i < BN; i += TROWS) {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((read_n + i) < params->O) {
#pragma clang loop unroll(full)
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = src[i * src_ld + j];
          }
        } else {
#pragma clang loop unroll(full)
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }
  METAL_FUNC void next() {
    if (++weight_hw < (params->wS[1] * params->wS[0])) {
      src += params->wt_strides[2];
      return;
    }
    weight_hw = 0;
    src += BK - (params->wS[1] * params->wS[0] - 1) * params->wt_strides[2];
  }
};
}
}
namespace mlx {
namespace steel {
template <short n_channels_>
struct ChannelHelper {
  static constant constexpr const short n_channels = n_channels_;
  static constant constexpr const short vec_size = n_channels_ <= 4 ? 4 : 8;
  static constant constexpr const short excess = vec_size - n_channels_;
};
template <>
struct ChannelHelper<1> {
  static constant constexpr const short n_channels = 1;
  static constant constexpr const short vec_size = 1;
  static constant constexpr const short excess = 0;
};
template <>
struct ChannelHelper<2> {
  static constant constexpr const short n_channels = 2;
  static constant constexpr const short vec_size = 2;
  static constant constexpr const short excess = 0;
};
template <>
struct ChannelHelper<3> {
  static constant constexpr const short n_channels = 3;
  static constant constexpr const short vec_size = 4;
  static constant constexpr const short excess = 1;
};
template <>
struct ChannelHelper<4> {
  static constant constexpr const short n_channels = 4;
  static constant constexpr const short vec_size = 4;
  static constant constexpr const short excess = 0;
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
  static constant constexpr const short BROWS = BM;
  static constant constexpr const short BCOLS = BK;
  static constant constexpr const short dst_ld = BCOLS + tgp_padding;
  static constant constexpr const short vec_size = ChannelHelper<n_channels>::vec_size;
  static constant constexpr const short TCOLS = BCOLS / vec_size;
  static constant constexpr const short TROWS = tgp_size / TCOLS;
  static constant constexpr const short n_rows = BROWS / TROWS;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const constant MLXConvParams<2>* params;
  const constant ImplicitGemmConv2DParams* gemm_params;
  short weight_hw;
  const device T* src[n_rows];
  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];
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
#pragma clang loop unroll(full)
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / out_n_pixels;
      int hw = offset_nhw % out_n_pixels;
      int oh = hw / params->oS[1];
      int ow = hw % params->oS[1];
      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];
      src[i] = src_ + n * params->in_strides[0] + ih * params->in_strides[1] +
          iw * params->in_strides[2];
      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;
    }
  }
  METAL_FUNC void load_unsafe() const {
    if (weight_hw >= params->wS[1] * params->wS[0]) {
#pragma clang loop unroll(full)
      for (short i = 0; i < BROWS; i += TROWS) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      int n = read_n[i];
      int ih = read_ih[i] + weight_h;
      int iw = read_iw[i] + weight_w;
      if ((n < params->N) && (ih >= 0 && ih < params->iS[0]) &&
          (iw >= 0 && iw < params->iS[1])) {
        const device T* curr_src = src[i] + weight_h * params->in_strides[1] +
            weight_w * params->in_strides[2];
#pragma clang loop unroll(full)
        for (short j = 0; j < n_channels; ++j) {
          dst[is * dst_ld + j] = curr_src[j];
        }
#pragma clang loop unroll(full)
        for (short j = n_channels; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
      else {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = T(0);
        }
      }
    }
  }
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
  static constant constexpr const short BROWS = BN;
  static constant constexpr const short BCOLS = BK;
  static constant constexpr const short dst_ld = BCOLS + tgp_padding;
  static constant constexpr const short vec_size = ChannelHelper<n_channels>::vec_size;
  static constant constexpr const short TCOLS = BCOLS / vec_size;
  static constant constexpr const short TROWS = tgp_size / TCOLS;
  static constant constexpr const short n_rows = BROWS / TROWS;
  const int src_ld;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;
  const constant MLXConvParams<2>* params;
  int weight_hw;
  const int read_n;
  const bool do_read;
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
  METAL_FUNC void load_unsafe() const {
    if (bi >= BROWS || bj >= BCOLS)
      return;
    if (read_n >= params->O || weight_hw >= params->wS[1] * params->wS[0]) {
#pragma clang loop unroll(full)
      for (short i = 0; i < BROWS; i += TROWS) {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }
    const device T* curr_src = src + weight_hw * params->wt_strides[2];
    if (BN != 8 || do_read) {
#pragma clang loop unroll(full)
      for (short i = 0; i < BROWS; i += TROWS) {
#pragma clang loop unroll(full)
        for (short j = 0; j < n_channels; j++) {
          dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        }
#pragma clang loop unroll(full)
        for (short j = n_channels; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
    } else {
      for (short i = 0; i < BROWS; i += TROWS) {
        if (((read_n + i) < params->O)) {
#pragma clang loop unroll(full)
          for (short j = 0; j < n_channels; j++) {
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
          }
#pragma clang loop unroll(full)
          for (short j = n_channels; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        } else {
#pragma clang loop unroll(full)
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = T(0);
          }
        }
      }
    }
  }
  METAL_FUNC void next() {
    weight_hw += TCOLS;
  }
};
}
}

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
    return static_cast<OutT>(x * alpha + (beta * c));
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
}
}
#pragma METAL internals : enable
namespace metal {
template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};
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
}
#pragma METAL internals : disable

#pragma METAL internals : enable
namespace mlx {
namespace steel {
template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;
  METAL_FUNC constexpr operator value_type() const noexcept {
    return value;
  }
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
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator+( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv + uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator-( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv - uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator*( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv * uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator/( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv / uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator==( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv == uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator!=( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv != uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator<( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv < uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator>( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv > uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator<=( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv <= uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator>=( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv >= uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator&&( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv && uv; return integral_constant<decltype(res), res>{}; };
template <typename T, T tv, typename U, U uv> METAL_FUNC constexpr auto operator||( integral_constant<T, tv>, integral_constant<U, uv>) { constexpr auto res = tv || uv; return integral_constant<decltype(res), res>{}; };
template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}
template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}
}
}
#pragma METAL internals : disable

using namespace metal;
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
  static constant constexpr const int kFragRows = 8;
  static constant constexpr const int kFragCols = 8;
  static constant constexpr const int kElemsPerFrag = (kFragRows * kFragCols) / 32;
  static constant constexpr const int kElemRows = 1;
  static constant constexpr const int kElemCols = 2;
  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");
  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;
  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id
                                               [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }
  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  load(thread frag_type& dst, SrcPtrType src, StrX str_x, StrY str_y) {
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; i++) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; i++) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; i++) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; i++) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0; i < kElemRows; i++) {
#pragma clang loop unroll(full)
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
  static constant constexpr const int kFragRows = MMAFrag_t::kFragRows;
  static constant constexpr const int kFragCols = MMAFrag_t::kFragCols;
  static constant constexpr const int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;
  static constant constexpr const int kTileRows = kTileRows_;
  static constant constexpr const int kTileCols = kTileCols_;
  static constant constexpr const int kRows = kTileRows * kFragRows;
  static constant constexpr const int kCols = kTileCols * kFragCols;
  static constant constexpr const int kNumFrags = kTileRows * kTileCols;
  static constant constexpr const int kElemsPerTile = kNumFrags * kElemsPerFrag;
  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;
  frag_type val_frags[kNumFrags] = {frag_type(0)};
  METAL_FUNC MMATile() thread {}
  METAL_FUNC constexpr void clear() {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0; i < kTileRows; ++i) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0; i < kTileRows; ++i) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0; i < kTileRows; ++i) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (short i = 0; i < kTileRows; ++i) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (int i = 0; i < kTileRows; ++i) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (int i = 0; i < kTileRows; ++i) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
    for (int i = 0; i < kTileRows; ++i) {
#pragma clang loop unroll(full)
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
#pragma clang loop unroll(full)
  for (short m = 0; m < M; ++m) {
#pragma clang loop unroll(full)
    for (short n = 0; n < N; ++n) {
      short n_serp = (m % 2) ? (N - 1 - n) : n;
#pragma clang loop unroll(full)
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
  static constant constexpr const short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  static constant constexpr const short TM_stride = kFragSize * WM;
  static constant constexpr const short TN_stride = kFragSize * WN;
  static constant constexpr const short TM = BM / (kFragSize * WM);
  static constant constexpr const short TN = BN / (kFragSize * WN);
  static constant constexpr const short A_str_m = transpose_a ? 1 : lda_tgp;
  static constant constexpr const short A_str_k = transpose_a ? lda_tgp : 1;
  static constant constexpr const short B_str_k = transpose_b ? 1 : ldb_tgp;
  static constant constexpr const short B_str_n = transpose_b ? ldb_tgp : 1;
  static constant constexpr const short tile_stride_a = kFragSize * A_str_k;
  static constant constexpr const short tile_stride_b = kFragSize * B_str_k;
  MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;
  short sm;
  short sn;
  short As_offset;
  short Bs_offset;
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);
    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;
    As_offset = (tm + sm) * A_str_m + (sn)*A_str_k;
    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n;
    sm += tm;
    sn += tn;
  }
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    As += As_offset;
    Bs += Bs_offset;
#pragma clang loop unroll(full)
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);
      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);
      simdgroup_barrier(mem_flags::mem_none);
      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Ctile, Atile, Btile, Ctile);
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }
  METAL_FUNC void store_result(device U* D, const int ldd) {
#pragma clang loop unroll(full)
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }
    D += sm * ldd + sn;
    Ctile.template store<U, WM, WN>(D, ldd);
  }
  METAL_FUNC void
  store_result_slice(device U* D, const int ldd, short2 start, short2 stop) {
#pragma clang loop unroll(full)
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }
    D += sm * ldd + sn;
    start -= short2(sn, sm);
    stop -= short2(sn, sm);
    if (stop.y <= 0 || stop.x <= 0) {
      return;
    }
    Ctile.template store_slice<U, WM, WN>(D, ldd, start, stop);
  }
  METAL_FUNC void
  store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) {
#pragma clang loop unroll(full)
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
    Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
  }
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue& epilogue_op) {
#pragma clang loop unroll(full)
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = epilogue_op.apply(Ctile.elems()[i]);
    }
  }
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) {
    C += (sm)*ldc + (sn)*fdc;
#pragma clang loop unroll(full)
    for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
#pragma clang loop unroll(full)
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          accum[k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const BinaryEpilogue& epilogue_op) {
    C += (sm)*ldc + (sn)*fdc;
    dst_tile_dims -= short2(sn, sm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
#pragma clang loop unroll(full)
    for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        thread auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        constexpr short kelems = decltype(Ctile)::kElemsPerFrag;
        U c_elems[kelems] = {0};
#pragma clang loop unroll(full)
        for (short k = 0; k < kelems; k++) {
          if ((j * TN_stride + k) < dst_tile_dims.x) {
            c_elems[k] = C[offset_c + k * fdc];
          }
        }
#pragma clang loop unroll(full)
        for (short k = 0; k < kelems; k++) {
          accum[k] = epilogue_op.apply(accum[k], c_elems[k]);
        }
      }
    }
  }
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;
    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;
#pragma clang loop unroll(full)
    for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        thread const auto& accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);
#pragma clang loop unroll(full)
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
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;
    dst_tile_dims -= short2(sn, sm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
#pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) {
          thread const auto& accum = Ctile.frag_at(i, j);
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);
#pragma clang loop unroll(full)
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
}
}

using namespace metal;
using namespace mlx::steel;
)preamble";
}

} // namespace mlx::core::metal
