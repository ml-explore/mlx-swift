namespace mlx::core::metal {

const char* conv() {
  return R"preamble(
METAL_FUNC ulong2 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
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
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
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
  const size_t in_strides[NDIM + 2];
  const size_t wt_strides[NDIM + 2];
  const size_t out_strides[NDIM + 2];
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

using namespace metal;
namespace mlx {
namespace steel {
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
  static constant constexpr const short TM_stride = 8 * WM;
  static constant constexpr const short TN_stride = 8 * WN;
  static constant constexpr const short TM = BM / TM_stride;
  static constant constexpr const short TN = BN / TN_stride;
  static constant constexpr const short simd_stride_a = {
      transpose_a ? TM_stride : TM_stride * lda_tgp};
  static constant constexpr const short simd_stride_b = {
      transpose_b ? TN_stride * ldb_tgp : TN_stride};
  static constant constexpr const short jump_a = {transpose_a ? lda_tgp : 1};
  static constant constexpr const short jump_b = {transpose_b ? ldb_tgp : 1};
  static constant constexpr const short tile_stride_a = {transpose_a ? 8 * lda_tgp : 8};
  static constant constexpr const short tile_stride_b = {transpose_b ? 8 : 8 * ldb_tgp};
  simdgroup_matrix<AccumType, 8, 8> Asimd[TM];
  simdgroup_matrix<AccumType, 8, 8> Bsimd[TN];
  simdgroup_matrix<AccumType, 8, 8> results[TM * TN] = {
      simdgroup_matrix<AccumType, 8, 8>(0)};
  const short tm;
  const short tn;
  short sm;
  short sn;
  short As_offset;
  short Bs_offset;
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : tm(8 * (simd_group_id / WN)), tn(8 * (simd_group_id % WN)) {
    short qid = simd_lane_id / 4;
    sm = (qid & 4) + (simd_lane_id / 2) % 4;
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    As_offset =
        transpose_a ? ((sn)*lda_tgp + (tm + sm)) : ((sn) + (tm + sm) * lda_tgp);
    Bs_offset =
        transpose_b ? ((tn + sn) * ldb_tgp + (sm)) : ((sm)*ldb_tgp + (tn + sn));
  }
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    As += As_offset;
    Bs += Bs_offset;
#pragma clang loop unroll(full)
    for (short kk = 0; kk < BK; kk += 8) {
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (short i = 0; i < TM; i++) {
        Asimd[i].thread_elements()[0] =
            static_cast<AccumType>(As[i * simd_stride_a + 0]);
        Asimd[i].thread_elements()[1] =
            static_cast<AccumType>(As[i * simd_stride_a + jump_a]);
      }
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        Bsimd[j].thread_elements()[0] =
            static_cast<AccumType>(Bs[j * simd_stride_b + 0]);
        Bsimd[j].thread_elements()[1] =
            static_cast<AccumType>(Bs[j * simd_stride_b + jump_b]);
      }
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
        for (short j = 0; j < TN; j++) {
          short j_serp = (i % 2) ? (TN - 1 - j) : j;
          simdgroup_multiply_accumulate(
              results[i * TN + j_serp],
              Asimd[i],
              Bsimd[j_serp],
              results[i * TN + j_serp]);
        }
      }
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }
  METAL_FUNC void store_result(device U* D, const int ldd) const {
    D += (sm + tm) * ldd + tn + sn;
#pragma clang loop unroll(full)
    for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset = (i * TM_stride) * ldd + (j * TN_stride);
        U outs[2] = {Epilogue::apply(accum[0]), Epilogue::apply(accum[1])};
        D[offset] = outs[0];
        D[offset + 1] = outs[1];
      }
    }
  }
  METAL_FUNC void
  store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) const {
    D += (sm + tm) * ldd + (tn + sn);
    dst_tile_dims -= short2(tn + sn, sm + tm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
#pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) {
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset = (i * TM_stride) * ldd + (j * TN_stride);
          if (j * TN_stride < dst_tile_dims.x) {
            D[offset] = Epilogue::apply(accum[0]);
          }
          if (j * TN_stride + 1 < dst_tile_dims.x) {
            D[offset + 1] = Epilogue::apply(accum[1]);
          }
        }
      }
    }
  }
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue& epilogue_op) {
#pragma clang loop unroll(full)
    for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        thread auto& accum = results[i * TN + j].thread_elements();
        accum[0] = epilogue_op.apply(accum[0]);
        accum[1] = epilogue_op.apply(accum[1]);
      }
    }
  }
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int ldc,
      const int fdc,
      thread const BinaryEpilogue& epilogue_op) {
    C += (sm + tm) * ldc + (tn + sn) * fdc;
#pragma clang loop unroll(full)
    for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        thread auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        accum[0] = epilogue_op.apply(accum[0], C[offset_c]);
        accum[1] = epilogue_op.apply(accum[1], C[offset_c + fdc]);
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
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    dst_tile_dims -= short2(tn + sn, sm + tm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
#pragma clang loop unroll(full)
    for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        thread auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        U c_elems[2] = {0};
        if ((j * TN_stride + 1) < dst_tile_dims.x) {
          c_elems[0] = C[offset_c];
          c_elems[1] = C[offset_c + fdc];
        } else if ((j * TN_stride) < dst_tile_dims.x) {
          c_elems[0] = C[offset_c];
        }
        accum[0] = epilogue_op.apply(accum[0], c_elems[0]);
        accum[1] = epilogue_op.apply(accum[1], c_elems[1]);
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
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;
#pragma clang loop unroll(full)
    for (short i = 0; i < TM; i++) {
#pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);
        U outs[2] = {
            epilogue_op.apply(accum[0], C[offset_c]),
            epilogue_op.apply(accum[1], C[offset_c + fdc])};
        D[offset_d] = outs[0];
        D[offset_d + 1] = outs[1];
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
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;
    dst_tile_dims -= short2(tn + sn, sm + tm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
#pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
#pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) {
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);
          if (j * TN_stride < dst_tile_dims.x) {
            D[offset_d] = epilogue_op.apply(accum[0], C[offset_c]);
          }
          if (j * TN_stride + 1 < dst_tile_dims.x) {
            D[offset_d + 1] = epilogue_op.apply(accum[1], C[offset_c + fdc]);
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
