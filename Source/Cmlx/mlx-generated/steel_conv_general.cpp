namespace mlx::core::metal {

const char* steel_conv_general() {
  return R"preamble(
namespace mlx {
namespace steel {
template <
    typename T,
    short BM,
    short BN,
    short BK,
    short tgp_size,
    short tgp_padding = 0>
struct Conv2DInputBlockLoaderGeneral {
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
  const constant Conv2DGeneralJumpParams* jump_params;
  const short base_wh;
  const short base_ww;
  short weight_h;
  short weight_w;
  const device T* src[n_rows];
  int read_n[n_rows];
  int read_ih[n_rows];
  int read_iw[n_rows];
  METAL_FUNC Conv2DInputBlockLoaderGeneral(
      const device T* src_,
      threadgroup T* dst_,
      const int4 offsets,
      const constant MLXConvParams<2>* params_,
      const constant Conv2DGeneralJumpParams* jump_params_,
      const short base_wh_,
      const short base_ww_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        params(params_),
        jump_params(jump_params_),
        base_wh(base_wh_),
        base_ww(base_ww_),
        weight_h(base_wh_),
        weight_w(base_ww_) {
#pragma clang loop unroll(full)
    for (short i = 0; i < n_rows; ++i) {
      int offset_nhw = offsets.y + bi + i * TROWS;
      int n = offset_nhw / jump_params->adj_out_hw;
      int hw = offset_nhw % jump_params->adj_out_hw;
      int oh =
          (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h + offsets.z;
      int ow =
          (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w + offsets.w;
      int ih = oh * params->str[0] - params->pad[0];
      int iw = ow * params->str[1] - params->pad[1];
      read_n[i] = n;
      read_ih[i] = ih;
      read_iw[i] = iw;
      src[i] = src_ + n * params->in_strides[0] + bj;
    }
  }
  METAL_FUNC void load_unsafe() const {
#pragma clang loop unroll(full)
    for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
      int n = read_n[i];
      int h_flip = params->flip ? params->wS[0] - weight_h - 1 : weight_h;
      int w_flip = params->flip ? params->wS[1] - weight_w - 1 : weight_w;
      int ih_dil = read_ih[i] + h_flip * params->kdil[0];
      int iw_dil = read_iw[i] + w_flip * params->kdil[1];
      int ih = ih_dil / params->idil[0];
      int iw = iw_dil / params->idil[1];
      size_t offset = ih * params->in_strides[1] + iw * params->in_strides[2];
      if ((n < params->N) && (ih_dil >= 0 && ih < params->iS[0]) &&
          (iw_dil >= 0 && iw < params->iS[1])) {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst[is * dst_ld + j] = (src[i])[offset + j];
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
    weight_w += jump_params->f_wgt_jump_w;
    if (weight_w < params->wS[1]) {
      return;
    }
    weight_w = base_ww;
    weight_h += jump_params->f_wgt_jump_h;
    if (weight_h < params->wS[0]) {
      return;
    }
    weight_h = base_wh;
#pragma clang loop unroll(full)
    for (short i = 0; i < n_rows; i++) {
      src[i] += BK;
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
struct Conv2DWeightBlockLoaderGeneral {
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
  const constant Conv2DGeneralJumpParams* jump_params;
  const short base_wh;
  const short base_ww;
  short weight_h;
  short weight_w;
  const int start_row;
  METAL_FUNC Conv2DWeightBlockLoaderGeneral(
      const device T* src_,
      threadgroup T* dst_,
      const int2 offsets,
      const constant MLXConvParams<2>* params_,
      const constant Conv2DGeneralJumpParams* jump_params_,
      const short base_wh_,
      const short base_ww_,
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(params_->wt_strides[0]),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj),
        params(params_),
        jump_params(jump_params_),
        base_wh(base_wh_),
        base_ww(base_ww_),
        weight_h(base_wh_),
        weight_w(base_ww_),
        start_row(offsets.y + bi) {}
  METAL_FUNC void load_unsafe() const {
    const device T* curr_src = src + weight_h * params->wt_strides[1] +
        weight_w * params->wt_strides[2];
    if ((start_row + BN <= params->O)) {
#pragma clang loop unroll(full)
      for (short i = 0; i < BN; i += TROWS) {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = curr_src[i * src_ld + j];
        }
      }
    } else {
      for (short i = 0; i < BN; i += TROWS) {
        if ((start_row + i) < params->O) {
#pragma clang loop unroll(full)
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = curr_src[i * src_ld + j];
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
    weight_w += jump_params->f_wgt_jump_w;
    if (weight_w < params->wS[1]) {
      return;
    }
    weight_w = base_ww;
    weight_h += jump_params->f_wgt_jump_h;
    if (weight_h < params->wS[0]) {
      return;
    }
    weight_h = base_wh;
    src += BK;
  }
};
}
}

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename AccumType = float,
    typename Epilogue = TransformNone<T, AccumType>>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
implicit_gemm_conv_2d_general(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    const constant MLXConvParams<2>* params [[buffer(3)]],
    const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],
    const constant Conv2DGeneralJumpParams* jump_params [[buffer(5)]],
    const constant Conv2DGeneralBaseInfo* base_h [[buffer(6)]],
    const constant Conv2DGeneralBaseInfo* base_w [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;
  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;
  constexpr short tgp_padding_a = 16 / sizeof(T);
  constexpr short tgp_padding_b = 16 / sizeof(T);
  constexpr short shape_a_cols = (transpose_a ? BM : BK) + tgp_padding_a;
  constexpr short shape_b_cols = (transpose_b ? BK : BN) + tgp_padding_b;
  constexpr short shape_a_rows = (transpose_a ? BK : BM);
  constexpr short shape_b_rows = (transpose_b ? BN : BK);
  constexpr short tgp_mem_size_a = shape_a_cols * shape_a_rows;
  constexpr short tgp_mem_size_b = shape_b_cols * shape_b_rows;
  constexpr short tgp_size = WM * WN * 32;
  using loader_a_t =
      Conv2DInputBlockLoaderGeneral<T, BM, BN, BK, tgp_size, tgp_padding_a>;
  using loader_b_t =
      Conv2DWeightBlockLoaderGeneral<T, BM, BN, BK, tgp_size, tgp_padding_b>;
  using mma_t = BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      shape_a_cols,
      shape_b_cols>;
  threadgroup T As[tgp_mem_size_a];
  threadgroup T Bs[tgp_mem_size_b];
  const int tid_y = ((tid.y) << gemm_params->swizzle_log) +
      ((tid.x) & ((1 << gemm_params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> gemm_params->swizzle_log;
  if (gemm_params->tiles_n <= tid_x || gemm_params->tiles_m <= tid_y) {
    return;
  }
  const int tid_z = tid.z;
  const int base_oh = tid_z / jump_params->f_out_jump_w;
  const int base_ow = tid_z % jump_params->f_out_jump_w;
  const int base_wh = base_h[base_oh].weight_base;
  const int base_ww = base_w[base_ow].weight_base;
  const int base_wh_size = base_h[base_oh].weight_size;
  const int base_ww_size = base_w[base_ow].weight_size;
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const int K = gemm_params->K;
  B += c_col * K;
  const int4 offsets_a(0, c_row, base_oh, base_ow);
  const int2 offsets_b(0, c_col);
  loader_a_t loader_a(
      A,
      As,
      offsets_a,
      params,
      jump_params,
      base_wh,
      base_ww,
      simd_gid,
      simd_lid);
  loader_b_t loader_b(
      B,
      Bs,
      offsets_b,
      params,
      jump_params,
      base_wh,
      base_ww,
      simd_gid,
      simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  int gemm_k_iterations =
      base_wh_size * base_ww_size * gemm_params->gemm_k_iterations;
  for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();
    loader_b.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }
  threadgroup_barrier(mem_flags::mem_none);
  {
    int offset_m = c_row + mma_op.sm;
    int offset_n = c_col + mma_op.sn;
    C += offset_n;
    if (offset_n >= gemm_params->N)
      return;
    short diff = gemm_params->N - offset_n;
#pragma clang loop unroll(full)
    for (int i = 0; i < mma_t::TM; i++) {
      int cm = offset_m + i * mma_t::TM_stride;
      int n = cm / jump_params->adj_out_hw;
      int hw = cm % jump_params->adj_out_hw;
      int oh =
          (hw / jump_params->adj_out_w) * jump_params->f_out_jump_h + base_oh;
      int ow =
          (hw % jump_params->adj_out_w) * jump_params->f_out_jump_w + base_ow;
      if (n < params->N && oh < params->oS[0] && ow < params->oS[1]) {
        int offset_cm = n * params->out_strides[0] +
            oh * params->out_strides[1] + ow * params->out_strides[2];
#pragma clang loop unroll(full)
        for (int j = 0; j < mma_t::TN; j++) {
          thread const auto& accum = mma_op.Ctile.frag_at(i, j);
          int offset = offset_cm + (j * mma_t::TN_stride);
          constexpr short kelems = decltype(mma_op.Ctile)::kElemsPerFrag;
#pragma clang loop unroll(full)
          for (short k = 0; k < kelems; k++) {
            if ((j * mma_t::TN_stride + k) < diff) {
              C[offset + k] = Epilogue::apply(accum[k]);
            }
          }
        }
      }
    }
  }
}
)preamble";
}

} // namespace mlx::core::metal
