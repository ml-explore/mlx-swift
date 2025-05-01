namespace mlx::core::metal {

const char* gemm() {
  return R"preamble(
namespace mlx {
namespace steel {
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoader {
  static constant constexpr const short n_rows = (BROWS + TROWS - 1) / TROWS;
  static constant constexpr const short vec_size = n_reads;
  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;
  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };
  METAL_FUNC BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
#pragma clang loop unroll(full)
    for (short i = 0; i < BROWS; i += TROWS) {
#pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = op.apply(dst[i * dst_ld + j]);
      }
    }
  }
  METAL_FUNC void load_unsafe() const {
#pragma clang loop unroll(full)
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
#pragma clang loop unroll(full)
      for (short i = 0; i < BROWS; i += TROWS) {
#pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];
#pragma clang loop unroll(full)
    for (short i = 0; i < BROWS; i += TROWS) {
#pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }
#pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }
#pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
#pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = tmp_val[j];
      }
    }
  }
  METAL_FUNC void next() {
    src += tile_stride;
  }
};
}
}
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
namespace mlx {
namespace steel {
struct GEMMParams {
  const int M;
  const int N;
  const int K;
  const int lda;
  const int ldb;
  const int ldd;
  const int tiles_n;
  const int tiles_m;
  const int64_t batch_stride_a;
  const int64_t batch_stride_b;
  const int64_t batch_stride_d;
  const int swizzle_log;
  const int gemm_k_iterations_aligned;
  const int batch_ndim;
};
struct GEMMSpiltKParams {
  const int M;
  const int N;
  const int K;
  const int lda;
  const int ldb;
  const int ldc;
  const int tiles_n;
  const int tiles_m;
  const int split_k_partitions;
  const int split_k_partition_stride;
  const int split_k_partition_size;
  const int gemm_k_iterations_aligned;
};
struct GEMMAddMMParams {
  const int ldc;
  const int fdc;
  const int64_t batch_stride_c;
  const float alpha;
  const float beta;
};
}
}
using namespace metal;
namespace mlx {
namespace steel {
template <bool M_aligned, bool N_aligned, bool K_aligned>
struct LoopAlignment {};
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
    bool MN_aligned,
    bool K_aligned,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<U, AccumType>>
struct GEMMKernel {
  static constant constexpr const short tgp_padding_a = 16 / sizeof(T);
  static constant constexpr const short tgp_padding_b = 16 / sizeof(T);
  static constant constexpr const short tgp_mem_size_a =
      transpose_a ? BK * (BM + tgp_padding_a) : BM * (BK + tgp_padding_a);
  static constant constexpr const short tgp_mem_size_b =
      transpose_b ? BN * (BK + tgp_padding_b) : BK * (BN + tgp_padding_b);
  static constant constexpr const short tgp_mem_size = tgp_mem_size_a + tgp_mem_size_b;
  static constant constexpr const short tgp_size = WM * WN * 32;
  using loader_a_t = BlockLoader<
      T,
      transpose_a ? BK : BM,
      transpose_a ? BM : BK,
      transpose_a ? BM + tgp_padding_a : BK + tgp_padding_a,
      !transpose_a,
      tgp_size>;
  using loader_b_t = BlockLoader<
      T,
      transpose_b ? BN : BK,
      transpose_b ? BK : BN,
      transpose_b ? BK + tgp_padding_b : BN + tgp_padding_b,
      transpose_b,
      tgp_size>;
  using mma_t = BlockMMA<
      T,
      U,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      transpose_a ? BM + tgp_padding_a : BK + tgp_padding_a,
      transpose_b ? BK + tgp_padding_b : BN + tgp_padding_b,
      AccumType,
      Epilogue>;
  template <bool M_aligned, bool N_aligned, bool K_aligned_>
  static METAL_FUNC void gemm_loop(
      threadgroup T* As [[threadgroup(0)]],
      threadgroup T* Bs [[threadgroup(1)]],
      const int gemm_k_iterations,
      thread loader_a_t& loader_a,
      thread loader_b_t& loader_b,
      thread mma_t& mma_op,
      thread const short& tgp_bm,
      thread const short& tgp_bn,
      thread const short& lbk,
      LoopAlignment<M_aligned, N_aligned, K_aligned_> l = {}) {
    (void)l;
    short2 tile_dims_A = transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm);
    short2 tile_dims_B = transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK);
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (M_aligned) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(tile_dims_A);
      }
      if (N_aligned) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dims_B);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    if (!K_aligned_) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      short2 tile_dims_A_last =
          transpose_a ? short2(tgp_bm, lbk) : short2(lbk, tgp_bm);
      short2 tile_dims_B_last =
          transpose_b ? short2(lbk, tgp_bn) : short2(tgp_bn, lbk);
      loader_a.load_safe(tile_dims_A_last);
      loader_b.load_safe(tile_dims_B_last);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
    }
  }
  static METAL_FUNC void run(
      const device T* A [[buffer(0)]],
      const device T* B [[buffer(1)]],
      device U* D [[buffer(2)]],
      const constant GEMMParams* params [[buffer(3)]],
      threadgroup T* As [[threadgroup(0)]],
      threadgroup T* Bs [[threadgroup(1)]],
      uint simd_lane_id [[thread_index_in_simdgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]) {
    (void)lid;
    const int tid_y = ((tid.y) << params->swizzle_log) +
        ((tid.x) & ((1 << params->swizzle_log) - 1));
    const int tid_x = (tid.x) >> params->swizzle_log;
    if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
      return;
    }
    threadgroup_barrier(mem_flags::mem_none);
    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const size_t c_row_long = size_t(c_row);
    const size_t c_col_long = size_t(c_col);
    A += transpose_a ? c_row_long : c_row_long * params->lda;
    B += transpose_b ? c_col_long * params->ldb : c_col_long;
    D += c_row_long * params->ldd + c_col_long;
    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);
    thread mma_t mma_op(simd_group_id, simd_lane_id);
    int gemm_k_iterations = params->gemm_k_iterations_aligned;
    if (MN_aligned) {
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
      if (!K_aligned) {
        int lbk = params->K - params->gemm_k_iterations_aligned * BK;
        short2 tile_dims_A = transpose_a ? short2(BM, lbk) : short2(lbk, BM);
        short2 tile_dims_B = transpose_b ? short2(lbk, BN) : short2(BN, lbk);
        loader_a.load_safe(tile_dims_A);
        loader_b.load_safe(tile_dims_B);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
      }
      mma_op.store_result(D, params->ldd);
      return;
    }
    else {
      short tgp_bm = min(BM, params->M - c_row);
      short tgp_bn = min(BN, params->N - c_col);
      short leftover_bk = params->K - params->gemm_k_iterations_aligned * BK;
      if (tgp_bm == BM && tgp_bn == BN) {
        gemm_loop<true, true, K_aligned>(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            leftover_bk);
        mma_op.store_result(D, params->ldd);
        return;
      } else if (tgp_bn == BN) {
        gemm_loop<false, true, K_aligned>(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            leftover_bk);
        mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
        return;
      } else if (tgp_bm == BM) {
        gemm_loop<true, false, K_aligned>(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            leftover_bk);
        mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
        return;
      } else {
        gemm_loop<false, false, K_aligned>(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            leftover_bk);
        mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
        return;
      }
    }
  }
};
}
}
)preamble";
}

} // namespace mlx::core::metal
