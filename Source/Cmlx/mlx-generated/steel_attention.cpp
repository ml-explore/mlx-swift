namespace mlx::core::metal {

const char* steel_attention() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/defines.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/defines.h"
// Copyright © 2024 Apple Inc.


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/attn/loader.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/attn/loader.h"
// Copyright © 2024 Apple Inc.



///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

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
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  /* Constructor */
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

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = op.apply(dst[i * dst_ld + j]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out unneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tile_stride;
  }
};

template <int R, int C>
struct CShape {
  STEEL_CONST int kRows = R;
  STEEL_CONST int kCols = C;
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  /* Constructor */
  METAL_FUNC BlockLoaderT(
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
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld + bj) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] =
            op.apply(dst[i * kDstStrRow + j * kDstStrCol]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out unneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tile_stride;
  }
};

} // namespace steel
} // namespace mlx

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
// Contents from "mlx/backend/metal/kernels/steel/attn/transforms.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/attn/transforms.h"
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
// Contents from "mlx/backend/metal/kernels/steel/attn/mma.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/attn/mma.h"
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

template <typename RInt, typename CInt>
struct Shape2D {
  RInt r;
  CInt c;

  Shape2D(RInt r_, CInt c_) : r(r_), c(c_) {}
};

template <typename Shape, typename Layout>
struct Layout2D {
  Shape shape;
  Layout layout;
};

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
  typedef metal::vec<T, kElemRows> row_frag_type;
  typedef metal::vec<T, kElemCols> col_frag_type;

  template <typename U>
  using dtype_mat_t = typename metal::simdgroup_matrix<U, kFragRows, kFragCols>;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

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
    src += off_x * str_x + off_y * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] = static_cast<T>(src[0]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
        src += str_y;
      }
      src -= kElemCols * str_y;
      src += str_x;
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

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread dtype_frag_t<Atype>& A,
      thread dtype_frag_t<Btype>& B,
      thread dtype_frag_t<Ctype>& C) {
    mat_type D_mat;
    dtype_mat_t<Atype> A_mat;
    dtype_mat_t<Btype> B_mat;
    dtype_mat_t<Ctype> C_mat;

    reinterpret_cast<thread dtype_frag_t<Atype>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread dtype_frag_t<Btype>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread dtype_frag_t<Ctype>&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread mat_type& D,
      thread dtype_mat_t<Atype>& A,
      thread dtype_mat_t<Btype>& B,
      thread dtype_mat_t<Ctype>& C) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(
      thread const frag_type& inp_vals,
      thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);

    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(
      thread frag_type& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
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

  STEEL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;
  STEEL_CONST int kColsPerThread = kTileCols * MMAFrag_t::kElemCols;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

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

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
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
};

template <
    typename Dtype,
    typename Atype,
    typename Btype,
    typename Ctype,
    int M,
    int N,
    int K,
    class MMAFragD,
    class MMAFragA,
    class MMAFragB,
    class MMAFragC>
METAL_FUNC void tile_matmad(
    thread MMATile<Dtype, M, N, MMAFragD>& D,
    thread MMATile<Atype, M, K, MMAFragA>& A,
    thread MMATile<Btype, K, N, MMAFragB>& B,
    thread MMATile<Ctype, M, N, MMAFragC>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short m_serp = m; //(n % 2) ? (M - 1 - m) : m;
      short n_serp = (m % 2) ? (N - 1 - n) : n;

      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMAFragD::mma(
            D.frag_at(m_serp, n_serp),
            A.frag_at(m_serp, k),
            B.frag_at(k, n_serp),
            C.frag_at(m_serp, n_serp));
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
  // MMAFrag size
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = kFragSize * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = kFragSize * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / TM_stride;
  // Warp tile size along N
  STEEL_CONST short TN = BN / TN_stride;

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

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/attn/params.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/attn/params.h"
// Copyright © 2024 Apple Inc.


///////////////////////////////////////////////////////////////////////////////
// Attn param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct AttnParams {
  int B; ///< Batch Size
  int H; ///< Heads
  int D; ///< Head Dim

  int qL; ///< Query Sequence Length
  int kL; ///< Key Sequence Length

  int gqa_factor; ///< Group Query factor
  float scale; ///< Attention scale

  int NQ; ///< Number of query blocks
  int NK; ///< Number of key/value blocks

  int NQ_aligned; ///< Number of full query blocks
  int NK_aligned; ///< Number of full key/value blocks

  int qL_rem; ///< Remainder in last query block
  int kL_rem; ///< Remainder in last key/value block
  int qL_off; ///< Offset in query sequence start

  int64_t Q_strides[3]; ///< Query  strides (B, H, L, D = 1)
  int64_t K_strides[3]; ///< Key    strides (B, H, L, D = 1)
  int64_t V_strides[3]; ///< Value  strides (B, H, L, D = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct AttnMaskParams {
  int64_t M_strides[3]; ///< Mask  strides (B, H, qL, kL = 1)
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/gemm/params.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/params.h"
// Copyright © 2024 Apple Inc.


///////////////////////////////////////////////////////////////////////////////
// GEMM param classes
///////////////////////////////////////////////////////////////////////////////

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

  const int swizzle_log;
  const int gemm_k_iterations_aligned;
};

struct GEMMAddMMParams {
  const int ldc;
  const int fdc;

  const int64_t batch_stride_c;

  const float alpha;
  const float beta;
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/attn/attn.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/attn/attn.h"
// Copyright © 2024 Apple Inc.



using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel class
///////////////////////////////////////////////////////////////////////////////

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
  STEEL_CONST short tgp_padding_a = 16 / sizeof(T);
  STEEL_CONST short tgp_padding_b = 16 / sizeof(T);
  STEEL_CONST short tgp_mem_size_a =
      transpose_a ? BK * (BM + tgp_padding_a) : BM * (BK + tgp_padding_a);
  STEEL_CONST short tgp_mem_size_b =
      transpose_b ? BN * (BK + tgp_padding_b) : BK * (BN + tgp_padding_b);
  STEEL_CONST short tgp_mem_size = tgp_mem_size_a + tgp_mem_size_b;

  STEEL_CONST short tgp_size = WM * WN * 32;

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

  /* Main kernel function */
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
    // Appease the compiler
    (void)l;

    short2 tile_dims_A = transpose_a ? short2(tgp_bm, BK) : short2(BK, tgp_bm);

    short2 tile_dims_B = transpose_b ? short2(BK, tgp_bn) : short2(tgp_bn, BK);

    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Load elements into threadgroup
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

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);

      // Prepare for next iteration
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

  /* Main kernel function */
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
    // Pacifying compiler
    (void)lid;

    const int tid_y = ((tid.y) << params->swizzle_log) +
        ((tid.x) & ((1 << params->swizzle_log) - 1));
    const int tid_x = (tid.x) >> params->swizzle_log;

    if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Find block in A, B, C
    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const size_t c_row_long = size_t(c_row);
    const size_t c_col_long = size_t(c_col);

    A += transpose_a ? c_row_long : c_row_long * params->lda;
    B += transpose_b ? c_col_long * params->ldb : c_col_long;
    D += c_row_long * params->ldd + c_col_long;

    // Prepare threadgroup loading operations
    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    int gemm_k_iterations = params->gemm_k_iterations_aligned;

    ///////////////////////////////////////////////////////////////////////////////
    // MNK aligned loop
    if (MN_aligned) {
      for (int k = 0; k < gemm_k_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Load elements into threadgroup
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }

      threadgroup_barrier(mem_flags::mem_none);

      // Loop tail
      if (!K_aligned) {
        int lbk = params->K - params->gemm_k_iterations_aligned * BK;
        short2 tile_dims_A = transpose_a ? short2(BM, lbk) : short2(lbk, BM);
        short2 tile_dims_B = transpose_b ? short2(lbk, BN) : short2(BN, lbk);

        loader_a.load_safe(tile_dims_A);
        loader_b.load_safe(tile_dims_B);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(As, Bs);
      }

      // Store results to device memory
      mma_op.store_result(D, params->ldd);
      return;

    }
    ///////////////////////////////////////////////////////////////////////////////
    // MN unaligned loop
    else { // Loop over K - unaligned case
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

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
// Copyright © 2024-25 Apple Inc.


using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  TransformScale<T> ts(static_cast<T>(params->scale * M_LOG2E_F));

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks apply scale
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }
  loader_q.apply_inplace_op(ts);

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);
  }

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Mask out length sequence
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= (kb_lim - ((BQ + BK - 1) / BK) - int(!align_K))) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += M_LOG2E_F * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (!align_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
  }

  // Normalize output
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  O += (tm + sm) * params->O_strides[2] + sn;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
