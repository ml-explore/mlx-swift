namespace mlx::core::metal {

const char* gemm_nax() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/steel/gemm/gemm_nax.h

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
// Contents from "/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.3.48.0.5BBI2H/Metal.xctoolchain/usr/metal/32023/lib/clang/32023.850/include/metal/__exec/units.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.3.48.0.5BBI2H/Metal.xctoolchain/usr/metal/32023/lib/clang/32023.850/include/metal/__exec/units.h"

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
// Contents from "mlx/backend/metal/kernels/steel/gemm/nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/nax.h"
// Copyright © 2025 Apple Inc.


#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// NAX Steel with new tiles
///////////////////////////////////////////////////////////////////////////////

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  METAL_FUNC static short2 get_coord(short idx) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4 + idx % 4;
    return short2{fn, fm};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      if (r < lim_x) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }

      } else {
        dst = dtype_frag_t<T>(0);
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lim_x && (c + j) < lim_y) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      if (r < lim_x) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump + sc.y;
      const auto c = off_y + sc.x;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lim_x && (c + j) < lim_y) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_slice(
      const thread dtype_frag_t<T>& src,
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

    const short2 sc = get_coord();

    const_for_loop<0, kElemRows, 1>([&](auto idx_row) {
      const auto r = off_x + idx_row * Int<kElemRowsJump>{};
      if (r >= stop_x - sc.y || r < start_x - sc.y) {
        return;
      }

      const_for_loop<0, kElemCols, 1>([&](auto idx_col) {
        const auto c = off_y + idx_col;
        if (c >= stop_y - sc.x || c < start_y - sc.x) {
          return;
        }

        const auto src_idx = idx_row * Int<kElemCols>{} + idx_col;
        dst[(r + sc.y) * str_x + (c + sc.x) * str_y] =
            static_cast<U>(src[src_idx]);
      });
    });
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));

      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
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
    short kRows_,
    short kCols_,
    typename NAXFrag_t = BaseNAXFrag>
struct NAXSubTile {
  STEEL_CONST short kRows = kRows_;
  STEEL_CONST short kCols = kCols_;

  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kSubTileRows = kRows / kFragRows;
  STEEL_CONST short kSubTileCols = kCols / kFragCols;

  STEEL_CONST short kNumFrags = kSubTileRows * kSubTileCols;
  STEEL_CONST short kElemsPerSubTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST int kRowsPerThread = kSubTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST int kColsPerThread = kSubTileCols * NAXFrag_t::kElemCols;

  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  using frag_type = typename NAXFrag_t::template dtype_frag_t<T>;

  frag_type val_frags[kNumFrags];

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kSubTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kSubTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() {
    return val_frags[i * kSubTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const {
    return val_frags[i * kSubTileCols + j];
  }

  METAL_FUNC thread T* elems() {
    return reinterpret_cast<thread T*>(val_frags);
  }

  METAL_FUNC const thread T* elems() const {
    return reinterpret_cast<const thread T*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vals[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vals[i * kFragThrRows]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC constexpr void load(
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::load(
            frag_at(i, j),
            src,
            str_x,
            str_y,
            off_x + i * kFragRows,
            off_y + j * kFragCols);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC constexpr void store(
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::store(
            frag_at(i, j),
            dst,
            str_x,
            str_y,
            off_x + i * kFragRows,
            off_y + j * kFragCols);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC constexpr void load_rows(
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::load_rows(
            frag_at(i, j),
            src,
            str_x,
            str_y,
            lim_x,
            off_x + (i * kFragRows),
            off_y + (j * kFragCols));
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC constexpr void load_safe(
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::load_safe(
            frag_at(i, j),
            src,
            str_x,
            str_y,
            lim_x,
            lim_y,
            off_x + (i * kFragRows),
            off_y + (j * kFragCols));
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC constexpr void store_safe(
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::store_safe(
            frag_at(i, j),
            dst,
            str_x,
            str_y,
            lim_x,
            lim_y,
            off_x + (i * kFragRows),
            off_y + (j * kFragCols));
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC constexpr void store_rows(
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::store_safe(
            frag_at(i, j),
            dst,
            str_x,
            str_y,
            lim_x,
            off_x + (i * kFragRows),
            off_y + (j * kFragCols));
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
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC constexpr void store_slice(
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) const {
    const_for_loop<0, kSubTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kSubTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_slice(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            str_x,
            str_y,
            start_x,
            stop_x,
            start_y,
            stop_y,
            off_x + idx_row * Int<kFragRows>{},
            off_y + idx_col * Int<kFragCols>{});
      });
    });
  }
};

template <
    short RC,
    short CC,
    short RA,
    short CA,
    short RB,
    short CB,
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a,
    bool transpose_b,
    typename NAXFrag_t = BaseNAXFrag>
METAL_FUNC void subtile_matmad_nax(
    thread NAXSubTile<CType, RC, CC, NAXFrag_t>& C,
    thread NAXSubTile<AType, RA, CA, NAXFrag_t>& A,
    metal::bool_constant<transpose_a>,
    thread NAXSubTile<BType, RB, CB, NAXFrag_t>& B,
    metal::bool_constant<transpose_b>) {
  // Static checks
  constexpr short FMa = transpose_a ? CA : RA;
  constexpr short FMc = RC;
  static_assert(FMa == FMc, "NAX matmul: M dimensions do not match");

  constexpr short FNb = transpose_b ? RB : CB;
  constexpr short FNc = CC;
  static_assert(FNb == FNc, "NAX matmul: N dimensions do not match");

  constexpr short FKa = transpose_a ? RA : CA;
  constexpr short FKb = transpose_b ? CB : RB;
  static_assert(FKa == FKb, "NAX matmul: N dimensions do not match");

  constexpr short FM = FMc;
  constexpr short FN = FNc;
  constexpr short FK = FKa;

  constexpr int TM = FM / 16;
  constexpr int TN = FN / 16;
  constexpr int TK = FK / 16;

  // Create Matmul descriptor
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      FM,
      FN,
      FK,
      transpose_a,
      transpose_b,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  // Create matmul op
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  // Create matmul operands in registers
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();

  // Create matmul output in register
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  // Load A in to left operand registers
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < TK; kk++) {
      const short fi = transpose_a ? kk : mm;
      const short fj = transpose_a ? mm : kk;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 8; i++) {
        ct_a[(TK * mm + kk) * 8 + i] = A.frag_at(fi, fj)[i];
      }
    }
  }

  // Load B into right operand registers
  STEEL_PRAGMA_UNROLL
  for (short nn = 0; nn < TN; nn++) {
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < TK; kk++) {
      const short fi = transpose_b ? nn : kk;
      const short fj = transpose_b ? kk : nn;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 8; i++) {
        ct_b[(TN * kk + nn) * 8 + i] = B.frag_at(fi, fj)[i];
      }
    }
  }

  // Load C into output registers (op handles accumulation)
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < ct_c.get_capacity(); i++) {
    ct_c[i] = C.elems()[i];
  }

  // Do matmul
  gemm_op.run(ct_a, ct_b, ct_c);

  // Copy out results
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < ct_c.get_capacity(); i++) {
    C.elems()[i] = ct_c[i];
  }
}

template <typename T, short kTileRows_, short kTileCols_, class NAXSubTile_>
struct NAXTile {
  using NAXSubTile_t = NAXSubTile_;
  using elem_type = T;
  STEEL_CONST short kSubTileRows = NAXSubTile_t::kRows;
  STEEL_CONST short kSubTileCols = NAXSubTile_t::kCols;
  STEEL_CONST short kElemsPerSubTile = NAXSubTile_t::kElemsPerSubTile;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kSubTileRows;
  STEEL_CONST short kCols = kTileCols * kSubTileCols;

  STEEL_CONST short kSubTiles = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kSubTiles * kElemsPerSubTile;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXSubTile_t::kRowsPerThread;
  STEEL_CONST short kColsPerThread = kTileCols * NAXSubTile_t::kColsPerThread;

  STEEL_CONST short kSubTileThrRows = NAXSubTile_t::kRowsPerThread;
  STEEL_CONST short kSubTileThrCols = NAXSubTile_t::kColsPerThread;

  NAXSubTile_t val_subtiles[kSubTiles];

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTiles; ++i) {
      val_subtiles[i].clear();
    }
  }

  METAL_FUNC constexpr thread NAXSubTile_t& subtile_at(
      const short i,
      const short j) {
    return val_subtiles[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread NAXSubTile_t& subtile_at(
      const short i,
      const short j) const {
    return val_subtiles[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread NAXSubTile_t& subtile_at() const {
    return val_subtiles[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_subtiles[0].elems());
  }

  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_subtiles[0].elems());
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const {
    auto sub_rows = (thread metal::vec<T, kSubTileThrRows>*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).template row_reduce<Op>(sub_rows[i]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    auto sub_rows = (thread metal::vec<T, kSubTileThrRows>*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).template row_bin_op<Op>(sub_rows[i]);
      }
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).load(
            src,
            Int<str_x>{},
            Int<str_y>{},
            i * kSubTileRows,
            j * kSubTileCols);
      }
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).store(
            dst,
            Int<str_x>{},
            Int<str_y>{},
            i * kSubTileRows,
            j * kSubTileCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).load(
            &src[(i * kSubTileRows * ld + j * kSubTileCols)], ld, Int<1>{});
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).store(
            &dst[(i * kSubTileRows * ld + j * kSubTileCols)], ld, Int<1>{});
      }
    }
  }

  template <typename U>
  METAL_FUNC void
  load_rows(const device U* src, const int ld, const short n_rows) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).load_rows(
            &src[(i * kSubTileRows) * ld + (j * kSubTileCols)],
            ld,
            Int<1>{},
            n_rows - i * kSubTileRows);
      }
    }
  }

  template <typename U>
  METAL_FUNC void
  load_safe(const device U* src, const int ld, const short2 src_tile_dims) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).load_safe(
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            i * kSubTileRows,
            j * kSubTileCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).store_rows(
            &dst[(i * kSubTileRows) * ld + (j * kSubTileCols)],
            ld,
            Int<1>{},
            n_rows - i * kSubTileRows);
      }
    }
  }

  template <typename U>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        subtile_at(i, j).store_safe(
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            i * kSubTileRows,
            j * kSubTileCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        subtile_at<idx_row.value, idx_col.value>().store_slice(
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            idx_row * Int<kSubTileRows>{},
            idx_col * Int<kSubTileCols>{});
      });
    });
  }
};

template <
    class CTile,
    class ATile,
    class BTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad_nax(
    thread CTile& C,
    thread ATile& A,
    metal::bool_constant<transpose_a>,
    thread BTile& B,
    metal::bool_constant<transpose_b>) {
  // Static checks
  constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
  constexpr short TMc = CTile::kTileRows;
  static_assert(TMa == TMc, "NAX tile matmul: M dimensions do not match");

  constexpr short FMa = transpose_a ? ATile::kSubTileCols : ATile::kSubTileRows;
  constexpr short FMc = CTile::kSubTileRows;
  static_assert(FMa == FMc, "NAX subtile matmul: M dimensions do not match");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TNc = CTile::kTileCols;
  static_assert(TNb == TNc, "NAX tile matmul: N dimensions do not match");

  constexpr short FNb = transpose_b ? BTile::kSubTileRows : BTile::kSubTileCols;
  constexpr short FNc = CTile::kSubTileCols;
  static_assert(FNb == FNc, "NAX subtile matmul: N dimensions do not match");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TKb = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TKb, "NAX tile matmul: K dimensions do not match");

  constexpr short FKa = transpose_a ? ATile::kSubTileRows : ATile::kSubTileCols;
  constexpr short FKb = transpose_b ? BTile::kSubTileCols : BTile::kSubTileRows;
  static_assert(FKa == FKb, "NAX subtile matmul: K dimensions do not match");

  constexpr short TM = TMc;
  constexpr short TN = TNc;
  constexpr short TK = TKa;

  // Do matmul here
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TN; ++j) {
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < TK; ++k) {
        const short ra = transpose_a ? k : i;
        const short ca = transpose_a ? i : k;
        const short rb = transpose_b ? j : k;
        const short cb = transpose_b ? k : j;

        subtile_matmad_nax(
            C.subtile_at(i, j),
            A.subtile_at(ra, ca),
            metal::bool_constant<transpose_a>{},
            B.subtile_at(rb, cb),
            metal::bool_constant<transpose_b>{});
      }
    }
  }
}

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
// Contents from "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
// Copyright © 2025 Apple Inc.



using namespace metal;

namespace mlx::steel {

template <
    typename T,
    short SM,
    short SN,
    short SK,
    short BK,
    bool transpose_a,
    bool transpose_b,
    bool kAlignedM,
    bool kAlignedN,
    bool kAlignedK,
    short UM,
    short UN,
    short UK,
    typename AccumType = float>
auto gemm_loop(
    const device T* A,
    const device T* B,
    int lda,
    int ldb,
    int K,
    int gemm_k_iterations_aligned,
    const short sgp_sm,
    const short sgp_sn) {
  constexpr short TM = SM / UM;
  constexpr short TN = SN / UN;
  constexpr short TK = SK / UK;

  constexpr int RA = transpose_a ? TK : TM;
  constexpr int CA = transpose_a ? TM : TK;

  constexpr int RB = transpose_b ? TN : TK;
  constexpr int CB = transpose_b ? TK : TN;

  using DSubTile = NAXSubTile<AccumType, UM, UN>;
  using ASubTile =
      NAXSubTile<T, (transpose_a ? UK : UM), (transpose_a ? UM : UK)>;
  using BSubTile =
      NAXSubTile<T, (transpose_b ? UN : UK), (transpose_b ? UK : UN)>;

  NAXTile<AccumType, TM, TN, DSubTile> Dtile;
  Dtile.clear();

  int gemm_k_iterations_ = gemm_k_iterations_aligned;

  STEEL_PRAGMA_NO_UNROLL
  for (int kk0 = 0; kk0 < gemm_k_iterations_; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, RA, CA, ASubTile> Atile;
      NAXTile<T, RB, CB, BSubTile> Btile;
      const int k = kk1;

      volatile int compiler_barrier;

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      if constexpr (kAlignedM) {
        Atile.load(A + A_offset, lda);
      } else {
        const short rmax = transpose_a ? SK : sgp_sm;
        const short cmax = transpose_a ? sgp_sm : SK;
        Atile.load_safe(A + A_offset, lda, short2(cmax, rmax));
      }

      if constexpr (kAlignedN) {
        Btile.load(B + B_offset, ldb);
      } else {
        const short rmax = transpose_b ? sgp_sn : SK;
        const short cmax = transpose_b ? SK : sgp_sn;
        Btile.load_safe(B + B_offset, ldb, short2(cmax, rmax));
      }

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    A += transpose_a ? (BK * lda) : BK;
    B += transpose_b ? BK : (BK * ldb);
  }

  if constexpr (!kAlignedK) {
    simdgroup_barrier(mem_flags::mem_none);

    const short rem_bk = K - gemm_k_iterations_ * BK;

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < rem_bk; kk1 += SK) {
      NAXTile<T, 1, 1, ASubTile> Atile;
      NAXTile<T, 1, 1, BSubTile> Btile;

      STEEL_PRAGMA_UNROLL
      for (int mm = 0; mm < TM; mm++) {
        STEEL_PRAGMA_UNROLL
        for (int nn = 0; nn < TN; nn++) {
          STEEL_PRAGMA_UNROLL
          for (int kk = 0; kk < TK; kk++) {
            const int m = mm * UM;
            const int n = nn * UN;
            const int k = kk1 + kk * UK;
            const short psk = max(0, rem_bk - k);

            const int A_offset = transpose_a ? (m + k * lda) : (m * lda + k);
            const int B_offset = transpose_b ? (k + n * ldb) : (k * ldb + n);

            {
              const short psm = kAlignedM ? SM : max(0, sgp_sm - m);
              const short rmax = transpose_a ? psk : psm;
              const short cmax = transpose_a ? psm : psk;
              Atile.load_safe(A + A_offset, lda, short2(cmax, rmax));
            }

            {
              const short psn = kAlignedN ? SN : max(0, sgp_sn - n);
              const short rmax = transpose_b ? psn : psk;
              const short cmax = transpose_b ? psk : psn;
              Btile.load_safe(B + B_offset, ldb, short2(cmax, rmax));
            }

            subtile_matmad_nax(
                Dtile.subtile_at(mm, nn),
                Atile.subtile_at(0, 0),
                metal::bool_constant<transpose_a>{},
                Btile.subtile_at(0, 0),
                metal::bool_constant<transpose_b>{});
          }
        }
      }
    }
  }

  return Dtile;
}

} // namespace mlx::steel

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
