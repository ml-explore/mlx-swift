namespace mlx::core::metal {

const char* steel_attention_nax() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/defines.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/defines.h"
// Copyright © 2024 Apple Inc.


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

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
// Contents from "mlx/backend/metal/kernels/steel/attn/nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/attn/nax.h"
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
    const short2 sc = short2{0, 0}; // get_coord();
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
    const short2 sc = short2{0, 0}; // get_coord();
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
    const short2 sc = short2{0, 0}; // get_coord();
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

    const short2 sc = short2{0, 0}; // get_coord();
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

    const short2 sc = short2{0, 0}; // get_coord();
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

    const short2 sc = short2{0, 0}; // get_coord();
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

    const short2 sc = short2{0, 0}; // get_coord();

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
    typename NAXFrag_ = BaseNAXFrag>
struct NAXSubTile {
  using NAXFrag_t = NAXFrag_;
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
    thread T* vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    thread T* vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kSubTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kSubTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
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

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      FM,
      FN,
      FK,
      transpose_a,
      transpose_b,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

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

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < ct_c.get_capacity(); i++) {
    ct_c[i] = C.elems()[i];
  }

  gemm_op.run(ct_a, ct_b, ct_c);

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
            &src[(i * kSubTileRows) * ld + (j * kSubTileCols)], ld, Int<1>{});
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
            &dst[(i * kSubTileRows) * ld + (j * kSubTileCols)], ld, Int<1>{});
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
// Contents from "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
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
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(
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
  (void)simd_lane_id;

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

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short UQ = 16;
  constexpr short UD = 32;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * UQ) && BQ % (kNWarps * UQ) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * UQ);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / UD;

  static_assert(TQ == 1, "Check TQ");

  using OSubTile = NAXSubTile<AccumType, UQ, UD>;
  NAXTile<AccumType, TQ, TD, OSubTile> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = OSubTile::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = UQ * TQ * simd_group_id;

  Q += (tm + sm) * int(params->Q_strides[2]) + sn;
  K += sm * int(params->K_strides[2]) + sn;
  V += sm * int(params->V_strides[2]) + sn;

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Otile)::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

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

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - (tm + sm);
  const short lim_rows_k = params->kL_rem - sm;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    constexpr short UDs = 16;
    constexpr short UKs = 32;

    constexpr short TDs = BD / UDs;
    constexpr short TKs = BK / UKs;

    using SSubTile = NAXSubTile<AccumType, UQ, UKs>;
    using QSubTile = NAXSubTile<T, UQ, UDs>;
    using KSubTile = NAXSubTile<T, UKs, UDs>;

    NAXTile<AccumType, TQ, TKs, SSubTile> Stile;

    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TKs; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDs; id++) {
          NAXTile<T, 1, 1, QSubTile> Qtile;
          NAXTile<T, 1, 1, KSubTile> Ktile;

          const int Q_load_off = iq * UQ * int(params->Q_strides[2]) + id * UDs;
          const int K_load_off =
              ik * UKs * int(params->K_strides[2]) + id * UDs;

          if (!align_Q && is_last_q) {
            // Qtile.load_rows(
            //     Q + Q_load_off,
            //     int(params->Q_strides[2]),
            //     lim_rows_q - iq * UQ);
            Qtile.load_safe(
                Q + Q_load_off,
                int(params->Q_strides[2]),
                short2(BD, lim_rows_q - iq * UQ));
          } else {
            Qtile.load(Q + Q_load_off, int(params->Q_strides[2]));
          }

          if (!align_K && is_last_k) {
            // Ktile.load_rows(
            //     K + K_load_off,
            //     int(params->K_strides[2]),
            //     lim_rows_k - ik * UKs);
            Ktile.load_safe(
                K + K_load_off,
                int(params->K_strides[2]),
                short2(BD, lim_rows_k - ik * UKs));
          } else {
            Ktile.load(K + K_load_off, int(params->K_strides[2]));
          }

          subtile_matmad_nax(
              Stile.subtile_at(iq, ik),
              Qtile.subtile_at(0, 0),
              metal::false_type{},
              Ktile.subtile_at(0, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Scale and Retile S
    constexpr short UK = 16;
    constexpr short TK = BK / UK;
    using PSubTile = NAXSubTile<AccumType, UQ, UK>;

    NAXTile<AccumType, TQ, TK, PSubTile> Ptile;

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Ptile.elems()[ii] = Stile.elems()[ii];
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = sn + ik * UK;

          thread auto& fg = Ptile.subtile_at(iq, ik).frag_at(0, 0);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < PSubTile::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < PSubTile::kFragThrCols; jj++) {
              const auto loc = ii * PSubTile::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) >= params->kL_rem) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= (kb_lim - ((BQ + BK - 1) / BK) - int(!align_K))) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short row_pos = base_row + iq * UQ;
          const short col_pos = base_col + ik * UK;

          thread auto& fg = Ptile.subtile_at(iq, ik).frag_at(0, 0);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < PSubTile::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < PSubTile::kFragThrCols; jj++) {
              const auto r = row_pos + ii * PSubTile::kFragRowsJump + sm;
              const auto c = col_pos + jj + sn;
              const auto loc = ii * PSubTile::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using MSubTile = NAXSubTile<melem_t, UQ, UK>;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short row_pos = base_row + iq * UQ + sm;
          const short col_pos = base_col + ik * UK + sn;

          MSubTile mfrag;
          mfrag.load_safe(
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          thread auto& fg = Ptile.subtile_at(iq, ik).frag_at(0, 0);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < MSubTile::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag.elems()[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag.elems()[jj]);
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Ptile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Ptile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Ptile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        if constexpr (BD == 128) {
          if (id == 2) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          using VSubTile = NAXSubTile<T, UK, UD>;
          NAXTile<T, 1, 1, VSubTile> Vtile;

          const int V_load_off = ik * UK * int(params->V_strides[2]) + id * UD;

          if (!align_K && is_last_k) {
            // Vtile.load_rows(
            //     V + V_load_off,
            //     int(params->V_strides[2]),
            //     lim_rows_k - ik * UK);
            Vtile.load_safe(
                V + V_load_off,
                int(params->V_strides[2]),
                short2(BD, lim_rows_k - ik * UK));
          } else {
            Vtile.load(V + V_load_off, int(params->V_strides[2]));
          }

          subtile_matmad_nax(
              Otile.subtile_at(iq, id),
              Ptile.subtile_at(iq, ik),
              metal::bool_constant<false>{},
              Vtile.subtile_at(0, 0),
              metal::bool_constant<false>{});
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += (tm + sm) * int(params->O_strides[2]) + sn;

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    // Otile.store_rows(O, params->O_strides[2], lim_rows_q);
    Otile.store_safe(O, params->O_strides[2], short2(BD, lim_rows_q));
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
