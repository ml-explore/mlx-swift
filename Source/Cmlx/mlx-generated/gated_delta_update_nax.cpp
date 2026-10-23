namespace mlx::core::metal {

const char* gated_delta_update_nax() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/gated_delta_update_nax.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/defines.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/defines.h"
// Copyright © 2024 Apple Inc.


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

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

  METAL_FUNC constexpr operator value_type() const thread noexcept {
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

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    using res_t = metal::remove_addrspace_t<decltype(res)>; \
    return integral_constant<res_t, res>{};                 \
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
    src += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

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
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
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
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
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
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((r < lx) && ((c + j) < ly)) {
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
    dst += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

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
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
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
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lx && (c + j) < ly) {
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

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    // Create Matmul descriptor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = Am0[i];
      ct_a[kElemsPerFrag + i] = Am1[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = B[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cm0[i];
      ct_c[kElemsPerFrag + i] = Cm1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cm0[i] = ct_c[i];
      Cm1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <
    typename T,
    short kTileRows_,
    short kTileCols_,
    class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;

  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;

  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j)
      thread {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    return val_frags[i * kTileCols + j];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) const thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  METAL_FUNC thread elem_type* elems() thread {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const thread {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(
      thread metal::vec<T, kRowsPerThread>& vals) const thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(
      thread metal::vec<T, kRowsPerThread>& vals) thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_rows(const device U* src, const int ld, const short n_rows) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load_safe(
      const device U* src,
      const int ld,
      const short2 src_tile_dims) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_safe(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_rows(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_safe(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_slice(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
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
  constexpr short TM = CTile::kTileRows;
  static_assert(TMa == TM, "MXU tile matmul: M dimensions do not match");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TN = CTile::kTileCols;
  static_assert(TNb == TN, "MXU tile matmul: N dimensions do not match");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TK, "MXU tile matmul: K dimensions do not match");

  constexpr auto ta = metal::bool_constant<transpose_a>{};
  constexpr auto tb = metal::bool_constant<transpose_b>{};

  if constexpr (TN == 1 && TM % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm += 2) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm + 1, nn),
              A.frag_at(mm, kk, ta),
              A.frag_at(mm + 1, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  } else if constexpr (TN % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; ++mm) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              B.frag_at(kk, nn + 1, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/gated_delta_nax_ops.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/gated_delta_nax_ops.h"
// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>


using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

typedef mlx::steel::NAXTile<float, 1, 1> _M16x16;
typedef mlx::steel::NAXTile<float, 1, 2> _M16x32;

// NAX MACROS
#define AT_NAX(TILE, IDX) TILE.elems()[IDX]

// out = a - b
template <typename T>
METAL_FUNC T operator-(const thread T& a, const thread T& b) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) - b.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC T operator-(const thread T& a) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = -a.frag_at(0, f);
  }
  return out;
}

// out = a + b
template <typename T>
METAL_FUNC T operator+(const thread T& a, const thread T& b) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) + b.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC T operator+(const thread T& a, float addend) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) + addend;
  }
  return out;
}

#define TRIL_NAX(TILE0, TILE1)                                                 \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) = (_c.x >= _c.y) ? 0.f : AT_NAX(TILE1, _i);            \
    }                                                                          \
  }

// out = a * b, elementwise. Not a matmul.
template <typename T>
METAL_FUNC T operator*(const thread T& a, const thread T& b) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) * b.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC T operator*(const thread T& a, float s) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) * s;
  }
  return out;
}

template <typename T>
METAL_FUNC T fmadd(const thread T& a, const thread T& b, const thread T& c) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) * b.frag_at(0, f) + c.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC _M16x16 reduce(const thread T& a) {
  _M16x16 out;
  out.frag_at(0, 0) = a.frag_at(0, 0);
  STEEL_PRAGMA_UNROLL
  for (short f = 1; f < T::kNumFrags; f++) {
    out.frag_at(0, 0) += a.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC T scale_rows(const thread T& tile, const thread float* s) {
  typename T::frag_type sv;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < T::kElemsPerFrag; i++) {
    sv[i] = s[i >> 2];
  }

  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = tile.frag_at(0, f) * sv;
  }
  return out;
}

template <typename T>
METAL_FUNC void row_sum(thread float* dst, const thread T& tile) {
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < T::kElemsPerFrag; i++) {
      dst[i >> 2] += tile.frag_at(0, f)[i];
    }
  }
}

// Scales row i by DEC2[group(i)] == exp(gamma_{C-1} - gamma_i).
#define SCALE2_P(TILE0, DEC2)                                       \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (DEC2)[_w >> 2];                         \
    }                                                               \
  }

#define SUB_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) - AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define ADD_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) + AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define FMA_NAX(TILE0, S, TILE1, TILE2)                                     \
  {                                                                         \
    STEEL_PRAGMA_UNROLL                                                     \
    for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) { \
      (TILE0)[_i] = (S) * (TILE1)[_i] + (TILE2)[_i];                        \
    }                                                                       \
  }

#define SCALE_NAX(TILE0, S)                                         \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) *= (S);                                     \
    }                                                               \
  }

#define SCALE_ROW_NAX(TILE0, S)                                            \
  {                                                                        \
    STEEL_PRAGMA_UNROLL                                                    \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {        \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;        \
      AT_NAX(TILE0, _i) *=                                                 \
          metal::fast::exp((S)[mlx::steel::BaseNAXFrag::get_coord(_w).y]); \
    }                                                                      \
  }

#define TRIL_NAX(TILE0, TILE1)                                                 \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) = (_c.x >= _c.y) ? 0.f : AT_NAX(TILE1, _i);            \
    }                                                                          \
  }

#define SCALE_BETA_NAX(TILE0, BETA2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (BETA2)[_w >> 2];                        \
    }                                                               \
  }

#define SCALE2_NAX(TILE0, GAMMA)                                              \
  {                                                                           \
    STEEL_PRAGMA_UNROLL                                                       \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {           \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;           \
      const short _fm = mlx::steel::BaseNAXFrag::get_coord(_w).y;             \
      AT_NAX(TILE0, _i) *= metal::fast::exp((GAMMA)[(C) - 1] - (GAMMA)[_fm]); \
    }                                                                         \
  }

#define SCALE_TRI_NAX(TILE0, GAMMA)                                            \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *= (_c.x > _c.y)                                       \
          ? 0.f                                                                \
          : metal::fast::exp((GAMMA)[_c.y] - (GAMMA)[_c.x]);                   \
    }                                                                          \
  }

#define SCALE_TRIEQ_NAX1(TILE0, BETA)                                          \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *= (_c.x >= _c.y) ? 0.f : (BETA)[_i >> 2];             \
    }                                                                          \
  }

namespace mlx {
namespace steel {
template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>
METAL_FUNC static constexpr void mma(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A0,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A1,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B1,
    metal::bool_constant<transpose_b>) {
  // M=16, N=16, K=32: A and B each two K-fragments, single 16x16 C.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 16, 32, transpose_a, transpose_b, true, Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());
  using ct_b_t = decltype(gemm_op.template get_right_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c =
      gemm_op
          .template get_destination_cooperative_tensor<ct_a_t, ct_b_t, CType>();

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = A0[i];
    ct_a[BaseNAXFrag::kElemsPerFrag + i] = A1[i];
    ct_b[i] = B0[i];
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = B1[i];
    ct_c[i] = C[i];
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    C[i] = ct_c[i];
  }
}

template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a,
    bool transpose_b,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>
METAL_FUNC static constexpr void mma(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B,
    metal::bool_constant<transpose_b>) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, transpose_a, transpose_b, true, Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());
  using ct_b_t = decltype(gemm_op.template get_right_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c =
      gemm_op
          .template get_destination_cooperative_tensor<ct_a_t, ct_b_t, CType>();

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = A[i];
    ct_b[i] = B[i];
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
    ct_c[i] = C[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    C[i] = ct_c[i];
  }
}

template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>
METAL_FUNC static constexpr void mman(
    thread BaseNAXFrag::dtype_frag_t<CType>& Cn0,
    thread BaseNAXFrag::dtype_frag_t<CType>& Cn1,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& Bn0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& Bn1,
    metal::bool_constant<transpose_b>) {
  // M=16, N=32, K=16: single A (K=16), B and C two N-fragments each.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, transpose_a, transpose_b, true, Mode);

  // Create matmul op
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());
  using ct_b_t = decltype(gemm_op.template get_right_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());

  // Create matmul operands in registers
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();

  // Create matmul output in register
  auto ct_c =
      gemm_op
          .template get_destination_cooperative_tensor<ct_a_t, ct_b_t, CType>();

  // Load A in to left operand registers
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = A[i];
    ct_b[i] = Bn0[i];
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = Bn1[i];
    ct_c[i] = Cn0[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = Cn1[i];
  }

  // Do matmul
  gemm_op.run(ct_a, ct_b, ct_c);

  // Copy out results
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[BaseNAXFrag::kElemsPerFrag + i];
  }
}

} // namespace steel
} // namespace mlx

#define MM16x16x16(C, CO, A, TA, AO, B, TB, BO)              \
  mlx::steel::mma<                                           \
      typename decltype(C)::elem_type,                       \
      typename decltype(A)::elem_type,                       \
      typename decltype(B)::elem_type,                       \
      TA,                                                    \
      TB,                                                    \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply>( \
      C.frag_at(0, (CO)),                                    \
      A.frag_at(0, (AO)),                                    \
      metal::bool_constant<TA>{},                            \
      B.frag_at(0, (BO)),                                    \
      metal::bool_constant<TB>{});

#define MMA16x16x16(C, CO, A, TA, AO, B, TB, BO)                        \
  mlx::steel::mma<                                                      \
      typename decltype(C)::elem_type,                                  \
      typename decltype(A)::elem_type,                                  \
      typename decltype(B)::elem_type,                                  \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>( \
      C.frag_at(0, (CO)),                                               \
      A.frag_at(0, (AO)),                                               \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      metal::bool_constant<TB>{});

#define MMA16x16x32(C, CO, A, TA, AO, B, TB, BO)                        \
  mlx::steel::mma<                                                      \
      typename decltype(C)::elem_type,                                  \
      typename decltype(A)::elem_type,                                  \
      typename decltype(B)::elem_type,                                  \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>( \
      C.frag_at(0, (CO)),                                               \
      A.frag_at(0, (AO)),                                               \
      A.frag_at(0, (AO) + 1),                                           \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      B.frag_at(0, (BO) + 1),                                           \
      metal::bool_constant<TB>{});

#define MM16x32x16(C, CO, A, TA, AO, B, TB, BO)              \
  mlx::steel::mman<                                          \
      typename decltype(C)::elem_type,                       \
      typename decltype(A)::elem_type,                       \
      typename decltype(B)::elem_type,                       \
      TA,                                                    \
      TB,                                                    \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply>( \
      C.frag_at(0, (CO)),                                    \
      C.frag_at(0, (CO) + 1),                                \
      A.frag_at(0, (AO)),                                    \
      metal::bool_constant<TA>{},                            \
      B.frag_at(0, (BO)),                                    \
      B.frag_at(0, (BO) + 1),                                \
      metal::bool_constant<TB>{});

#define MMA16x32x16(C, CO, A, TA, AO, B, TB, BO)                        \
  mlx::steel::mman<                                                     \
      typename decltype(C)::elem_type,                                  \
      typename decltype(A)::elem_type,                                  \
      typename decltype(B)::elem_type,                                  \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>( \
      C.frag_at(0, (CO)),                                               \
      C.frag_at(0, (CO) + 1),                                           \
      A.frag_at(0, (AO)),                                               \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      B.frag_at(0, (BO) + 1),                                           \
      metal::bool_constant<TB>{});

// Row sum of an elementwise product, reduced over the fragment's column axis.
// DST[0] is the fm row group, DST[1] the fm + kElemRowsJump group.
#define ROWSUM_NAX(DST, TILE0, TILE1)                               \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      (DST)[_w >> 2] += AT_NAX(TILE0, _i) * AT_NAX(TILE1, _i);      \
    }                                                               \
  }

// already generic
#define SCALE_BETA_NAX_O(DST, TILE, BETA2)                          \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE)::kElemsPerTile; _i++) {  \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(DST, _i) = AT_NAX(TILE, _i) * (BETA2)[_w >> 2];        \
    }                                                               \
  }

#define ROWSUM1_NAX(DST, TILE0)                                     \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      (DST)[_w >> 2] += AT_NAX(TILE0, _i);                          \
    }                                                               \
  }

#define MUL_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) * AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define MULA_NAX(TILE0, TILE1, TILE2)                               \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) += AT_NAX(TILE1, _i) * AT_NAX(TILE2, _i);   \
    }                                                               \
  }

#define MULS_NAX(TILE0, TILE1, TILE2)                               \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) -= AT_NAX(TILE1, _i) * AT_NAX(TILE2, _i);   \
    }                                                               \
  }

#define NSCALE_ROW_NAX(TILE0, S)                                           \
  {                                                                        \
    STEEL_PRAGMA_UNROLL                                                    \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {        \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;        \
      AT_NAX(TILE0, _i) = -AT_NAX(TILE0, _i) *                             \
          metal::fast::exp((S)[mlx::steel::BaseNAXFrag::get_coord(_w).y]); \
    }                                                                      \
  }

#define SCALE_ROW_P(TILE0, ROW2)                                    \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (ROW2)[_w >> 2];                         \
    }                                                               \
  }

// As SCALE_ROW_P but also negates, in one pass.
#define NSCALE_ROW_P(TILE0, ROW2)                                   \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) = -AT_NAX(TILE0, _i) * (ROW2)[_w >> 2];     \
    }                                                               \
  }

#define SCALE2_P(TILE0, DEC2)                                       \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (DEC2)[_w >> 2];                         \
    }                                                               \
  }

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/gated_delta_update_nax.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/gated_delta_update_nax.h"
// Copyright © 2026 Apple Inc.


using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

///////////////////////////////////////////////////////////////////////////////
// Function constants
///////////////////////////////////////////////////////////////////////////////

constant bool save_state_cache [[function_constant(200)]];

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C, int Ckpt>
[[kernel]] void gated_delta_fused_nax(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device float* state_in [[buffer(3)]],
    const device InT* g [[buffer(4)]],
    const device InT* beta [[buffer(5)]],
    device InT* y [[buffer(6)]],
    device float* state_out [[buffer(7)]],
    constant int& T [[buffer(8)]],
    device float* state_cache [[buffer(9)]], // [B, Hv, n_ckpt, Dv, Dk]
    device float* chunk_mats [[buffer(10)]], // [B, Hv, n_chunks, 3, 16, 16]
    device float* chunk_delta [[buffer(11)]], // [B, Hv, n_chunks, C, Dv]
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dv_idx = thread_position_in_grid.y * 16;
  const short sg_id = thread_position_in_threadgroup.y; // 0..3

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  // set up pointers
  // g: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  // v, y: [B, T, Hv, Dv]
  y += b_idx * T * Hv * Dv + hv_idx * Dv;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto beta_ = beta + b_idx * T * Hv;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  const int n_chunks = (T + C - 1) / C;
  const int n_ckpt = (n_chunks + Ckpt - 1) / Ckpt;
  auto c_state = state_cache + (n * n_ckpt * Dv + dv_idx) * Dk;

  auto o_chunk_mats = chunk_mats + n * n_chunks * 3 * 256;
  auto o_chunk_delta = chunk_delta + n * n_chunks * C * Dv + dv_idx;

  threadgroup float gamma_all[C * 4];
  threadgroup float* gamma = gamma_all + sg_id * C;

  float beta_fm[2];

  mlx::steel::NAXTile<float, 1, Dk / 16> S_tile;
  S_tile.load(i_state, Dk);

  mlx::steel::NAXTile<float, 1, 2> K_tile, Q_tile;
  mlx::steel::NAXTile<float, 1, Dk / 16> W_tile; // panel

  mlx::steel::NAXTile<float, 1, 1> V_tile;
  mlx::steel::NAXTile<float, 1, 1> U_tile;
  mlx::steel::NAXTile<float, 1, 1> WS_tile;
  mlx::steel::NAXTile<float, 1, 1> delta_tile;
  mlx::steel::NAXTile<float, 1, 1> tmp_tile;
  mlx::steel::NAXTile<float, 1, 1> QKt_tile;
  mlx::steel::NAXTile<float, 1, 1> out_tile;
  mlx::steel::NAXTile<float, 1, 1> Tinv_tile, P;

  mlx::steel::NAXTile<float, 1, 1> KKtK_tile, KKt_tile;

  mlx::steel::NAXTile<float, 1, 1> I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    const short _fn = _c.x;
    const short _fm = _c.y;
    AT_NAX(I_tile, _i) = (_fn == _fm) ? 1.0f : 0.0f;
  }
  mlx::steel::NAXTile<float, 1, 1> TMP_tile;

  // Which chunk process_chunk is on, so it knows whether this is a segment
  // boundary.
  int chunk_idx = 0;

  auto process_chunk = [&](const short valid_rows,
                           auto bounded_tag) __attribute__((always_inline)) {
    constexpr bool B = decltype(bounded_tag)::value;

    auto load_seq = [&](thread auto& tile, auto src, int ld) {
      if constexpr (B) {
        tile.load_rows(src, ld, valid_rows);
      } else {
        tile.load(src, ld);
      }
    };

    // Checkpoint the state entering this chunk before anything mutates it, on
    // segment boundaries only.
    if (save_state_cache && (chunk_idx % Ckpt) == 0) {
      S_tile.store(c_state, Dk);
      c_state += Dv * Dk;
    }

    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(
              metal::max(
                  float(g_[thread_index_in_simdgroup * Hv + hv_idx]), 1e-6))
        : 0.0f;

    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val);
    }

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    KKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      MMA16x16x32(KKt_tile, 0, K_tile, false, 0, K_tile, true, 0);
    }

    // KKt_raw is identical for every Dv tile of this chunk; only the first
    // tile needs to write it out for the vjp kernel to reuse.
    if (save_state_cache && dv_idx == 0) {
      KKt_tile.store(o_chunk_mats, 16);
    }

    KKtK_tile = KKt_tile;

    SCALE_TRIEQ_NAX1(KKtK_tile, beta_fm);
    SUB_NAX(Tinv_tile, I_tile, KKtK_tile);
    STEEL_PRAGMA_UNROLL
    for (int step = 0; step < 15; step++) {
      MM16x16x16(TMP_tile, 0, KKtK_tile, false, 0, Tinv_tile, false, 0);
      SUB_NAX(Tinv_tile, I_tile, TMP_tile);
    }

    // Same as KKt_raw: cache the pre-decay inverse before it is overwritten
    // below by the gamma-scaled (TUinv) version.
    if (save_state_cache && dv_idx == 0) {
      Tinv_tile.store(o_chunk_mats + 256, 16);
    }

    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < Dk / 16; nn += 2) {
      load_seq(K_tile, k_ + nn * 16, Dk * Hk);
      SCALE_BETA_NAX(K_tile, beta_fm);
      MM16x32x16(W_tile, nn, Tinv_tile, false, 0, K_tile, false, 0);
    }
    SCALE_ROW_NAX(W_tile, gamma)

    SCALE_TRI_NAX(Tinv_tile, gamma)
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    MM16x16x16(U_tile, 0, Tinv_tile, false, 0, V_tile, false, 0)

        WS_tile.clear();
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < Dk / 16; kk += 2) {
      MMA16x16x32(WS_tile, 0, W_tile, false, kk, S_tile, true, kk)
    }

    SUB_NAX(delta_tile, U_tile, WS_tile)

    QKt_tile.clear();
    tmp_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Hk * Dk);
      MMA16x16x32(QKt_tile, 0, Q_tile, false, 0, K_tile, true, 0);
      SCALE_ROW_NAX(Q_tile, gamma);
      MMA16x16x32(tmp_tile, 0, Q_tile, false, 0, S_tile, true, kk / 16);
    }

    if (save_state_cache) {
      if (dv_idx == 0) {
        QKt_tile.store(o_chunk_mats + 512, 16);
      }
      // Every Dv tile stores its own slice.
      delta_tile.store(o_chunk_delta, Dv);
      o_chunk_mats += 3 * 256;
      o_chunk_delta += C * Dv;
    }

    // Output path runs unconditionally: the forward serves both inference and
    // the training save, so y and state_out are always needed.
    SCALE_TRI_NAX(QKt_tile, gamma)
    out_tile = tmp_tile;
    MMA16x16x16(out_tile, 0, QKt_tile, false, 0, delta_tile, false, 0);

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(out_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      const short _fn = _c.x;
      const short _fm = _c.y;
      if (_fm < valid_rows) {
        y[_fm * Hv * Dv + dv_idx + _fn] =
            static_cast<InT>(AT_NAX(out_tile, _i));
      }
    }

    SCALE_NAX(S_tile, metal::fast::exp(gamma[C - 1]));

    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Hk * Dk);
      SCALE2_NAX(K_tile, gamma);
      MMA16x32x16(S_tile, kk / 16, delta_tile, true, 0, K_tile, false, 0);
    }

    chunk_idx++;
  };

  int t = 0;
  for (; t + C <= T; t += C) {
    process_chunk(C, metal::false_type{});
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }
  if (t < T) {
    process_chunk(short(T - t), metal::true_type{});
  }

  S_tile.store(o_state, Dk);
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
