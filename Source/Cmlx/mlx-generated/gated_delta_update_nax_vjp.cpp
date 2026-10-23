namespace mlx::core::metal {

const char* gated_delta_update_nax_vjp() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/gated_delta_update_nax_vjp.h

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
// Contents from "mlx/backend/metal/kernels/atomic.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/atomic.h"
// Copyright © 2023 Apple Inc.


#include <metal_atomic>
#include <metal_stdlib>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Atomic utils
///////////////////////////////////////////////////////////////////////////////

#pragma METAL internals : enable
template <typename T>
constexpr constant bool is_metal_atomic = _disjunction<
    is_same<T, int>,
    is_same<T, uint>,
    is_same<T, ulong>,
    is_same<T, float>>::value;

#pragma METAL internals : disable

template <typename T, typename = void>
struct mlx_atomic {
  atomic<uint> val;
};

template <typename T>
struct mlx_atomic<T, enable_if_t<is_metal_atomic<T>>> {
  atomic<T> val;
};

///////////////////////////////////////////////////////////////////////////////
// Native metal atomics
///////////////////////////////////////////////////////////////////////////////

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC T
mlx_atomic_load_explicit(device mlx_atomic<T>* object, size_t offset) {
  return atomic_load_explicit(&(object[offset].val), memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void
mlx_atomic_store_explicit(device mlx_atomic<T>* object, T val, size_t offset) {
  atomic_store_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_and_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_and_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_or_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_or_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_min_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_min_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_max_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_max_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_add_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  atomic_fetch_add_explicit(&(object[offset].val), val, memory_order_relaxed);
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_mul_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  T expected = mlx_atomic_load_explicit(object, offset);
  while (!mlx_atomic_compare_exchange_weak_explicit(
      object, &expected, val * expected, offset)) {
    // Workaround: Metal's atomic_compare_exchange_weak_explicit<float> does
    // not perform bitwise comparison as required by the C++ atomics spec.
    // The compiler lowers the success check to `fcmp fast ueq` under
    // no-nans-fp-math, which evaluates to false when either operand is NaN -
    // even when the bit patterns are identical. With NaN in memory the CAS
    // can never succeed, so the loop spins. Bail out instead: memory is
    // already NaN and that is the correct reduction result regardless of
    // this thread's update.
    if constexpr (metal::is_floating_point_v<T>) {
      if (isnan(expected)) {
        break;
      }
    }
  }
}

template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
METAL_FUNC bool mlx_atomic_compare_exchange_weak_explicit(
    device mlx_atomic<T>* object,
    thread T* expected,
    T val,
    size_t offset) {
  return atomic_compare_exchange_weak_explicit(
      &(object[offset].val),
      expected,
      val,
      memory_order_relaxed,
      memory_order_relaxed);
}

// Specialization for float since it does not atomic_fetch_min_explicit
template <>
METAL_FUNC void mlx_atomic_fetch_min_explicit<float>(
    device mlx_atomic<float>* object,
    float val,
    size_t offset) {
  float expected = mlx_atomic_load_explicit(object, offset);
  while (val < expected) {
    if (mlx_atomic_compare_exchange_weak_explicit(
            object, &expected, val, offset)) {
      return;
    }
  }
}

// Specialization for float since it does not atomic_fetch_max_explicit
template <>
METAL_FUNC void mlx_atomic_fetch_max_explicit<float>(
    device mlx_atomic<float>* object,
    float val,
    size_t offset) {
  float expected = mlx_atomic_load_explicit(object, offset);
  while (val > expected) {
    if (mlx_atomic_compare_exchange_weak_explicit(
            object, &expected, val, offset)) {
      return;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Custom atomics
///////////////////////////////////////////////////////////////////////////////

namespace {

template <typename T>
constexpr constant uint packing_size = sizeof(uint) / sizeof(T);

template <typename T>
union uint_or_packed {
  T val[packing_size<T>];
  uint bits;
};

template <typename T, typename Op>
struct mlx_atomic_update_helper {
  uint operator()(uint_or_packed<T> init, T update, size_t elem_offset) thread {
    Op op;
    init.val[elem_offset] = op(update, init.val[elem_offset]);
    return init.bits;
  }
};

template <typename T, typename Op>
METAL_FUNC void mlx_atomic_update_and_store(
    device mlx_atomic<T>* object,
    T update,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;

  mlx_atomic_update_helper<T, Op> helper;
  uint_or_packed<T> expected;
  expected.bits =
      atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);

  while (Op::condition(update, expected.val[elem_offset]) &&
         !mlx_atomic_compare_exchange_weak_explicit(
             object,
             &(expected.bits),
             helper(expected, update, elem_offset),
             pack_offset)) {
  }
}

template <typename T>
struct __None {
  static bool condition(T a, T b) {
#pragma unused(a)
#pragma unused(b)
    return true;
  }

  T operator()(T a, T b) thread {
#pragma unused(b)
    return a;
  }
};

template <typename T>
struct __Add {
  static bool condition(T a, T b) {
#pragma unused(a)
#pragma unused(b)
    return true;
  }

  T operator()(T a, T b) thread {
    return a + b;
  }
};

template <typename T>
struct __Mul {
  static bool condition(T a, T b) {
#pragma unused(a)
    return b != 0;
  }

  T operator()(T a, T b) thread {
    return a * b;
  }
};

template <typename T>
struct __Max {
  static bool condition(T a, T b) {
    return a > b;
  }

  T operator()(T a, T b) thread {
    return max(a, b);
  }
};

template <typename T>
struct __Min {
  static bool condition(T a, T b) {
    return a < b;
  }

  T operator()(T a, T b) thread {
    return min(a, b);
  }
};

} // namespace

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC T
mlx_atomic_load_explicit(device mlx_atomic<T>* object, size_t offset) {
  size_t pack_offset = offset / sizeof(T);
  size_t elem_offset = offset % sizeof(T);
  uint_or_packed<T> packed_val;
  packed_val.bits =
      atomic_load_explicit(&(object[pack_offset].val), memory_order_relaxed);
  return packed_val.val[elem_offset];
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void
mlx_atomic_store_explicit(device mlx_atomic<T>* object, T val, size_t offset) {
  mlx_atomic_update_and_store<T, __None<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_and_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;
  uint_or_packed<T> identity;
  identity.bits = __UINT32_MAX__;
  identity.val[elem_offset] = val;

  atomic_fetch_and_explicit(
      &(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_or_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  size_t pack_offset = offset / packing_size<T>;
  size_t elem_offset = offset % packing_size<T>;
  uint_or_packed<T> identity;
  identity.bits = 0;
  identity.val[elem_offset] = val;

  atomic_fetch_or_explicit(
      &(object[pack_offset].val), identity.bits, memory_order_relaxed);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_min_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Min<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_max_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Max<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_add_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Add<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC void mlx_atomic_fetch_mul_explicit(
    device mlx_atomic<T>* object,
    T val,
    size_t offset) {
  mlx_atomic_update_and_store<T, __Mul<T>>(object, val, offset);
}

template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
METAL_FUNC bool mlx_atomic_compare_exchange_weak_explicit(
    device mlx_atomic<T>* object,
    thread uint* expected,
    uint val,
    size_t offset) {
  return atomic_compare_exchange_weak_explicit(
      &(object[offset].val),
      expected,
      val,
      memory_order_relaxed,
      memory_order_relaxed);
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/gated_delta_update_nax_vjp.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/gated_delta_update_nax_vjp.h"
// Copyright © 2026 Apple Inc.


#include <metal_atomic>

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

template <int kNSG, bool kGQA, typename TileT>
METAL_FUNC void reduce_tile_tg(
    thread TileT& acc,
    threadgroup float* scratch,
    device mlx_atomic<float>* out,
    int row_stride,
    int col_off,
    short valid_rows,
    const short sg_id,
    const ushort simd_lane_id) {
  constexpr short kE = TileT::kElemsPerTile;

  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < kE; _i++) {
    scratch[(sg_id * 32 + simd_lane_id) * kE + _i] = AT_NAX(acc, _i);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg_id == 0) {
    STEEL_PRAGMA_UNROLL
    for (short _s = 1; _s < kNSG; _s++) {
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < kE; _i++) {
        AT_NAX(acc, _i) += scratch[(_s * 32 + simd_lane_id) * kE + _i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Simdgroup 0 alone holds the total.
  if (sg_id != 0) {
    return;
  }

  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < kE; _i++) {
    const short _f = _i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
    const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_w); // {fn, fm}
    if (_c.y < valid_rows) {
      const int idx = _c.y * row_stride + col_off + _c.x + _f * 16;
      if (kGQA) {
        mlx_atomic_fetch_add_explicit(out, AT_NAX(acc, _i), idx);
      } else {
        mlx_atomic_store_explicit(out, AT_NAX(acc, _i), idx);
      }
    }
  }
}

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C, int Ckpt>
[[kernel]] void gated_delta_vjp_fused_nax(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B, Hv, n_ckpt, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<float>* dq [[buffer(9)]],
    device mlx_atomic<float>* dk [[buffer(10)]],
    device float* dv [[buffer(11)]],
    device float* dg [[buffer(12)]],
    device float* db [[buffer(13)]],
    device float* dh [[buffer(14)]],
    const device float* chunk_mats [[buffer(15)]],
    const device float* chunk_delta [[buffer(16)]], // [B, Hv, n_chunks, C, Dv]
    device float* seg_states [[buffer(17)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  using _M16xDk = mlx::steel::NAXTile<float, 1, Dk / 16>;

  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dv_idx = thread_position_in_grid.y * 16;
  const short sg_id = thread_position_in_threadgroup.y;

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
  const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;

  const int n_chunks = (T + C - 1) / C;
  // Ceiling: the floor form gives 0 when n_chunks < Ckpt, which would leave the
  // allocation empty and fault on the first read.
  const int n_ckpt = (n_chunks + Ckpt - 1) / Ckpt;
  const short tail = short(T - (n_chunks - 1) * C);

  // Bases, indexed by chunk rather than walked. With two nested loops the
  // incremental decrements are no longer expressible.
  auto q_base = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_base = k + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto dq_base = dq + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto dk_base = dk + b_idx * T * Hk * Dk + hk_idx * Dk;

  auto v_base = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto dv_base = dv + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto co_base = cot_o + b_idx * T * Hv * Dv + hv_idx * Dv;

  auto g_base = g + b_idx * T * Hv;
  auto beta_base = beta + b_idx * T * Hv;
  auto dg_base = dg + b_idx * T * Hv;
  auto db_base = db + b_idx * T * Hv;

  auto cm_base = chunk_mats + n * n_chunks * 3 * 256;
  auto cd_base = chunk_delta + n * n_chunks * C * Dv + dv_idx;
  auto ck_base = state_cache + (n * n_ckpt * Dv + dv_idx) * Dk;
  // Each simdgroup owns a distinct dv slice, so these regions are disjoint and
  // the replay needs no cross-simdgroup synchronisation.
  auto seg_base = seg_states + (n * Ckpt * Dv + dv_idx) * Dk;

  // Per-chunk pointers, assigned from the bases before each process_chunk call.
  auto q_ = q_base;
  auto k_ = k_base;
  auto dq_ = dq_base;
  auto dk_ = dk_base;
  auto v_ = v_base;
  auto dv_ = dv_base;
  auto co_ = co_base;
  auto g_ = g_base;
  auto beta_ = beta_base;
  auto dg_ = dg_base;
  auto db_ = db_base;
  auto i_chunk_mats = cm_base;
  auto i_chunk_delta = cd_base;
  auto i_seg = seg_base;

  auto i_cot_h = cot_h + (n * Dv + dv_idx) * Dk;
  auto o_dh = dh + (n * Dv + dv_idx) * Dk;

  // One threadgroup per (b, hv) covers every Dv slice, so the Dv reduction
  // never leaves threadgroup memory.
  constexpr int kNSG = Dv / 16;
  // Under GQA several threadgroups still share an hk, so dq/dk keep an atomic
  // for that axis -- but with kNSG writers folded away first.
  constexpr bool kGQA = (Hv != Hk);

  threadgroup float gamma_all[C * kNSG];
  threadgroup float* gamma = gamma_all + sg_id * C;

  threadgroup float red_scratch[(kNSG / 1) * 32 * 16];
  threadgroup float db_stage[kNSG * C];
  threadgroup float dg_stage[kNSG * C];

  float beta_fm[2];

  // Carried state gradient, dL/dS. Same [dv, dk] orientation as S_tile.
  _M16xDk dS_tile;
  dS_tile.load(i_cot_h, Dk);

  // Forward state at chunk entry, taken from the replayed segment.
  _M16xDk S_tile;

  _M16x32 K_tile, Q_tile;
  _M16x16 V_tile;
  _M16x16 delta_tile;
  _M16x16 QKt_tile, QKt_raw;
  _M16x16 KKt_tile;
  _M16x16 TWinv_tile, TUinv_tile;
  _M16x16 TMP_tile;

  _M16x16 dout_tile;
  _M16x16 ddelta_tile;
  _M16x16 dTU_tile, dTW_tile;
  _M16x16 dTinv_tile;
  _M16x16 dA_tile, G_tile;
  _M16x16 tri_decay;

  _M16x16 I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    AT_NAX(I_tile, _i) = (_c.x == _c.y) ? 1.0f : 0.0f;
  }

  _M16x16 Ones_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(Ones_tile)::kElemsPerFrag; _i++) {
    AT_NAX(Ones_tile, _i) = 1.0f;
  }

  _M16x16 tril_mask;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(tril_mask)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
    AT_NAX(tril_mask, _i) = (_c.x >= _c.y) ? 0.0f : 1.0f;
  }

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

    // From the replayed segment rather than a per-chunk checkpoint.
    S_tile.load(i_seg, Dk);

    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(
              metal::max(g_[thread_index_in_simdgroup * Hv + hv_idx], 1e-6))
        : 0.0f;

    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float gamma_last = gamma[C - 1];
    const float gamma_last_exp = metal::fast::exp(gamma_last);

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    // Per-lane decay factors. gamma[j] == gamma[valid_rows-1] for
    // j >= valid_rows (the prefix sum of zeros), so the tail chunk needs no
    // special case here.
    const float row_exp[2] = {
        metal::fast::exp(gamma[fm]), metal::fast::exp(gamma[fm1])};
    const float dec_exp[2] = {
        metal::fast::exp(gamma_last - gamma[fm]),
        metal::fast::exp(gamma_last - gamma[fm1])};

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(tri_decay)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      AT_NAX(tri_decay, _i) =
          (_c.x > _c.y) ? 0.f : metal::fast::exp(gamma[_c.y] - gamma[_c.x]);
    }

    KKt_tile.load(i_chunk_mats, 16);
    TWinv_tile.load(i_chunk_mats + 256, 16);
    QKt_raw.load(i_chunk_mats + 512, 16);

    QKt_tile = QKt_raw * tri_decay;
    TUinv_tile = TWinv_tile * tri_decay;

    delta_tile.load(i_chunk_delta, Dv);

    // dgamma accumulators:
    //   dgam_row  : per-row
    //   dgam_pair : row-minus-col
    //   dgam_last : the gamma_{C-1}
    _M16x16 dgam_row, dgam_pair;
    dgam_row.clear();
    dgam_pair.clear();
    float dgam_last = 0.0f;
    float dbeta_acc[2] = {0.0f, 0.0f};

    load_seq(dout_tile, co_ + dv_idx, Hv * Dv);

    // dODT = (dO @ delta.T * D)
    _M16x16 dODT;
    MM16x16x16(dODT, 0, dout_tile, false, 0, delta_tile, true, 0);
    dODT = dODT * tri_decay;

    // cannot fuse this because ddelta is used by the next loop
    MM16x16x16(ddelta_tile, 0, QKt_tile, true, 0, dout_tile, false, 0);
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      K_tile = scale_rows(K_tile, dec_exp);
      MMA16x16x32(ddelta_tile, 0, K_tile, false, 0, dS_tile, true, kk / 16);
    }

    // ddelta = M^T @ dO + K_dec @ dS^T
    // dq = gamma * (dO @ S) + dODT @ K
    dTW_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dW_raw;
      _M16x32 dq_acc;
      load_seq(K_tile, k_ + kk, Dk * Hk);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      dW_raw = scale_rows(-dW_raw, row_exp);
      _M16x32 Kb_tile = scale_rows(K_tile, beta_fm);
      MMA16x16x32(dTW_tile, 0, dW_raw, false, 0, Kb_tile, true, 0);

      MM16x32x16(dq_acc, 0, dout_tile, false, 0, S_tile, false, kk / 16);
      dq_acc = scale_rows(dq_acc, row_exp);
      MMA16x32x16(dq_acc, 0, dODT, false, 0, K_tile, false, 0);

      // dq reduces over Dv, which is entirely inside this threadgroup.
      reduce_tile_tg<kNSG, kGQA>(
          dq_acc,
          red_scratch,
          dq_,
          Hk * Dk,
          kk,
          valid_rows,
          sg_id,
          simd_lane_id);
    }

    // dv = B(Tu.T @ ddelta).  Indexed by dv_idx, so no reduction.
    _M16x16 dVb_tile;
    MM16x16x16(dVb_tile, 0, TUinv_tile, true, 0, ddelta_tile, false, 0);
    _M16x16 TUddelta = dVb_tile; // pre-beta, reused by dbeta
    dVb_tile = scale_rows(dVb_tile, beta_fm);

    dVb_tile.store_rows(dv_ + dv_idx, Hv * Dv, valid_rows);

    // dT_U = ddelta @ (beta * V).T
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    V_tile = scale_rows(V_tile, beta_fm);
    MM16x16x16(dTU_tile, 0, ddelta_tile, false, 0, V_tile, true, 0);
    dTU_tile = dTU_tile * tri_decay;

    dTinv_tile = dTW_tile + dTU_tile;

    // dA = -T.T @ dTinv @ T.T
    MM16x16x16(TMP_tile, 0, TWinv_tile, true, 0, dTinv_tile, false, 0);
    MM16x16x16(dA_tile, 0, TMP_tile, false, 0, TWinv_tile, true, 0);
    dA_tile = dA_tile * -1.0f;

    // G = tril_(dA) * beta, and GGt = G + G.T
    G_tile = scale_rows(dA_tile, beta_fm) * tril_mask;
    _M16x16 GGt_tile = G_tile;
    MMA16x16x16(GGt_tile, 0, I_tile, true, 0, G_tile, true, 0);

    // dgamma pair sites
    _M16x16 P_tile = QKt_raw * dODT;
    _M16x16 R_tile = TWinv_tile * dTU_tile;
    dgam_pair = P_tile + R_tile;

    _M16x32 KdKb; // rowsum(K * dK_b) source, needed for dbeta
    KdKb.clear();
    _M16x32 dgam_row32;
    dgam_row32.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dk_acc, dW_raw, t1;

      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      {
        _M16x32 QgdOS;
        MM16x32x16(QgdOS, 0, dout_tile, false, 0, S_tile, false, kk / 16);
        dgam_row32 = dgam_row32 + scale_rows(QgdOS, row_exp) * Q_tile;
      }

      MM16x32x16(dk_acc, 0, dODT, true, 0, Q_tile, false, 0);
      MMA16x32x16(dk_acc, 0, GGt_tile, false, 0, K_tile, false, 0);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      dW_raw = scale_rows(-dW_raw, row_exp);

      {
        _M16x32 dKb_tile;
        MM16x32x16(dKb_tile, 0, TWinv_tile, true, 0, dW_raw, false, 0);
        KdKb = KdKb + dKb_tile * K_tile;
        dk_acc = dk_acc + scale_rows(dKb_tile, beta_fm);
      }

      MM16x32x16(t1, 0, delta_tile, false, 0, dS_tile, false, kk / 16);
      t1 = scale_rows(t1, dec_exp);
      dk_acc = dk_acc + t1;

      {
        const _M16x32 KdKC = K_tile * t1;
        dgam_row32 = dgam_row32 - KdKC;
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < decltype(KdKC)::kElemsPerTile; _i++) {
          dgam_last += AT_NAX(KdKC, _i);
        }
      }

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
        dgam_last += gamma_last_exp *
            (S_tile.frag_at(0, kk / 16)[_i] * dS_tile.frag_at(0, kk / 16)[_i] +
             S_tile.frag_at(0, kk / 16 + 1)[_i] *
                 dS_tile.frag_at(0, kk / 16 + 1)[_i]);
      }

      {
        _M16x32 TbK;
        K_tile = scale_rows(K_tile, beta_fm);
        MM16x32x16(TbK, 0, TWinv_tile, false, 0, K_tile, false, 0);
        dgam_row32 = fmadd(TbK, dW_raw, dgam_row32);
      }

      reduce_tile_tg<kNSG, kGQA>(
          dk_acc,
          red_scratch,
          dk_,
          Hk * Dk,
          kk,
          valid_rows,
          sg_id,
          simd_lane_id);
    }

    // dbeta = rowsum(V * dV_b) + rowsum(K * dK_b) + rowsum(tril_(dA) * KKt)
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);

    _M16x16 AKKt = dA_tile * KKt_tile;
    AKKt = AKKt * tril_mask;
    _M16x16 VdVb = fmadd(V_tile, TUddelta, AKKt);
    row_sum(dbeta_acc, VdVb);
    row_sum(dbeta_acc, KdKb);

    {
      threadgroup float* db_st = db_stage + sg_id * C;
      float _b0 = dbeta_acc[0];
      float _b1 = dbeta_acc[1];
      _b0 += simd_shuffle_xor(_b0, ushort(1));
      _b0 += simd_shuffle_xor(_b0, ushort(8));
      _b1 += simd_shuffle_xor(_b1, ushort(1));
      _b1 += simd_shuffle_xor(_b1, ushort(8));

      if (fm < valid_rows) {
        db_st[fm] = _b0;
      }
      if (fm1 < valid_rows) {
        db_st[fm1] = _b1;
      }
    }

    dgam_row = reduce(dgam_row32);

    float dgam_acc[2] = {0.0f, 0.0f};
    dgam_row = dgam_row + dgam_pair;
    row_sum(dgam_acc, dgam_row);

    _M16x16 cs_tile;
    MM16x16x16(cs_tile, 0, dgam_pair, true, 0, Ones_tile, false, 0);

    const float dgam_last_red = simd_sum(dgam_last);

    {
      threadgroup float* dg_st = dg_stage + sg_id * C;
      float _d0 = dgam_acc[0];
      float _d1 = dgam_acc[1];
      _d0 += simd_shuffle_xor(_d0, ushort(1));
      _d0 += simd_shuffle_xor(_d0, ushort(8));
      _d1 += simd_shuffle_xor(_d1, ushort(1));
      _d1 += simd_shuffle_xor(_d1, ushort(8));

      _d0 -= AT_NAX(cs_tile, 0);
      _d1 -= AT_NAX(cs_tile, mlx::steel::BaseNAXFrag::kElemCols);

      if (fm == valid_rows - 1) {
        _d0 += dgam_last_red;
      }
      if (fm < valid_rows) {
        dg_st[fm] = _d0;
      }
      if (fm1 == valid_rows - 1) {
        _d1 += dgam_last_red;
      }
      if (fm1 < valid_rows) {
        dg_st[fm1] = _d1;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0 && thread_index_in_simdgroup < (uint)valid_rows) {
      float b_sum = 0.0f;
      float g_sum = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (short _s = 0; _s < kNSG; _s++) {
        b_sum += db_stage[_s * C + thread_index_in_simdgroup];
        g_sum += dg_stage[_s * C + thread_index_in_simdgroup];
      }
      const int idx = thread_index_in_simdgroup * Hv + hv_idx;
      db_[idx] = b_sum;
      dg_[idx] = g_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dS update
    // dS = gamma_C * dS + dO.T @ (gamma * Q) - ddelta.T @ W
    SCALE_NAX(dS_tile, gamma_last_exp);
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 term, W_raw;
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      // dO.T @ (gamma * Q)  -> [dv, dk]
      Q_tile = scale_rows(Q_tile, row_exp);
      MM16x32x16(term, 0, dout_tile, true, 0, Q_tile, false, 0);

      // - ddelta.T @ W,  with W = gamma * (TWinv @ beta*K)
      K_tile = scale_rows(K_tile, beta_fm);
      MM16x32x16(W_raw, 0, TWinv_tile, false, 0, K_tile, false, 0);
      W_raw = scale_rows(-W_raw, row_exp);
      MMA16x32x16(term, 0, ddelta_tile, true, 0, W_raw, false, 0);

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(term)::kElemsPerFrag; _i++) {
        dS_tile.frag_at(0, kk / 16)[_i] += AT_NAX(term, _i);
        dS_tile.frag_at(0, kk / 16 + 1)[_i] +=
            AT_NAX(term, decltype(term)::kElemsPerFrag + _i);
      }
    }
  };

  // Walk segments in reverse. Each is replayed forward from its checkpoint to
  // recover the entry states, then walked backwards.
  // Backward over one segment: replay its entry states from the checkpoint,
  // then walk it in reverse. HasTail is a compile-time tag so valid_rows is the
  // constant C on every chunk of an interior segment, which folds away the
  // row guards in the dq/dk stores and the load_rows switch in the replay.
  auto process_segment = [&](const int seg,
                             auto tail_tag) __attribute__((always_inline)) {
    constexpr bool HasTail = decltype(tail_tag)::value;

    const int seg_start = seg * Ckpt;
    const int rem = n_chunks - seg_start;
    const int n_in_seg = (rem < Ckpt) ? rem : Ckpt;
    const int n_full = HasTail ? n_in_seg - 1 : n_in_seg;

    // delta is cached per chunk, so the replay is the state recurrence alone:
    //   S <- exp(gamma_C) S + K_dec^T delta
    // No T^-1, no W, no U -- roughly a tenth of a full chunk's work, which is
    // what makes trading Ckpt-fold memory for it worthwhile.
    //
    // Scoped so S is dead before process_chunk runs: it is a _M16xDk, and
    // having it live alongside S_tile and dS_tile would put three of them in
    // flight.
    {
      _M16xDk S;
      S.load(ck_base + seg * Dv * Dk, Dk);

      auto replay_one = [&](const int j,
                            const short valid_rows,
                            auto bounded_tag) __attribute__((always_inline)) {
        constexpr bool B = decltype(bounded_tag)::value;

        S.store(seg_base + j * Dv * Dk, Dk);

        const int c = seg_start + j;
        auto k_c = k_base + c * C * Hk * Dk;
        auto g_c = g_base + c * C * Hv;
        auto cd_c = cd_base + c * C * Dv;

        float gv = (thread_index_in_simdgroup < (uint)valid_rows)
            ? metal::fast::log(
                  metal::max(
                      g_c[thread_index_in_simdgroup * Hv + hv_idx], 1e-6))
            : 0.0f;
        auto gs = simd_prefix_inclusive_sum(gv);
        if (thread_index_in_simdgroup < C) {
          gamma[thread_index_in_simdgroup] = static_cast<float>(gs);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float g_last = gamma[C - 1];
        const float dec[2] = {
            metal::fast::exp(g_last - gamma[fm]),
            metal::fast::exp(g_last - gamma[fm1])};

        _M16x16 delta_tile;
        delta_tile.load(cd_c, Dv);

        S = S * metal::fast::exp(g_last);
        for (int kk = 0; kk < Dk; kk += 32) {
          _M16x32 Kd;
          if constexpr (B) {
            Kd.load_rows(k_c + kk, Dk * Hk, valid_rows);
          } else {
            Kd.load(k_c + kk, Dk * Hk);
          }
          Kd = scale_rows(Kd, dec);
          MMA16x32x16(S, kk / 16, delta_tile, true, 0, Kd, false, 0);
        }
      };

      for (int j = 0; j < n_full; j++) {
        replay_one(j, C, metal::false_type{});
      }
      if constexpr (HasTail) {
        replay_one(n_in_seg - 1, tail, metal::true_type{});
      }
    }

    auto set_chunk_ptrs = [&](const int j) __attribute__((always_inline)) {
      const int c = seg_start + j;
      q_ = q_base + c * C * Hk * Dk;
      k_ = k_base + c * C * Hk * Dk;
      dq_ = dq_base + c * C * Hk * Dk;
      dk_ = dk_base + c * C * Hk * Dk;
      v_ = v_base + c * C * Hv * Dv;
      dv_ = dv_base + c * C * Hv * Dv;
      co_ = co_base + c * C * Hv * Dv;
      g_ = g_base + c * C * Hv;
      beta_ = beta_base + c * C * Hv;
      dg_ = dg_base + c * C * Hv;
      db_ = db_base + c * C * Hv;
      i_chunk_mats = cm_base + c * 3 * 256;
      i_chunk_delta = cd_base + c * C * Dv;
      i_seg = seg_base + j * Dv * Dk;
    };

    if constexpr (HasTail) {
      set_chunk_ptrs(n_in_seg - 1);
      process_chunk(tail, metal::true_type{});
    }
    for (int j = n_full - 1; j >= 0; --j) {
      set_chunk_ptrs(j);
      process_chunk(C, metal::false_type{});
    }
  };

  // Only the last segment can hold a short chunk
  int seg = n_ckpt - 1;
  if (tail != C) {
    process_segment(seg, metal::true_type{});
    --seg;
  }
  for (; seg >= 0; --seg) {
    process_segment(seg, metal::false_type{});
  }

  dS_tile.store(o_dh, Dk);
}

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_vjp_fused_nax1(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B, Hv, n_chunks, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<float>* dq [[buffer(9)]],
    device mlx_atomic<float>* dk [[buffer(10)]],
    device float* dv [[buffer(11)]],
    device float* dg [[buffer(12)]],
    device float* db [[buffer(13)]],
    device float* dh [[buffer(14)]],
    // Cached, chunk-local forward intermediates produced by the forward-save
    // pass (see gated_delta_fused_nax): avoids redoing the K@K^T contraction,
    // the 15-step Neumann inversion, and the W/U/S contraction for delta.
    const device float* chunk_mats [[buffer(15)]],
    const device float* chunk_delta [[buffer(16)]], // [B, Hv, n_chunks, C, Dv]
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  using _M16xDk = mlx::steel::NAXTile<float, 1, Dk / 16>;

  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dv_idx = thread_position_in_grid.y * 16;
  const short sg_id = thread_position_in_threadgroup.y; // 0..3

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  const int n_chunks = (T + C - 1) / C;
  const int t_last = (n_chunks - 1) * C;

  // Pointers positioned at the final chunk
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto dq_ = dq + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto dk_ = dk + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;

  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;
  auto dv_ = dv + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;
  auto co_ = cot_o + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;

  auto g_ = g + b_idx * T * Hv + t_last * Hv;
  auto beta_ = beta + b_idx * T * Hv + t_last * Hv;
  auto dg_ = dg + b_idx * T * Hv + t_last * Hv;
  auto db_ = db + b_idx * T * Hv + t_last * Hv;

  auto c_state = state_cache + (n * n_chunks * Dv + dv_idx) * Dk +
      (n_chunks - 1) * Dv * Dk;

  // Cached forward intermediates, walked in reverse alongside c_state.
  auto i_chunk_mats =
      chunk_mats + n * n_chunks * 3 * 256 + (n_chunks - 1) * 3 * 256;
  auto i_chunk_delta =
      chunk_delta + n * n_chunks * C * Dv + dv_idx + (n_chunks - 1) * C * Dv;

  auto i_cot_h = cot_h + (n * Dv + dv_idx) * Dk;
  auto o_dh = dh + (n * Dv + dv_idx) * Dk;

  // One threadgroup per (b, hv) now covers every Dv slice, so the Dv reduction
  // never leaves threadgroup memory.
  constexpr int kNSG = Dv / 16;
  // Under GQA several threadgroups still share an hk, so dq/dk keep an atomic
  // for that axis -- but with kNSG writers folded away first.
  constexpr bool kGQA = (Hv != Hk);

  threadgroup float gamma_all[C * kNSG];
  threadgroup float* gamma = gamma_all + sg_id * C;

  // (kNSG/2) destinations x 32 lanes x kElemsPerTile(_M16x32) floats.
  threadgroup float red_scratch[(kNSG / 1) * 32 * 16];
  threadgroup float db_stage[kNSG * C];
  threadgroup float dg_stage[kNSG * C];

  float beta_fm[2];

  // Carried state gradient, dL/dS. Same [dv, dk] orientation as S_tile.
  _M16xDk dS_tile;
  dS_tile.load(i_cot_h, Dk);

  // Forward state at chunk entry, reloaded from the checkpoint each chunk
  _M16xDk S_tile;

  // Recomputed forward tiles
  _M16x32 K_tile, Q_tile;
  _M16xDk W_tile;
  _M16x16 K16_tile, Q16_tile;
  _M16x16 V_tile;
  _M16x16 U_tile;
  _M16x16 WS_tile;
  _M16x16 delta_tile;
  _M16x16 QKt_tile, QKt_raw;
  _M16x16 KKt_tile;
  _M16x16 TWinv_tile, TUinv_tile;
  _M16x16 TMP_tile;

  // Backward tiles
  _M16x16 dout_tile;
  _M16x16 ddelta_tile;
  _M16x16 dTU_tile, dTW_tile;
  _M16x16 dTinv_tile;
  _M16x16 dA_tile, G_tile;
  _M16x16 tri_decay;

  _M16x16 I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    AT_NAX(I_tile, _i) = (_c.x == _c.y) ? 1.0f : 0.0f;
  }

  _M16x16 Ones_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(Ones_tile)::kElemsPerFrag; _i++) {
    AT_NAX(Ones_tile, _i) = 1.0f;
  }

  _M16x16 tril_mask;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(tril_mask)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
    AT_NAX(tril_mask, _i) = (_c.x >= _c.y) ? 0.0f : 1.0f;
  }

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

    // reload the checkpoint
    S_tile.load(c_state, Dk);

    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(
              metal::max(g_[thread_index_in_simdgroup * Hv + hv_idx], 1e-6))
        : 0.0f;

    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float gamma_last = gamma[C - 1];
    const float gamma_last_exp = metal::fast::exp(gamma_last);

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    // Per-lane decay factors. gamma[j] == gamma[valid_rows-1] for
    // j >= valid_rows (the prefix sum of zeros), so the tail chunk needs no
    // special case here.
    const float row_exp[2] = {
        metal::fast::exp(gamma[fm]), metal::fast::exp(gamma[fm1])};
    const float dec_exp[2] = {
        metal::fast::exp(gamma_last - gamma[fm]),
        metal::fast::exp(gamma_last - gamma[fm1])};

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(tri_decay)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      AT_NAX(tri_decay, _i) =
          (_c.x > _c.y) ? 0.f : metal::fast::exp(gamma[_c.y] - gamma[_c.x]);
    }

    KKt_tile.load(i_chunk_mats, 16);
    TWinv_tile.load(i_chunk_mats + 256, 16);
    QKt_raw.load(i_chunk_mats + 512, 16);

    QKt_tile = QKt_raw * tri_decay;
    TUinv_tile = TWinv_tile * tri_decay;

    delta_tile.load(i_chunk_delta, Dv);

    // dgamma accumulators:
    //   dgam_row  : per-row
    //   dgam_pair : row-minus-col
    //   dgam_last : the gamma_{C-1}
    _M16x16 dgam_row, dgam_pair;
    dgam_row.clear();
    dgam_pair.clear();
    float dgam_last = 0.0f;
    float dbeta_acc[2] = {0.0f, 0.0f};

    load_seq(dout_tile, co_ + dv_idx, Hv * Dv);

    // dODT = (dO @ delta.T * D)
    _M16x16 dODT;
    MM16x16x16(dODT, 0, dout_tile, false, 0, delta_tile, true, 0);
    dODT = dODT * tri_decay;

    // cannot fuse this because ddelta is used by the next loop
    MM16x16x16(ddelta_tile, 0, QKt_tile, true, 0, dout_tile, false, 0);
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      K_tile = scale_rows(K_tile, dec_exp);
      MMA16x16x32(ddelta_tile, 0, K_tile, false, 0, dS_tile, true, kk / 16);
    }

    // ddelta = M^T @ dO + K_dec @ dS^T
    // dq = gamma * (dO @ S) + dODT @ K
    dTW_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dW_raw;
      _M16x32 dq_acc;
      load_seq(K_tile, k_ + kk, Dk * Hk);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      dW_raw = scale_rows(-dW_raw, row_exp);
      _M16x32 Kb_tile = scale_rows(K_tile, beta_fm);
      MMA16x16x32(dTW_tile, 0, dW_raw, false, 0, Kb_tile, true, 0);

      MM16x32x16(dq_acc, 0, dout_tile, false, 0, S_tile, false, kk / 16);
      dq_acc = scale_rows(dq_acc, row_exp);
      MMA16x32x16(dq_acc, 0, dODT, false, 0, K_tile, false, 0);

      // dq reduces over Dv, which is now entirely inside this threadgroup.
      reduce_tile_tg<kNSG, kGQA>(
          dq_acc,
          red_scratch,
          dq_,
          Hk * Dk,
          kk,
          valid_rows,
          sg_id,
          simd_lane_id);
    }

    // dv = B(Tu.T @ ddelta).  Indexed by dv_idx, so no reduction.
    _M16x16 dVb_tile;
    MM16x16x16(dVb_tile, 0, TUinv_tile, true, 0, ddelta_tile, false, 0);
    _M16x16 TUddelta = dVb_tile; // pre-beta, reused by dbeta
    dVb_tile = scale_rows(dVb_tile, beta_fm);

    dVb_tile.store_rows(dv_ + dv_idx, Hv * Dv, valid_rows);

    // dT_U = ddelta @ (beta * V).T
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    V_tile = scale_rows(V_tile, beta_fm);
    MM16x16x16(dTU_tile, 0, ddelta_tile, false, 0, V_tile, true, 0);
    dTU_tile = dTU_tile * tri_decay;

    dTinv_tile = dTW_tile + dTU_tile;

    // dA = -T.T @ dTinv @ T.T
    MM16x16x16(TMP_tile, 0, TWinv_tile, true, 0, dTinv_tile, false, 0);
    MM16x16x16(dA_tile, 0, TMP_tile, false, 0, TWinv_tile, true, 0);
    dA_tile = dA_tile * -1.0f;

    // G = tril_(dA) * beta, and GGt = G + G.T
    G_tile = scale_rows(dA_tile, beta_fm) * tril_mask;
    _M16x16 GGt_tile = G_tile;
    MMA16x16x16(GGt_tile, 0, I_tile, true, 0, G_tile, true, 0);

    // dgamma pair sites
    _M16x16 P_tile = QKt_raw * dODT;
    _M16x16 R_tile = TWinv_tile * dTU_tile;
    dgam_pair = P_tile + R_tile;

    _M16x32 KdKb; // rowsum(K * dK_b) source, needed for dbeta
    KdKb.clear();
    _M16x32 dgam_row32;
    dgam_row32.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dk_acc, dW_raw, t1;

      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      {
        _M16x32 QgdOS;
        MM16x32x16(QgdOS, 0, dout_tile, false, 0, S_tile, false, kk / 16);
        dgam_row32 = dgam_row32 + scale_rows(QgdOS, row_exp) * Q_tile;
      }

      MM16x32x16(dk_acc, 0, dODT, true, 0, Q_tile, false, 0);
      MMA16x32x16(dk_acc, 0, GGt_tile, false, 0, K_tile, false, 0);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      dW_raw = scale_rows(-dW_raw, row_exp);

      {
        _M16x32 dKb_tile;
        MM16x32x16(dKb_tile, 0, TWinv_tile, true, 0, dW_raw, false, 0);
        KdKb = KdKb + dKb_tile * K_tile; // KdKb_tmp folded away
        dk_acc = dk_acc + scale_rows(dKb_tile, beta_fm);
      }

      MM16x32x16(t1, 0, delta_tile, false, 0, dS_tile, false, kk / 16);
      t1 = scale_rows(t1, dec_exp);
      dk_acc = dk_acc + t1;

      {
        const _M16x32 KdKC = K_tile * t1;
        dgam_row32 = dgam_row32 - KdKC;
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < decltype(KdKC)::kElemsPerTile; _i++) {
          dgam_last += AT_NAX(KdKC, _i);
        }
      }

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
        dgam_last += gamma_last_exp *
            (S_tile.frag_at(0, kk / 16)[_i] * dS_tile.frag_at(0, kk / 16)[_i] +
             S_tile.frag_at(0, kk / 16 + 1)[_i] *
                 dS_tile.frag_at(0, kk / 16 + 1)[_i]);
      }

      {
        _M16x32 TbK;
        K_tile = scale_rows(K_tile, beta_fm);
        MM16x32x16(TbK, 0, TWinv_tile, false, 0, K_tile, false, 0);
        dgam_row32 = fmadd(TbK, dW_raw, dgam_row32);
      }

      reduce_tile_tg<kNSG, kGQA>(
          dk_acc,
          red_scratch,
          dk_,
          Hk * Dk,
          kk,
          valid_rows,
          sg_id,
          simd_lane_id);
    }

    // dbeta = rowsum(V * dV_b) + rowsum(K * dK_b) + rowsum(tril_(dA) * KKt)

    load_seq(V_tile, v_ + dv_idx, Dv * Hv);

    _M16x16 AKKt = dA_tile * KKt_tile;
    AKKt = AKKt * tril_mask;
    _M16x16 VdVb = fmadd(V_tile, TUddelta, AKKt);
    row_sum(dbeta_acc, VdVb);
    row_sum(dbeta_acc, KdKb);

    {
      threadgroup float* db_st = db_stage + sg_id * C;
      float _b0 = dbeta_acc[0];
      float _b1 = dbeta_acc[1];
      _b0 += simd_shuffle_xor(_b0, ushort(1));
      _b0 += simd_shuffle_xor(_b0, ushort(8));
      _b1 += simd_shuffle_xor(_b1, ushort(1));
      _b1 += simd_shuffle_xor(_b1, ushort(8));

      const short _r0 = fm;
      const short _r1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
      if (_r0 < valid_rows) {
        db_st[_r0] = _b0;
      }
      if (_r1 < valid_rows) {
        db_st[_r1] = _b1;
      }
    }

    dgam_row = reduce(dgam_row32);

    float dgam_acc[2] = {0.0f, 0.0f};
    dgam_row = dgam_row + dgam_pair;
    row_sum(dgam_acc, dgam_row);

    _M16x16 cs_tile;
    MM16x16x16(cs_tile, 0, dgam_pair, true, 0, Ones_tile, false, 0);

    const float dgam_last_red = simd_sum(dgam_last);

    {
      threadgroup float* dg_st = dg_stage + sg_id * C;
      float _d0 = dgam_acc[0];
      float _d1 = dgam_acc[1];
      _d0 += simd_shuffle_xor(_d0, ushort(1));
      _d0 += simd_shuffle_xor(_d0, ushort(8));
      _d1 += simd_shuffle_xor(_d1, ushort(1));
      _d1 += simd_shuffle_xor(_d1, ushort(8));

      const short _r0 = fm;
      const short _r1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;

      _d0 -= AT_NAX(cs_tile, 0);
      _d1 -= AT_NAX(cs_tile, mlx::steel::BaseNAXFrag::kElemCols);

      if (_r0 == valid_rows - 1) {
        _d0 += dgam_last_red;
      }
      if (_r0 < valid_rows) {
        dg_st[_r0] = _d0;
      }
      if (_r1 == valid_rows - 1) {
        _d1 += dgam_last_red;
      }
      if (_r1 < valid_rows) {
        dg_st[_r1] = _d1;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0 && thread_index_in_simdgroup < (uint)valid_rows) {
      float b_sum = 0.0f;
      float g_sum = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (short _s = 0; _s < kNSG; _s++) {
        b_sum += db_stage[_s * C + thread_index_in_simdgroup];
        g_sum += dg_stage[_s * C + thread_index_in_simdgroup];
      }
      const int idx = thread_index_in_simdgroup * Hv + hv_idx;
      db_[idx] = b_sum;
      dg_[idx] = g_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dS update
    // dS = gamma_C * dS + dO.T @ (gamma * Q) - ddelta.T @ W
    SCALE_NAX(dS_tile, gamma_last_exp);
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 term, W_raw;
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      // dO.T @ (gamma * Q)  -> [dv, dk]
      Q_tile = scale_rows(Q_tile, row_exp);
      MM16x32x16(term, 0, dout_tile, true, 0, Q_tile, false, 0);

      // - ddelta.T @ W,  with W = gamma * (TWinv @ beta*K)
      K_tile = scale_rows(K_tile, beta_fm);
      MM16x32x16(W_raw, 0, TWinv_tile, false, 0, K_tile, false, 0);
      W_raw = scale_rows(-W_raw, row_exp);
      MMA16x32x16(term, 0, ddelta_tile, true, 0, W_raw, false, 0);

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(term)::kElemsPerFrag; _i++) {
        dS_tile.frag_at(0, kk / 16)[_i] += AT_NAX(term, _i);
        dS_tile.frag_at(0, kk / 16 + 1)[_i] +=
            AT_NAX(term, decltype(term)::kElemsPerFrag + _i);
      }
    }
  };

  // Walk chunks in reverse. The tail chunk comes first.
  int c = n_chunks - 1;
  const short tail = short(T - c * C);
  if (tail != C) {
    process_chunk(tail, metal::true_type{});
    q_ -= C * Hk * Dk;
    k_ -= C * Hk * Dk;
    v_ -= C * Hv * Dv;
    co_ -= C * Hv * Dv;
    g_ -= C * Hv;
    beta_ -= C * Hv;
    dq_ -= C * Hk * Dk;
    dk_ -= C * Hk * Dk;
    dv_ -= C * Hv * Dv;
    dg_ -= C * Hv;
    db_ -= C * Hv;
    c_state -= Dv * Dk;
    i_chunk_mats -= 3 * 256;
    i_chunk_delta -= C * Dv;
    --c;
  }
  for (; c >= 0; --c) {
    process_chunk(C, metal::false_type{});
    q_ -= C * Hk * Dk;
    k_ -= C * Hk * Dk;
    v_ -= C * Hv * Dv;
    co_ -= C * Hv * Dv;
    g_ -= C * Hv;
    beta_ -= C * Hv;
    dq_ -= C * Hk * Dk;
    dk_ -= C * Hk * Dk;
    dv_ -= C * Hv * Dv;
    dg_ -= C * Hv;
    db_ -= C * Hv;
    c_state -= Dv * Dk;
    i_chunk_mats -= 3 * 256;
    i_chunk_delta -= C * Dv;
  }

  dS_tile.store(o_dh, Dk);
}

template <int kNSG, typename TileT>
METAL_FUNC void reduce_tile_tg0(
    thread TileT& acc,
    threadgroup float* scratch,
    const short sg_id,
    const ushort simd_lane_id) {
  constexpr short kE = TileT::kElemsPerTile;

  STEEL_PRAGMA_UNROLL
  for (short lo = kNSG / 2; lo > 0; lo >>= 1) {
    if (sg_id >= lo && sg_id < 2 * lo) {
      const short dst = sg_id - lo;
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < kE; _i++) {
        scratch[(dst * 32 + simd_lane_id) * kE + _i] = AT_NAX(acc, _i);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id < lo) {
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < kE; _i++) {
        AT_NAX(acc, _i) += scratch[(sg_id * 32 + simd_lane_id) * kE + _i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_vjp_fused_nax0(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B, Hv, n_chunks, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<float>* dq [[buffer(9)]],
    device mlx_atomic<float>* dk [[buffer(10)]],
    device float* dv [[buffer(11)]],
    device float* dg [[buffer(12)]],
    device float* db [[buffer(13)]],
    device float* dh [[buffer(14)]],
    // Cached, chunk-local forward intermediates produced by the forward-save
    // pass (see gated_delta_fused_nax): avoids redoing the K@K^T contraction,
    // the 15-step Neumann inversion, and the W/U/S contraction for delta.
    const device float* chunk_mats [[buffer(15)]],
    const device float* chunk_delta [[buffer(16)]], // [B, Hv, n_chunks, C, Dv]
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  using _M16xDk = mlx::steel::NAXTile<float, 1, Dk / 16>;

  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dv_idx = thread_position_in_grid.y * 16;
  const short sg_id = thread_position_in_threadgroup.y; // 0..3

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  const int n_chunks = (T + C - 1) / C;
  const int t_last = (n_chunks - 1) * C;

  // Pointers positioned at the final chunk
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto dq_ = dq + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto dk_ = dk + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;

  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;
  auto dv_ = dv + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;
  auto co_ = cot_o + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;

  auto g_ = g + b_idx * T * Hv + t_last * Hv;
  auto beta_ = beta + b_idx * T * Hv + t_last * Hv;
  auto dg_ = dg + b_idx * T * Hv + t_last * Hv;
  auto db_ = db + b_idx * T * Hv + t_last * Hv;

  auto c_state = state_cache + (n * n_chunks * Dv + dv_idx) * Dk +
      (n_chunks - 1) * Dv * Dk;

  // Cached forward intermediates, walked in reverse alongside c_state.
  auto i_chunk_mats =
      chunk_mats + n * n_chunks * 3 * 256 + (n_chunks - 1) * 3 * 256;
  auto i_chunk_delta =
      chunk_delta + n * n_chunks * C * Dv + dv_idx + (n_chunks - 1) * C * Dv;

  auto i_cot_h = cot_h + (n * Dv + dv_idx) * Dk;
  auto o_dh = dh + (n * Dv + dv_idx) * Dk;

  // One threadgroup per (b, hv) now covers every Dv slice, so the Dv reduction
  // never leaves threadgroup memory.
  constexpr int kNSG = Dv / 16;
  // Under GQA several threadgroups still share an hk, so dq/dk keep an atomic
  // for that axis -- but with kNSG writers folded away first.
  constexpr bool kGQA = (Hv != Hk);

  threadgroup float gamma_all[C * kNSG];
  threadgroup float* gamma = gamma_all + sg_id * C;

  // (kNSG/2) destinations x 32 lanes x kElemsPerTile(_M16x32) floats.
  threadgroup float red_scratch[(kNSG / 2) * 32 * 16];
  threadgroup float db_stage[kNSG * C];
  threadgroup float dg_stage[kNSG * C];

  float beta_fm[2];

  // Carried state gradient, dL/dS. Same [dv, dk] orientation as S_tile.
  _M16xDk dS_tile;
  dS_tile.load(i_cot_h, Dk);

  // Forward state at chunk entry, reloaded from the checkpoint each chunk
  _M16xDk S_tile;

  // Recomputed forward tiles
  _M16x32 K_tile, Q_tile;
  _M16xDk W_tile;
  _M16x16 K16_tile, Q16_tile;
  _M16x16 V_tile;
  _M16x16 U_tile;
  _M16x16 WS_tile;
  _M16x16 delta_tile;
  _M16x16 QKt_tile, QKt_raw;
  _M16x16 KKt_tile;
  _M16x16 TWinv_tile, TUinv_tile;
  _M16x16 TMP_tile;

  // Backward tiles
  _M16x16 dout_tile;
  _M16x16 ddelta_tile;
  _M16x16 dTU_tile, dTW_tile;
  _M16x16 dTinv_tile;
  _M16x16 dA_tile, G_tile;
  _M16x16 tri_decay;

  _M16x16 I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    AT_NAX(I_tile, _i) = (_c.x == _c.y) ? 1.0f : 0.0f;
  }

  _M16x16 NI_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(NI_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    AT_NAX(NI_tile, _i) = (_c.x == _c.y) ? -1.0f : 0.0f;
  }

  _M16x16 Ones_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(Ones_tile)::kElemsPerFrag; _i++) {
    AT_NAX(Ones_tile, _i) = 1.0f;
  }

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

    // reload the checkpoint
    S_tile.load(c_state, Dk);

    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(
              metal::max(g_[thread_index_in_simdgroup * Hv + hv_idx], 1e-6))
        : 0.0f;

    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float gamma_last = gamma[C - 1];
    const float gamma_last_exp = metal::fast::exp(gamma_last);

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    // Per-lane decay factors. gamma[j] == gamma[valid_rows-1] for
    // j >= valid_rows (the prefix sum of zeros), so the tail chunk needs no
    // special case here.
    const float row_exp[2] = {
        metal::fast::exp(gamma[fm]), metal::fast::exp(gamma[fm1])};
    const float dec_exp[2] = {
        metal::fast::exp(gamma_last - gamma[fm]),
        metal::fast::exp(gamma_last - gamma[fm1])};

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(tri_decay)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      AT_NAX(tri_decay, _i) =
          (_c.x > _c.y) ? 0.f : metal::fast::exp(gamma[_c.y] - gamma[_c.x]);
    }

    KKt_tile.load(i_chunk_mats, 16);
    TWinv_tile.load(i_chunk_mats + 256, 16);
    QKt_raw.load(i_chunk_mats + 512, 16);

    QKt_tile = QKt_raw;
    MUL_NAX(QKt_tile, QKt_tile, tri_decay);

    TUinv_tile = TWinv_tile;
    MUL_NAX(TUinv_tile, TUinv_tile, tri_decay);

    delta_tile.load(i_chunk_delta, Dv);

    // dgamma accumulators:
    //   dgam_row  : per-row
    //   dgam_pair : row-minus-col
    //   dgam_last : the gamma_{C-1}
    _M16x16 dgam_row, dgam_pair;
    dgam_row.clear();
    dgam_pair.clear();
    float dgam_last = 0.0f;
    float dbeta_acc[2] = {0.0f, 0.0f};

    load_seq(dout_tile, co_ + dv_idx, Hv * Dv);

    // dODT = (dO @ delta.T * D)
    _M16x16 dODT;
    MM16x16x16(dODT, 0, dout_tile, false, 0, delta_tile, true, 0);
    MUL_NAX(dODT, dODT, tri_decay);

    // cannot fuse this because ddelta is used by the next loop
    MM16x16x16(ddelta_tile, 0, QKt_tile, true, 0, dout_tile, false, 0);
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      SCALE2_P(K_tile, dec_exp);
      MMA16x16x32(ddelta_tile, 0, K_tile, false, 0, dS_tile, true, kk / 16);
    }

    // ddelta = M^T @ dO + K_dec @ dS^T
    // dq = gamma * (dO @ S) + dODT @ K
    dTW_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dW_raw;
      _M16x32 dq_acc;
      _M16x32 Kb_tile;
      load_seq(K_tile, k_ + kk, Dk * Hk);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      NSCALE_ROW_P(dW_raw, row_exp);

      SCALE_BETA_NAX_O(Kb_tile, K_tile, beta_fm);
      MMA16x16x32(dTW_tile, 0, dW_raw, false, 0, Kb_tile, true, 0);

      MM16x32x16(dq_acc, 0, dout_tile, false, 0, S_tile, false, kk / 16);
      SCALE_ROW_P(dq_acc, row_exp);
      MMA16x32x16(dq_acc, 0, dODT, false, 0, K_tile, false, 0);

      // dq reduces over Dv, which is now entirely inside this threadgroup.
      reduce_tile_tg<kNSG>(dq_acc, red_scratch, sg_id, simd_lane_id);

      if (sg_id == 0) {
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < decltype(dq_acc)::kElemsPerTile; _i++) {
          const short _f = _i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_w); // {fn, fm}
          const short _fn = _c.x + _f * 16;
          const short _fm = _c.y;
          if (_fm < valid_rows) {
            const int idx = _fm * Hk * Dk + kk + _fn;
            if (kGQA) {
              mlx_atomic_fetch_add_explicit(dq_, AT_NAX(dq_acc, _i), idx);
            } else {
              // Dv reduction done in threadgroup memory: single writer.
              mlx_atomic_store_explicit(dq_, AT_NAX(dq_acc, _i), idx);
            }
          }
        }
      }
    }

    // dv = B(Tu.T @ ddelta).  Indexed by dv_idx, so no reduction.
    _M16x16 dVb_tile;
    MM16x16x16(dVb_tile, 0, TUinv_tile, true, 0, ddelta_tile, false, 0);
    _M16x16 TUddelta = dVb_tile; // pre-beta, reused by dbeta
    SCALE_BETA_NAX(dVb_tile, beta_fm);

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(dVb_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      if (_c.y < valid_rows) {
        dv_[_c.y * Hv * Dv + dv_idx + _c.x] = AT_NAX(dVb_tile, _i);
      }
    }

    // dT_U = ddelta @ (beta * V).T
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    MM16x16x16(dTU_tile, 0, ddelta_tile, false, 0, V_tile, true, 0);
    MUL_NAX(dTU_tile, dTU_tile, tri_decay); // dTinv = dT_W + (dT_U * D).

    ADD_NAX(dTinv_tile, dTW_tile, dTU_tile);

    // dA = -T.T @ dTinv @ T.T
    MM16x16x16(TMP_tile, 0, TWinv_tile, true, 0, dTinv_tile, false, 0);
    MM16x16x16(dA_tile, 0, TMP_tile, false, 0, TWinv_tile, true, 0);
    SCALE_NAX(dA_tile, -1.0f);

    // G = tril_(dA) * beta, and GGt = G + G.T
    G_tile = dA_tile;
    SCALE_TRIEQ_NAX1(G_tile, beta_fm);
    _M16x16 GGt_tile;
    GGt_tile = G_tile;
    MMA16x16x16(GGt_tile, 0, I_tile, true, 0, G_tile, true, 0);

    // dgamma pair sites
    _M16x16 P_tile;
    _M16x16 R_tile;

    MUL_NAX(P_tile, QKt_raw, dODT);
    MUL_NAX(R_tile, TWinv_tile, dTU_tile);
    ADD_NAX(dgam_pair, P_tile, R_tile);

    _M16x32 KdKb; // rowsum(K * dK_b) source, needed for dbeta
    KdKb.clear();
    _M16x32 dgam_row32;
    dgam_row32.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dk_acc, dW_raw, dKb_tile, t1, KdKb_tmp;
      _M16x32 QgdOS, TbK, KdKC_blk;

      load_seq(Q_tile, q_ + kk, Hk * Dk); // [16 x 32]
      load_seq(K_tile, k_ + kk, Dk * Hk); // [16 x 32]

      // gamma: + rowsum(Q * gamma*(dO @ S))
      MM16x32x16(QgdOS, 0, dout_tile, false, 0, S_tile, false, kk / 16);
      SCALE_ROW_P(QgdOS, row_exp);
      MULA_NAX(dgam_row32, QgdOS, Q_tile);

      // dODT.T @ Q  and  (G + G.T) @ K
      MM16x32x16(dk_acc, 0, dODT, true, 0, Q_tile, false, 0);
      MMA16x32x16(dk_acc, 0, GGt_tile, false, 0, K_tile, false, 0);

      // B (T.T @ dW_raw),  dW_raw = -gamma * (ddelta @ S)
      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      NSCALE_ROW_P(dW_raw, row_exp);
      MM16x32x16(dKb_tile, 0, TWinv_tile, true, 0, dW_raw, false, 0);

      // dbeta term uses dK_b before the beta scaling, with raw K
      MUL_NAX(KdKb_tmp, dKb_tile, K_tile);
      ADD_NAX(KdKb, KdKb, KdKb_tmp);

      SCALE_BETA_NAX(dKb_tile, beta_fm);
      ADD_NAX(dk_acc, dk_acc, dKb_tile);

      // D_C * (delta @ dS)
      MM16x32x16(t1, 0, delta_tile, false, 0, dS_tile, false, kk / 16);
      SCALE2_P(t1, dec_exp);
      ADD_NAX(dk_acc, dk_acc, t1);

      // gamma stuff
      MUL_NAX(KdKC_blk, K_tile, t1);
      SUB_NAX(dgam_row32, dgam_row32, KdKC_blk);
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(KdKC_blk)::kElemsPerTile; _i++) {
        dgam_last += AT_NAX(KdKC_blk, _i);
      }

      // gamma stuff: gamma_C * sum(S * dS)
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
        dgam_last += gamma_last_exp * S_tile.frag_at(0, kk / 16)[_i] *
            dS_tile.frag_at(0, kk / 16)[_i];
        dgam_last += gamma_last_exp * S_tile.frag_at(0, kk / 16 + 1)[_i] *
            dS_tile.frag_at(0, kk / 16 + 1)[_i];
      }

      // gamma stuff
      SCALE_BETA_NAX(K_tile, beta_fm);
      SCALE_NAX(dW_raw, -1.0f);
      MM16x32x16(TbK, 0, TWinv_tile, false, 0, K_tile, false, 0);
      MULS_NAX(dgam_row32, TbK, dW_raw);

      // dk reduces over Dv the same way dq does.
      reduce_tile_tg<kNSG>(dk_acc, red_scratch, sg_id, simd_lane_id);

      if (sg_id == 0) {
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < decltype(dk_acc)::kElemsPerTile; _i++) {
          const short _f = _i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_w); // {fn, fm}
          if (_c.y < valid_rows) {
            const int idx = _c.y * Hk * Dk + kk + _c.x + _f * 16;
            if (kGQA) {
              mlx_atomic_fetch_add_explicit(dk_, AT_NAX(dk_acc, _i), idx);
            } else {
              mlx_atomic_store_explicit(dk_, AT_NAX(dk_acc, _i), idx);
            }
          }
        }
      }
    }

    // dbeta = rowsum(V * dV_b) + rowsum(K * dK_b) + rowsum(tril_(dA) * KKt)
    _M16x16 VdVb;
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    MUL_NAX(VdVb, V_tile, TUddelta);

    _M16x16 AKKt;
    TRIL_NAX(AKKt, dA_tile);
    MUL_NAX(AKKt, AKKt, KKt_tile);

    ADD_NAX(VdVb, VdVb, AKKt);
    ROWSUM1_NAX(dbeta_acc, VdVb);
    ROWSUM1_NAX(dbeta_acc, KdKb);

    {
      threadgroup float* db_st = db_stage + sg_id * C;
      STEEL_PRAGMA_UNROLL
      for (short _r = 0; _r < 2; _r++) {
        float _b = dbeta_acc[_r];
        _b += simd_shuffle_xor(_b, ushort(1));
        _b += simd_shuffle_xor(_b, ushort(8));

        const short _row = fm + _r * mlx::steel::BaseNAXFrag::kElemRowsJump;
        if (_row < valid_rows) {
          db_st[_row] = _b;
        }
      }
    }

    // Fold the two fragments: only the row sum of dgam_row is used downstream,
    // and row sums are additive across fragments.
    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
      AT_NAX(dgam_row, _i) = AT_NAX(dgam_row32, _i) +
          AT_NAX(dgam_row32, mlx::steel::BaseNAXFrag::kElemsPerFrag + _i);
    }

    // reduce and store dgamma: rowsum(row + pair) - colsum(pair)
    ADD_NAX(dgam_row, dgam_row, dgam_pair);
    _M16x16 dgam_tile, cs_tile;
    MM16x16x16(dgam_tile, 0, dgam_row, false, 0, Ones_tile, false, 0);
    MM16x16x16(cs_tile, 0, dgam_pair, true, 0, Ones_tile, false, 0);
    SUB_NAX(dgam_tile, dgam_tile, cs_tile);

    {
      threadgroup float* dg_st = dg_stage + sg_id * C;
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(dgam_tile)::kElemsPerFrag; _i++) {
        const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
        if (_c.x == 0) {
          dg_st[_c.y] = AT_NAX(dgam_tile, _i);
        }
      }
      // The gamma_{C-1}-only term lands on the same slot the loop just wrote,
      // so the lanes must be ordered before lane 0 accumulates into it.
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const float dgam_last_red = simd_sum(dgam_last);
      if (simd_lane_id == 0 && valid_rows > 0) {
        dg_st[valid_rows - 1] += dgam_last_red;
      }
    }

    // One barrier serves both stages; simdgroup 0 commits.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0 && thread_index_in_simdgroup < (uint)valid_rows) {
      float b_sum = 0.0f;
      float g_sum = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (short _s = 0; _s < kNSG; _s++) {
        b_sum += db_stage[_s * C + thread_index_in_simdgroup];
        g_sum += dg_stage[_s * C + thread_index_in_simdgroup];
      }
      const int idx = thread_index_in_simdgroup * Hv + hv_idx;
      db_[idx] = b_sum;
      dg_[idx] = g_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dS update
    // dS = gamma_C * dS + dO.T @ (gamma * Q) - ddelta.T @ W
    SCALE_NAX(dS_tile, gamma_last_exp);
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 term, W_raw;
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      // dO.T @ (gamma * Q)  -> [dv, dk]
      SCALE_ROW_P(Q_tile, row_exp);
      MM16x32x16(term, 0, dout_tile, true, 0, Q_tile, false, 0);

      // - ddelta.T @ W,  with W = gamma * (TWinv @ beta*K)
      SCALE_BETA_NAX(K_tile, beta_fm);
      MM16x32x16(W_raw, 0, TWinv_tile, false, 0, K_tile, false, 0);
      NSCALE_ROW_P(W_raw, row_exp);
      MMA16x32x16(term, 0, ddelta_tile, true, 0, W_raw, false, 0);

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(term)::kElemsPerFrag; _i++) {
        dS_tile.frag_at(0, kk / 16)[_i] += AT_NAX(term, _i);
        dS_tile.frag_at(0, kk / 16 + 1)[_i] +=
            AT_NAX(term, decltype(term)::kElemsPerFrag + _i);
      }
    }
  };

  // Walk chunks in reverse. The tail chunk comes first.
  int c = n_chunks - 1;
  const short tail = short(T - c * C);
  if (tail != C) {
    process_chunk(tail, metal::true_type{});
    q_ -= C * Hk * Dk;
    k_ -= C * Hk * Dk;
    v_ -= C * Hv * Dv;
    co_ -= C * Hv * Dv;
    g_ -= C * Hv;
    beta_ -= C * Hv;
    dq_ -= C * Hk * Dk;
    dk_ -= C * Hk * Dk;
    dv_ -= C * Hv * Dv;
    dg_ -= C * Hv;
    db_ -= C * Hv;
    c_state -= Dv * Dk;
    i_chunk_mats -= 3 * 256;
    i_chunk_delta -= C * Dv;
    --c;
  }
  for (; c >= 0; --c) {
    process_chunk(C, metal::false_type{});
    q_ -= C * Hk * Dk;
    k_ -= C * Hk * Dk;
    v_ -= C * Hv * Dv;
    co_ -= C * Hv * Dv;
    g_ -= C * Hv;
    beta_ -= C * Hv;
    dq_ -= C * Hk * Dk;
    dk_ -= C * Hk * Dk;
    dv_ -= C * Hv * Dv;
    dg_ -= C * Hv;
    db_ -= C * Hv;
    c_state -= Dv * Dk;
    i_chunk_mats -= 3 * 256;
    i_chunk_delta -= C * Dv;
  }

  dS_tile.store(o_dh, Dk);
}

// Postprocessing dg. Similar idea to what is done in the triton kernel.
// The kernel computes the gradient with respect to the log.
template <typename InT, int C>
[[kernel]] void gated_delta_dgamma_to_dg(
    const device InT* g [[buffer(0)]],
    device float* dg [[buffer(1)]], // in: dL/dgamma, out: dL/dg
    constant int& T [[buffer(2)]],
    constant int& Hv [[buffer(3)]],
    constant int& n_total [[buffer(4)]], // B * Hv
    uint2 pos [[thread_position_in_grid]]) {
  const int n = int(pos.x); // b * Hv + hv
  const int c = int(pos.y); // chunk index

  if (n >= n_total) {
    return;
  }

  const int t0 = c * C;
  if (t0 >= T) {
    return;
  }
  const int len = min(C, T - t0);

  const int b_idx = n / Hv;
  const int hv_idx = n % Hv;
  const int base = b_idx * T * Hv + hv_idx;

  float acc = 0.0f;
  for (int j = len - 1; j >= 0; --j) {
    const int idx = base + (t0 + j) * Hv;
    acc += static_cast<float>(dg[idx]);
    float gv = metal::max(static_cast<float>(g[idx]), 1e-6f);
    dg[idx] = acc / gv;
  }
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
