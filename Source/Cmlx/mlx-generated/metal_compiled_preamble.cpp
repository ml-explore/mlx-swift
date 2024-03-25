// Copyright © 2023-24 Apple Inc.

namespace mlx::core::metal {

const char* get_kernel_preamble() {
  return R"preamble(
# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/compiled_preamble.h"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 418 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/compiled_preamble.h" 2


# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/binary.h" 1







# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16.h" 1






using namespace metal;
# 19 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16.h"
constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {

  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }

  uint32_t float_bits = as_type<uint32_t>(x);


  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);


  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {

  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;





struct _MLX_BFloat16 {


  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}




  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}




  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};







constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}
# 166 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16.h"
constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC float operator+(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC float operator+(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC float operator+(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC float operator+(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator+(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator+(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator+(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator+(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); };;
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC float operator-(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC float operator-(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC float operator-(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC float operator-(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator-(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator-(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator-(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator-(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); };;
constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC float operator*(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC float operator*(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC float operator*(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC float operator*(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator*(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator*(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator*(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator*(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); };;
constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC float operator/(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC float operator/(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC float operator/(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC float operator/(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator/(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator/(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator/(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator/(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); };;
# 183 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16.h"
constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); };;
# 220 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16.h"
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device float& operator+=( device float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread float& operator+=( thread float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup float& operator+=( threadgroup float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device float& operator-=( device float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread float& operator-=( thread float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup float& operator-=( threadgroup float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device float& operator*=( device float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread float& operator*=( thread float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup float& operator*=( threadgroup float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device float& operator/=( device float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread float& operator/=( thread float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup float& operator/=( threadgroup float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device half& operator+=( device half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread half& operator+=( thread half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup half& operator+=( threadgroup half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device half& operator-=( device half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread half& operator-=( thread half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup half& operator-=( threadgroup half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device half& operator*=( device half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread half& operator*=( thread half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup half& operator*=( threadgroup half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device half& operator/=( device half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread half& operator/=( thread half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup half& operator/=( threadgroup half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int16_t& operator+=( device int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int16_t& operator+=( thread int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int16_t& operator+=( threadgroup int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int16_t& operator-=( device int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int16_t& operator-=( thread int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int16_t& operator-=( threadgroup int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int16_t& operator*=( device int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int16_t& operator*=( thread int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int16_t& operator*=( threadgroup int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int16_t& operator/=( device int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int16_t& operator/=( thread int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int16_t& operator/=( threadgroup int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int32_t& operator+=( device int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int32_t& operator+=( thread int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int32_t& operator+=( threadgroup int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int32_t& operator-=( device int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int32_t& operator-=( thread int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int32_t& operator-=( threadgroup int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int32_t& operator*=( device int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int32_t& operator*=( thread int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int32_t& operator*=( threadgroup int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int32_t& operator/=( device int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int32_t& operator/=( thread int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int32_t& operator/=( threadgroup int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int64_t& operator+=( device int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int64_t& operator+=( thread int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int64_t& operator+=( threadgroup int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int64_t& operator-=( device int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int64_t& operator-=( thread int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int64_t& operator-=( threadgroup int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int64_t& operator*=( device int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int64_t& operator*=( thread int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int64_t& operator*=( threadgroup int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int64_t& operator/=( device int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int64_t& operator/=( thread int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int64_t& operator/=( threadgroup int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint16_t& operator+=( device uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint16_t& operator+=( thread uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint16_t& operator+=( threadgroup uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint16_t& operator-=( device uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint16_t& operator-=( thread uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint16_t& operator-=( threadgroup uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint16_t& operator*=( device uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint16_t& operator*=( thread uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint16_t& operator*=( threadgroup uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint16_t& operator/=( device uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint16_t& operator/=( thread uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint16_t& operator/=( threadgroup uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint32_t& operator+=( device uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint32_t& operator+=( thread uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint32_t& operator+=( threadgroup uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint32_t& operator-=( device uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint32_t& operator-=( thread uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint32_t& operator-=( threadgroup uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint32_t& operator*=( device uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint32_t& operator*=( thread uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint32_t& operator*=( threadgroup uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint32_t& operator/=( device uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint32_t& operator/=( thread uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint32_t& operator/=( threadgroup uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint64_t& operator+=( device uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint64_t& operator+=( thread uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint64_t& operator+=( threadgroup uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint64_t& operator-=( device uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint64_t& operator-=( thread uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint64_t& operator-=( threadgroup uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint64_t& operator*=( device uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint64_t& operator*=( thread uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint64_t& operator*=( threadgroup uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint64_t& operator/=( device uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint64_t& operator/=( thread uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint64_t& operator/=( threadgroup uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
# 245 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16.h"
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };;
constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };;
constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };;
constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;
# 257 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16.h"
typedef struct _MLX_BFloat16 bfloat16_t;





#pragma METAL internals : enable

namespace metal {

template <>
struct _numeric_limits_impl<bfloat16_t> : _fp_numeric_limits_impl_base {
  static constexpr constant int digits = 8;
  static constexpr constant int digits10 = 2;
  static constexpr constant int max_digits10 = 4;
  static constexpr constant int radix = 2;
  static constexpr constant int min_exponent = -125;
  static constexpr constant int min_exponent10 = -37;
  static constexpr constant int max_exponent = 128;
  static constexpr constant int max_exponent10 = 38;

  static constexpr bfloat16_t min() {
    return _MLX_BFloat16(0x0080, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t lowest() {
    return _MLX_BFloat16(0xFF7F, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t max() {
    return _MLX_BFloat16(0x7F7F, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t epsilon() {
    return _MLX_BFloat16(0x3C00, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t round_error() {
    return _MLX_BFloat16(0x3F00, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t infinity() {
    return _MLX_BFloat16(0x7F80, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t quiet_NaN() {
    return _MLX_BFloat16(0x7FC0, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t signaling_NaN() {
    return _MLX_BFloat16(0x7F80, _MLX_BFloat16::bits_to_bfloat());
  }
  static constexpr bfloat16_t denorm_min() {
    return _MLX_BFloat16(0x0001, _MLX_BFloat16::bits_to_bfloat());
  }
};

METAL_FUNC bool isnan(_MLX_BFloat16 x) {
  return x != x;
}

}

#pragma METAL internals : disable



# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16_math.h" 1
# 228 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16_math.h"
namespace metal {

METAL_FUNC bfloat16_t abs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t acos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acos(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t acosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acosh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t asin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asin(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t asinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asinh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t atan(bfloat16_t y_over_x) { return static_cast<bfloat16_t>( __metal_atan(static_cast<float>(y_over_x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t atan2(bfloat16_t y, bfloat16_t x) { return static_cast<bfloat16_t>( __metal_atan2(static_cast<float>(y), static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t atanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_atanh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t ceil(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_ceil(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t cos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cos(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t cosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cosh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t cospi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cospi(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t divide(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_divide(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t exp(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t exp10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp10(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t exp2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp2(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fabs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fdim(bfloat16_t x, bfloat16_t y) { float t = static_cast<float>(x - y); return static_cast<bfloat16_t>(select(t, float(0), t < float(0) || x == y)); } METAL_FUNC bfloat16_t floor(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_floor(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fma(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fma( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))); } METAL_FUNC bfloat16_t fmax(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmax3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmedian3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmin(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmin3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmod(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmod(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fract(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fract(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t frexp(bfloat16_t x, thread int& exp) { return static_cast<bfloat16_t>(__metal_frexp(static_cast<float>(x), &exp)); } METAL_FUNC bfloat16_t ldexp(bfloat16_t x, int k) { return static_cast<bfloat16_t>(__metal_ldexp(static_cast<float>(x), k, __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t log(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t log10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log10(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t log2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log2(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t max(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t max3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t median3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t min(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t min3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t nextafter(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_nextafter(static_cast<float>(x), static_cast<float>(y))); } METAL_FUNC bfloat16_t pow(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_pow(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t powr(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_powr(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t rint(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rint(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t round(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_round(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t rsqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rsqrt(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t sin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sin(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t sinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t sinpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinpi(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t sqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sqrt(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t tan(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tan(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t tanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t tanpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanpi(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t trunc(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_trunc(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); };





namespace fast {

METAL_FUNC bfloat16_t abs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t acos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acos(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t acosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acosh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t asin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asin(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t asinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asinh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t atan(bfloat16_t y_over_x) { return static_cast<bfloat16_t>( __metal_atan(static_cast<float>(y_over_x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t atan2(bfloat16_t y, bfloat16_t x) { return static_cast<bfloat16_t>( __metal_atan2(static_cast<float>(y), static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t atanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_atanh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t ceil(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_ceil(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t cos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cos(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t cosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cosh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t cospi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cospi(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t divide(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_divide(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t exp(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t exp10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp10(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t exp2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp2(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fabs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fdim(bfloat16_t x, bfloat16_t y) { float t = static_cast<float>(x - y); return static_cast<bfloat16_t>(select(t, float(0), t < float(0) || x == y)); } METAL_FUNC bfloat16_t floor(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_floor(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fma(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fma( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))); } METAL_FUNC bfloat16_t fmax(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmax3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmedian3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmin(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmin3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmod(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmod(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fract(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fract(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t frexp(bfloat16_t x, thread int& exp) { return static_cast<bfloat16_t>(__metal_frexp(static_cast<float>(x), &exp)); } METAL_FUNC bfloat16_t ldexp(bfloat16_t x, int k) { return static_cast<bfloat16_t>(__metal_ldexp(static_cast<float>(x), k, __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t log(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t log10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log10(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t log2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log2(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t max(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t max3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t median3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t min(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t min3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t nextafter(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_nextafter(static_cast<float>(x), static_cast<float>(y))); } METAL_FUNC bfloat16_t pow(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_pow(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t powr(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_powr(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t rint(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rint(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t round(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_round(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t rsqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rsqrt(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t sin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sin(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t sinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t sinpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinpi(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t sqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sqrt(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t tan(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tan(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t tanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t tanpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanpi(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t trunc(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_trunc(static_cast<float>(x), __METAL_FAST_MATH__)); };





}

namespace precise {

METAL_FUNC bfloat16_t abs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t acos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acos(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t acosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acosh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t asin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asin(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t asinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asinh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t atan(bfloat16_t y_over_x) { return static_cast<bfloat16_t>( __metal_atan(static_cast<float>(y_over_x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t atan2(bfloat16_t y, bfloat16_t x) { return static_cast<bfloat16_t>( __metal_atan2(static_cast<float>(y), static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t atanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_atanh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t ceil(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_ceil(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t cos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cos(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t cosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cosh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t cospi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cospi(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t divide(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_divide(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t exp(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t exp10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp10(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t exp2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp2(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fabs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fdim(bfloat16_t x, bfloat16_t y) { float t = static_cast<float>(x - y); return static_cast<bfloat16_t>(select(t, float(0), t < float(0) || x == y)); } METAL_FUNC bfloat16_t floor(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_floor(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fma(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fma( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))); } METAL_FUNC bfloat16_t fmax(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmax3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmedian3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmin(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmin3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmod(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmod(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fract(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fract(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t frexp(bfloat16_t x, thread int& exp) { return static_cast<bfloat16_t>(__metal_frexp(static_cast<float>(x), &exp)); } METAL_FUNC bfloat16_t ldexp(bfloat16_t x, int k) { return static_cast<bfloat16_t>(__metal_ldexp(static_cast<float>(x), k, __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t log(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t log10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log10(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t log2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log2(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t max(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t max3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t median3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t min(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t min3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t nextafter(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_nextafter(static_cast<float>(x), static_cast<float>(y))); } METAL_FUNC bfloat16_t pow(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_pow(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t powr(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_powr(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t rint(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rint(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t round(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_round(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t rsqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rsqrt(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t sin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sin(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t sinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t sinpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinpi(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t sqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sqrt(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t tan(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tan(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t tanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t tanpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanpi(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t trunc(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_trunc(static_cast<float>(x), __METAL_PRECISE_MATH__)); };





}

}
# 384 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16_math.h"
namespace metal {

METAL_FUNC bfloat16_t simd_broadcast(bfloat16_t data, ushort broadcast_lane_id) { return _MLX_BFloat16(__metal_simd_broadcast(data.bits_, broadcast_lane_id), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle(bfloat16_t data, ushort simd_lane_id) { return _MLX_BFloat16(__metal_simd_shuffle(data.bits_, simd_lane_id), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_down( bfloat16_t data, bfloat16_t filling_data, ushort delta, ushort modulo) { return _MLX_BFloat16(__metal_simd_shuffle_and_fill_down( data.bits_, filling_data.bits_, delta, modulo), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_down( bfloat16_t data, bfloat16_t filling_data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_and_fill_down( data.bits_, filling_data.bits_, delta, __metal_get_simdgroup_size(ushort())), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_up( bfloat16_t data, bfloat16_t filling_data, ushort delta, ushort modulo) { return _MLX_BFloat16(__metal_simd_shuffle_and_fill_up( data.bits_, filling_data.bits_, delta, modulo), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_up( bfloat16_t data, bfloat16_t filling_data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_and_fill_up( data.bits_, filling_data.bits_, delta, __metal_get_simdgroup_size(ushort())), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_down(bfloat16_t data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_down(data.bits_, delta), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_rotate_down(bfloat16_t data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_rotate_down(data.bits_, delta), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_rotate_up(bfloat16_t data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_rotate_up(data.bits_, delta), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_up(bfloat16_t data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_up(data.bits_, delta), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_xor(bfloat16_t data, ushort mask) { return _MLX_BFloat16(__metal_simd_shuffle_xor(data.bits_, mask), _MLX_BFloat16::bits_to_bfloat()); };





METAL_FUNC bfloat16_t simd_max(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_max(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_min(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_min(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_prefix_exclusive_product(bfloat16_t data) { return static_cast<bfloat16_t>( __metal_simd_prefix_exclusive_product(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_prefix_exclusive_sum(bfloat16_t data) { return static_cast<bfloat16_t>( __metal_simd_prefix_exclusive_sum(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_prefix_inclusive_product(bfloat16_t data) { return static_cast<bfloat16_t>( __metal_simd_prefix_inclusive_product(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_prefix_inclusive_sum(bfloat16_t data) { return static_cast<bfloat16_t>( __metal_simd_prefix_inclusive_sum(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_product(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_product(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_sum(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_sum(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_xor(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_xor(static_cast<float>(data))); };

}
# 318 "Source/Cmlx/mlx/mlx/backend/metal/kernels/bf16.h" 2
# 9 "Source/Cmlx/mlx/mlx/backend/metal/kernels/binary.h" 2
# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/utils.h" 1






# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/complex.h" 1






using namespace metal;

struct complex64_t;

template <typename T>
static constexpr constant bool can_convert_to_complex64 =
    !is_same_v<T, complex64_t> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_complex64 =
    !is_same_v<T, complex64_t> &&
    (is_convertible_v<float, T> || is_convertible_v<bfloat16_t, T>);

struct complex64_t {
  float real;
  float imag;


  constexpr complex64_t(float real, float imag) : real(real), imag(imag){};


  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) thread : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) threadgroup : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) device : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) constant : real(x), imag(0) {}


  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const thread {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const threadgroup {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const device {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const constant {
    return static_cast<T>(real);
  }
};

constexpr complex64_t operator-(complex64_t x) {
  return {-x.real, -x.imag};
}

constexpr bool operator>=(complex64_t a, complex64_t b) {
  return (a.real > b.real) || (a.real == b.real && a.imag >= b.imag);
}

constexpr bool operator>(complex64_t a, complex64_t b) {
  return (a.real > b.real) || (a.real == b.real && a.imag > b.imag);
}

constexpr bool operator<=(complex64_t a, complex64_t b) {
  return operator>=(b, a);
}

constexpr bool operator<(complex64_t a, complex64_t b) {
  return operator>(b, a);
}

constexpr bool operator==(complex64_t a, complex64_t b) {
  return a.real == b.real && a.imag == b.imag;
}

constexpr complex64_t operator+(complex64_t a, complex64_t b) {
  return {a.real + b.real, a.imag + b.imag};
}

constexpr complex64_t operator-(complex64_t a, complex64_t b) {
  return {a.real - b.real, a.imag - b.imag};
}

constexpr complex64_t operator*(complex64_t a, complex64_t b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

constexpr complex64_t operator/(complex64_t a, complex64_t b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a.real * b.real + a.imag * b.imag;
  auto y = a.imag * b.real - a.real * b.imag;
  return {x / denom, y / denom};
}

constexpr complex64_t operator%(complex64_t a, complex64_t b) {
  auto real = a.real - (b.real * static_cast<int64_t>(a.real / b.real));
  auto imag = a.imag - (b.imag * static_cast<int64_t>(a.imag / b.imag));
  if (real != 0 && (real < 0 != b.real < 0)) {
    real += b.real;
  }
  if (imag != 0 && (imag < 0 != b.imag < 0)) {
    imag += b.imag;
  }
  return {real, imag};
}
# 8 "Source/Cmlx/mlx/mlx/backend/metal/kernels/utils.h" 2





template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};
# 32 "Source/Cmlx/mlx/mlx/backend/metal/kernels/utils.h"
template <> struct Limits<uint8_t> { static constexpr constant uint8_t max = metal::numeric_limits<uint8_t>::max(); static constexpr constant uint8_t min = metal::numeric_limits<uint8_t>::min(); static constexpr constant uint8_t finite_max = metal::numeric_limits<uint8_t>::max(); static constexpr constant uint8_t finite_min = metal::numeric_limits<uint8_t>::min(); };;
template <> struct Limits<uint16_t> { static constexpr constant uint16_t max = metal::numeric_limits<uint16_t>::max(); static constexpr constant uint16_t min = metal::numeric_limits<uint16_t>::min(); static constexpr constant uint16_t finite_max = metal::numeric_limits<uint16_t>::max(); static constexpr constant uint16_t finite_min = metal::numeric_limits<uint16_t>::min(); };;
template <> struct Limits<uint32_t> { static constexpr constant uint32_t max = metal::numeric_limits<uint32_t>::max(); static constexpr constant uint32_t min = metal::numeric_limits<uint32_t>::min(); static constexpr constant uint32_t finite_max = metal::numeric_limits<uint32_t>::max(); static constexpr constant uint32_t finite_min = metal::numeric_limits<uint32_t>::min(); };;
template <> struct Limits<uint64_t> { static constexpr constant uint64_t max = metal::numeric_limits<uint64_t>::max(); static constexpr constant uint64_t min = metal::numeric_limits<uint64_t>::min(); static constexpr constant uint64_t finite_max = metal::numeric_limits<uint64_t>::max(); static constexpr constant uint64_t finite_min = metal::numeric_limits<uint64_t>::min(); };;
template <> struct Limits<int8_t> { static constexpr constant int8_t max = metal::numeric_limits<int8_t>::max(); static constexpr constant int8_t min = metal::numeric_limits<int8_t>::min(); static constexpr constant int8_t finite_max = metal::numeric_limits<int8_t>::max(); static constexpr constant int8_t finite_min = metal::numeric_limits<int8_t>::min(); };;
template <> struct Limits<int16_t> { static constexpr constant int16_t max = metal::numeric_limits<int16_t>::max(); static constexpr constant int16_t min = metal::numeric_limits<int16_t>::min(); static constexpr constant int16_t finite_max = metal::numeric_limits<int16_t>::max(); static constexpr constant int16_t finite_min = metal::numeric_limits<int16_t>::min(); };;
template <> struct Limits<int32_t> { static constexpr constant int32_t max = metal::numeric_limits<int32_t>::max(); static constexpr constant int32_t min = metal::numeric_limits<int32_t>::min(); static constexpr constant int32_t finite_max = metal::numeric_limits<int32_t>::max(); static constexpr constant int32_t finite_min = metal::numeric_limits<int32_t>::min(); };;
template <> struct Limits<int64_t> { static constexpr constant int64_t max = metal::numeric_limits<int64_t>::max(); static constexpr constant int64_t min = metal::numeric_limits<int64_t>::min(); static constexpr constant int64_t finite_max = metal::numeric_limits<int64_t>::max(); static constexpr constant int64_t finite_min = metal::numeric_limits<int64_t>::min(); };;
# 54 "Source/Cmlx/mlx/mlx/backend/metal/kernels/utils.h"
template <> struct Limits<half> { static constexpr constant half max = metal::numeric_limits<half>::infinity(); static constexpr constant half min = -metal::numeric_limits<half>::infinity(); static constexpr constant half finite_max = metal::numeric_limits<half>::max(); static constexpr constant half finite_min = -metal::numeric_limits<half>::max(); };;
template <> struct Limits<float> { static constexpr constant float max = metal::numeric_limits<float>::infinity(); static constexpr constant float min = -metal::numeric_limits<float>::infinity(); static constexpr constant float finite_max = metal::numeric_limits<float>::max(); static constexpr constant float finite_min = -metal::numeric_limits<float>::max(); };;
template <> struct Limits<bfloat16_t> { static constexpr constant bfloat16_t max = metal::numeric_limits<bfloat16_t>::infinity(); static constexpr constant bfloat16_t min = -metal::numeric_limits<bfloat16_t>::infinity(); static constexpr constant bfloat16_t finite_max = metal::numeric_limits<bfloat16_t>::max(); static constexpr constant bfloat16_t finite_min = -metal::numeric_limits<bfloat16_t>::max(); };;

template <>
struct Limits<bool> {
  static constexpr constant bool max = true;
  static constexpr constant bool min = false;
};
# 73 "Source/Cmlx/mlx/mlx/backend/metal/kernels/utils.h"
template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    uint elem,
    device const int* shape,
    device const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}


template <typename stride_t>
METAL_FUNC stride_t elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = elem.x * strides[ndim - 1] + elem.y * strides[ndim - 2];
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}




template <typename stride_t>
METAL_FUNC stride_t elem_to_loc_1(uint elem, constant const stride_t& stride) {
  return elem * stride;
}

template <typename stride_t>
METAL_FUNC stride_t
elem_to_loc_2(uint2 elem, constant const stride_t strides[2]) {
  return elem.x * strides[1] + elem.y * strides[0];
}

template <typename stride_t>
METAL_FUNC stride_t
elem_to_loc_3(uint3 elem, constant const stride_t strides[3]) {
  return elem.x * strides[2] + elem.y * strides[1] + elem.z * strides[0];
}

template <int NDIM>
METAL_FUNC size_t elem_to_loc_nd(
    uint elem,
    device const int* shape,
    device const size_t* strides) {
  size_t loc = (elem % shape[NDIM - 1]) * strides[NDIM - 1];

#pragma clang loop unroll(full)
  for (int d = NDIM - 2; d >= 0; --d) {
    elem /= shape[d + 1];
    loc += (elem % shape[d]) * strides[d];
  }

  return loc;
}

template <int NDIM>
METAL_FUNC size_t elem_to_loc_nd(
    uint3 elem,
    constant const int shape[NDIM],
    constant const size_t strides[NDIM]) {
  size_t loc = elem.x * strides[NDIM - 1] + elem.y * strides[NDIM - 2];
  for (int d = NDIM - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

template <int NDIM>
METAL_FUNC int64_t elem_to_loc_nd(
    uint elem,
    constant const int shape[NDIM],
    constant const int64_t strides[NDIM]) {
  int64_t loc = (elem % shape[NDIM - 1]) * strides[NDIM - 1];

#pragma clang loop unroll(full)
  for (int d = NDIM - 2; d >= 0; --d) {
    elem /= shape[d + 1];
    loc += (elem % shape[d]) * strides[d];
  }

  return loc;
}

template <int NDIM>
METAL_FUNC int64_t elem_to_loc_nd(
    uint3 elem,
    constant const int shape[NDIM],
    constant const int64_t strides[NDIM]) {
  int64_t loc = elem.x * strides[NDIM - 1] + elem.y * strides[NDIM - 2];
  for (int d = NDIM - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * strides[d];
    elem.z /= shape[d];
  }
  return loc;
}




METAL_FUNC uint2 elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    int ndim) {
  uint2 loc = {
      static_cast<uint>(
          elem.x * a_strides[ndim - 1] + elem.y * a_strides[ndim - 2]),
      static_cast<uint>(
          elem.x * b_strides[ndim - 1] + elem.y * b_strides[ndim - 2])};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

METAL_FUNC uint3 elem_to_loc_3_nd(
    uint3 elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    int ndim) {
  uint3 loc = {
      static_cast<uint>(
          elem.x * a_strides[ndim - 1] + elem.y * a_strides[ndim - 2]),
      static_cast<uint>(
          elem.x * b_strides[ndim - 1] + elem.y * b_strides[ndim - 2]),
      static_cast<uint>(
          elem.x * c_strides[ndim - 1] + elem.y * c_strides[ndim - 2])};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    loc.z += l * c_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}




template <int NDIM>
METAL_FUNC uint2 elem_to_loc_2_nd(
    uint3 elem,
    constant const int shape[NDIM],
    constant const size_t a_strides[NDIM],
    constant const size_t b_strides[NDIM]) {
  uint2 loc = {
      static_cast<uint>(
          elem.x * a_strides[NDIM - 1] + elem.y * a_strides[NDIM - 2]),
      static_cast<uint>(
          elem.x * b_strides[NDIM - 1] + elem.y * b_strides[NDIM - 2])};
  for (int d = NDIM - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}

template <int NDIM>
METAL_FUNC uint3 elem_to_loc_3_nd(
    uint3 elem,
    constant const int shape[NDIM],
    constant const size_t a_strides[NDIM],
    constant const size_t b_strides[NDIM],
    constant const size_t c_strides[NDIM]) {
  uint3 loc = {
      static_cast<uint>(
          elem.x * a_strides[NDIM - 1] + elem.y * a_strides[NDIM - 2]),
      static_cast<uint>(
          elem.x * b_strides[NDIM - 1] + elem.y * b_strides[NDIM - 2]),
      static_cast<uint>(
          elem.x * c_strides[NDIM - 1] + elem.y * c_strides[NDIM - 2])};
  for (int d = NDIM - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    loc.z += l * c_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}






inline size_t ceildiv(size_t N, size_t M) {
  return (N + M - 1) / M;
}


inline float log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == Limits<float>::max) {
    return Limits<float>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline bfloat16_t log1p(bfloat16_t x) {
  float xp1 = 1.0f + static_cast<float>(x);
  if (xp1 == Limits<float>::max) {
    return Limits<bfloat16_t>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return bfloat16_t(x * (metal::log(xp1) / (xp1 - 1.0f)));
}





inline uint64_t simd_shuffle_down(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_down(bool data, uint16_t delta) {
  return simd_shuffle_down(static_cast<uint32_t>(data), delta);
}
# 10 "Source/Cmlx/mlx/mlx/backend/metal/kernels/binary.h" 2

struct Add {
  template <typename T>
  T operator()(T x, T y) {
    return x + y;
  }
};

struct Divide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
};

struct Remainder {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & !metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    return x % y;
  }
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    auto r = x % y;
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    T r = fmod(x, y);
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    return x % y;
  }
};

struct Equal {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y || (metal::isnan(x) && metal::isnan(y));
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x == y ||
        (metal::isnan(x.real) && metal::isnan(y.real) && metal::isnan(x.imag) &&
         metal::isnan(y.imag)) ||
        (x.real == y.real && metal::isnan(x.imag) && metal::isnan(y.imag)) ||
        (metal::isnan(x.real) && metal::isnan(y.real) && x.imag == y.imag);
  }
};

struct Greater {
  template <typename T>
  bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x <= y;
  }
};

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
  };
};

struct Maximum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::max(x, y);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x > y ? x : y;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x > y ? x : y;
  }
};

struct Minimum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::min(x, y);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x < y ? x : y;
  }
};

struct Multiply {
  template <typename T>
  T operator()(T x, T y) {
    return x * y;
  }
};

struct NotEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x != y;
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x.real != y.real || x.imag != y.imag;
  }
};

struct Power {
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T base, T exp) {
    return metal::pow(base, exp);
  }

  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T base, T exp) {
    T res = 1;
    while (exp) {
      if (exp & 1) {
        res *= base;
      }
      exp >>= 1;
      base *= base;
    }
    return res;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    auto x_theta = metal::atan(x.imag / x.real);
    auto x_ln_r = 0.5 * metal::log(x.real * x.real + x.imag * x.imag);
    auto mag = metal::exp(y.real * x_ln_r - y.imag * x_theta);
    auto phase = y.imag * x_ln_r + y.real * x_theta;
    return {mag * metal::cos(phase), mag * metal::sin(phase)};
  }
};

struct Subtract {
  template <typename T>
  T operator()(T x, T y) {
    return x - y;
  }
};

struct LogicalAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x && y;
  };
};

struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  };
};
# 4 "Source/Cmlx/mlx/mlx/backend/metal/kernels/compiled_preamble.h" 2
# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/ternary.h" 1




struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};
# 5 "Source/Cmlx/mlx/mlx/backend/metal/kernels/compiled_preamble.h" 2
# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/unary.h" 1








# 1 "Source/Cmlx/mlx/mlx/backend/metal/kernels/erf.h" 1
# 12 "Source/Cmlx/mlx/mlx/backend/metal/kernels/erf.h"
float erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {

    r = metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f);
    u = metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f);
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f);
    r = metal::fma(r, t, -6.34846687e-1f);
    r = metal::fma(r, t, -1.28717512e-1f);
    r = metal::fma(r, t, -t);

    r = 1.0f - metal::exp(r);
    r = metal::copysign(r, a);
  } else {

    r = -5.96761703e-4f;
    r = metal::fma(r, s, 4.99119423e-3f);
    r = metal::fma(r, s, -2.67681349e-2f);
    r = metal::fma(r, s, 1.12819925e-1f);
    r = metal::fma(r, s, -3.76125336e-1f);
    r = metal::fma(r, s, 1.28379166e-1f);
    r = metal::fma(r, a, a);
  }
  return r;
}

float erfinv(float a) {
  auto t = metal::fma(a, 0.0f - a, 1.0f);
  t = metal::log(t);
  float p;
  if (metal::abs(t) > 6.125f) {
    p = 3.03697567e-10f;
    p = metal::fma(p, t, 2.93243101e-8f);
    p = metal::fma(p, t, 1.22150334e-6f);
    p = metal::fma(p, t, 2.84108955e-5f);
    p = metal::fma(p, t, 3.93552968e-4f);
    p = metal::fma(p, t, 3.02698812e-3f);
    p = metal::fma(p, t, 4.83185798e-3f);
    p = metal::fma(p, t, -2.64646143e-1f);
    p = metal::fma(p, t, 8.40016484e-1f);
  } else {
    p = 5.43877832e-9f;
    p = metal::fma(p, t, 1.43285448e-7f);
    p = metal::fma(p, t, 1.22774793e-6f);
    p = metal::fma(p, t, 1.12963626e-7f);
    p = metal::fma(p, t, -5.61530760e-5f);
    p = metal::fma(p, t, -1.47697632e-4f);
    p = metal::fma(p, t, 2.31468678e-3f);
    p = metal::fma(p, t, 1.15392581e-2f);
    p = metal::fma(p, t, -2.32015476e-1f);
    p = metal::fma(p, t, 8.86226892e-1f);
  }
  return a * p;
}
# 10 "Source/Cmlx/mlx/mlx/backend/metal/kernels/unary.h" 2


namespace {
constant float inf = metal::numeric_limits<float>::infinity();
}

struct Abs {
  template <typename T>
  T operator()(T x) {
    return metal::abs(x);
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
  template <>
  complex64_t operator()(complex64_t x) {
    return {metal::precise::sqrt(x.real * x.real + x.imag * x.imag), 0};
  };
};

struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::acos(x);
  };
};

struct ArcCosh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::acosh(x);
  };
};

struct ArcSin {
  template <typename T>
  T operator()(T x) {
    return metal::precise::asin(x);
  };
};

struct ArcSinh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::asinh(x);
  };
};

struct ArcTan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::atan(x);
  };
};

struct ArcTanh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::atanh(x);
  };
};

struct Ceil {
  template <typename T>
  T operator()(T x) {
    return metal::ceil(x);
  };
  template <>
  int8_t operator()(int8_t x) {
    return x;
  };
  template <>
  int16_t operator()(int16_t x) {
    return x;
  };
  template <>
  int32_t operator()(int32_t x) {
    return x;
  };
  template <>
  int64_t operator()(int64_t x) {
    return x;
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cos(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::cos(x.real) * metal::precise::cosh(x.imag),
        -metal::precise::sin(x.real) * metal::precise::sinh(x.imag)};
  };
};

struct Cosh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cosh(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::cosh(x.real) * metal::precise::cos(x.imag),
        metal::precise::sinh(x.real) * metal::precise::sin(x.imag)};
  };
};

struct Erf {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(erf(static_cast<float>(x)));
  };
};

struct ErfInv {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(erfinv(static_cast<float>(x)));
  };
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    return metal::precise::exp(x);
  };
  template <>
  complex64_t operator()(complex64_t x) {
    auto m = metal::precise::exp(x.real);
    return {m * metal::precise::cos(x.imag), m * metal::precise::sin(x.imag)};
  }
};

struct Floor {
  template <typename T>
  T operator()(T x) {
    return metal::floor(x);
  };
  template <>
  int8_t operator()(int8_t x) {
    return x;
  };
  template <>
  int16_t operator()(int16_t x) {
    return x;
  };
  template <>
  int32_t operator()(int32_t x) {
    return x;
  };
  template <>
  int64_t operator()(int64_t x) {
    return x;
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
};

struct Log {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log(x);
  };
};

struct Log2 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log2(x);
  };
};

struct Log10 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log10(x);
  };
};

struct Log1p {
  template <typename T>
  T operator()(T x) {
    return log1p(x);
  };
};

struct LogicalNot {
  template <typename T>
  T operator()(T x) {
    return !x;
  };
};

struct Negative {
  template <typename T>
  T operator()(T x) {
    return -x;
  };
};

struct Round {
  template <typename T>
  T operator()(T x) {
    return metal::rint(x);
  };
  template <>
  complex64_t operator()(complex64_t x) {
    return {metal::rint(x.real), metal::rint(x.imag)};
  };
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x != 0;
  };
};

struct Sin {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sin(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::sin(x.real) * metal::precise::cosh(x.imag),
        metal::precise::cos(x.real) * metal::precise::sinh(x.imag)};
  };
};

struct Sinh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sinh(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::sinh(x.real) * metal::precise::cos(x.imag),
        metal::precise::cosh(x.real) * metal::precise::sin(x.imag)};
  };
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  };
};

struct Sqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sqrt(x);
  };
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::rsqrt(x);
  };
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tan(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    float tan_a = metal::precise::tan(x.real);
    float tanh_b = metal::precise::tanh(x.imag);
    float t1 = tan_a * tanh_b;
    float denom = 1. + t1 * t1;
    return {(tan_a - tanh_b * t1) / denom, (tanh_b + tan_a * t1) / denom};
  };
};

struct Tanh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tanh(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    float tanh_a = metal::precise::tanh(x.real);
    float tan_b = metal::precise::tan(x.imag);
    float t1 = tanh_a * tan_b;
    float denom = 1. + t1 * t1;
    return {(tanh_a + tan_b * t1) / denom, (tan_b - tanh_a * t1) / denom};
  };
};
# 6 "Source/Cmlx/mlx/mlx/backend/metal/kernels/compiled_preamble.h" 2

typedef half float16_t;
)preamble";

}

} // namespace mlx::core::metal
