namespace mlx::core::metal {

const char* utils() {
  return R"preamble(
using namespace metal;
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
constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC float operator+(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC float operator+(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC float operator+(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC float operator+(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator+(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator+(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator+(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator+(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); };;
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC float operator-(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC float operator-(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC float operator-(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC float operator-(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator-(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator-(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator-(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator-(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); };;
constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC float operator*(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC float operator*(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC float operator*(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC float operator*(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator*(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator*(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator*(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator*(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); };;
constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC float operator/(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC float operator/(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC float operator/(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC float operator/(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator/(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator/(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator/(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; constexpr METAL_FUNC _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } constexpr METAL_FUNC _MLX_BFloat16 operator/(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator>=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator>=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator<=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } constexpr METAL_FUNC bool operator<=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator==(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } constexpr METAL_FUNC bool operator==(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); };;
constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, half rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(half lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; constexpr METAL_FUNC bool operator!=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } constexpr METAL_FUNC bool operator!=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); };;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device float& operator+=( device float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread float& operator+=( thread float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup float& operator+=( threadgroup float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device float& operator-=( device float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread float& operator-=( thread float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup float& operator-=( threadgroup float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device float& operator*=( device float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread float& operator*=( thread float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup float& operator*=( threadgroup float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device float& operator/=( device float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread float& operator/=( thread float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, float rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup float& operator/=( threadgroup float& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device half& operator+=( device half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread half& operator+=( thread half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup half& operator+=( threadgroup half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device half& operator-=( device half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread half& operator-=( thread half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup half& operator-=( threadgroup half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device half& operator*=( device half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread half& operator*=( thread half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup half& operator*=( threadgroup half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device half& operator/=( device half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread half& operator/=( thread half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, half rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup half& operator/=( threadgroup half& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int16_t& operator+=( device int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int16_t& operator+=( thread int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int16_t& operator+=( threadgroup int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int16_t& operator-=( device int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int16_t& operator-=( thread int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int16_t& operator-=( threadgroup int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int16_t& operator*=( device int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int16_t& operator*=( thread int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int16_t& operator*=( threadgroup int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int16_t& operator/=( device int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int16_t& operator/=( thread int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, int16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int16_t& operator/=( threadgroup int16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int32_t& operator+=( device int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int32_t& operator+=( thread int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int32_t& operator+=( threadgroup int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int32_t& operator-=( device int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int32_t& operator-=( thread int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int32_t& operator-=( threadgroup int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int32_t& operator*=( device int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int32_t& operator*=( thread int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int32_t& operator*=( threadgroup int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int32_t& operator/=( device int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int32_t& operator/=( thread int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, int32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int32_t& operator/=( threadgroup int32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int64_t& operator+=( device int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int64_t& operator+=( thread int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int64_t& operator+=( threadgroup int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int64_t& operator-=( device int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int64_t& operator-=( thread int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int64_t& operator-=( threadgroup int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int64_t& operator*=( device int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int64_t& operator*=( thread int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int64_t& operator*=( threadgroup int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device int64_t& operator/=( device int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread int64_t& operator/=( thread int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, int64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup int64_t& operator/=( threadgroup int64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint16_t& operator+=( device uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint16_t& operator+=( thread uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint16_t& operator+=( threadgroup uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint16_t& operator-=( device uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint16_t& operator-=( thread uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint16_t& operator-=( threadgroup uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint16_t& operator*=( device uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint16_t& operator*=( thread uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint16_t& operator*=( threadgroup uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint16_t& operator/=( device uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint16_t& operator/=( thread uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, uint16_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint16_t& operator/=( threadgroup uint16_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint32_t& operator+=( device uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint32_t& operator+=( thread uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint32_t& operator+=( threadgroup uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint32_t& operator-=( device uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint32_t& operator-=( thread uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint32_t& operator-=( threadgroup uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint32_t& operator*=( device uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint32_t& operator*=( thread uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint32_t& operator*=( threadgroup uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint32_t& operator/=( device uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint32_t& operator/=( thread uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, uint32_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint32_t& operator/=( threadgroup uint32_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint64_t& operator+=( device uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint64_t& operator+=( thread uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint64_t& operator+=( threadgroup uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint64_t& operator-=( device uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint64_t& operator-=( thread uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint64_t& operator-=( threadgroup uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint64_t& operator*=( device uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint64_t& operator*=( thread uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint64_t& operator*=( threadgroup uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };; constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC device uint64_t& operator/=( device uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC thread uint64_t& operator/=( thread uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, uint64_t rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; } constexpr METAL_FUNC threadgroup uint64_t& operator/=( threadgroup uint64_t& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;;
constexpr METAL_FUNC device _MLX_BFloat16& operator+=( device _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator+=( thread _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator+=( threadgroup _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) + static_cast<float>(rhs); return lhs; };;
constexpr METAL_FUNC device _MLX_BFloat16& operator-=( device _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator-=( thread _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator-=( threadgroup _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) - static_cast<float>(rhs); return lhs; };;
constexpr METAL_FUNC device _MLX_BFloat16& operator*=( device _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator*=( thread _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator*=( threadgroup _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) * static_cast<float>(rhs); return lhs; };;
constexpr METAL_FUNC device _MLX_BFloat16& operator/=( device _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC thread _MLX_BFloat16& operator/=( thread _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; }; constexpr METAL_FUNC threadgroup _MLX_BFloat16& operator/=( threadgroup _MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs = static_cast<float>(lhs) / static_cast<float>(rhs); return lhs; };;
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
namespace metal {
METAL_FUNC bfloat16_t abs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t acos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acos(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t acosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acosh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t asin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asin(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t asinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asinh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t atan(bfloat16_t y_over_x) { return static_cast<bfloat16_t>( __metal_atan(static_cast<float>(y_over_x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t atan2(bfloat16_t y, bfloat16_t x) { return static_cast<bfloat16_t>( __metal_atan2(static_cast<float>(y), static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t atanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_atanh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t ceil(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_ceil(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t cos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cos(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t cosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cosh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t cospi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cospi(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t divide(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_divide(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t exp(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t exp10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp10(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t exp2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp2(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fabs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fdim(bfloat16_t x, bfloat16_t y) { float t = static_cast<float>(x - y); return static_cast<bfloat16_t>(select(t, float(0), t < float(0) || x == y)); } METAL_FUNC bfloat16_t floor(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_floor(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fma(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fma( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))); } METAL_FUNC bfloat16_t fmax(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmax3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmedian3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmin(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmin3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fmod(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmod(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t fract(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fract(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t frexp(bfloat16_t x, thread int& exp) { return static_cast<bfloat16_t>(__metal_frexp(static_cast<float>(x), &exp)); } METAL_FUNC bfloat16_t ldexp(bfloat16_t x, int k) { return static_cast<bfloat16_t>(__metal_ldexp(static_cast<float>(x), k, __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t log(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t log10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log10(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t log2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log2(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t max(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t max3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t median3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t min(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t min3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t nextafter(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_nextafter(static_cast<float>(x), static_cast<float>(y))); } METAL_FUNC bfloat16_t pow(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_pow(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t powr(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_powr(static_cast<float>(x), static_cast<float>(y), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t rint(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rint(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t round(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_round(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t rsqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rsqrt(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t sin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sin(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t sinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t sinpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinpi(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t sqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sqrt(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t tan(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tan(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t tanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanh(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t tanpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanpi(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); } METAL_FUNC bfloat16_t trunc(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_trunc(static_cast<float>(x), __METAL_MAYBE_FAST_MATH__)); };
namespace fast {
METAL_FUNC bfloat16_t abs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t acos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acos(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t acosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acosh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t asin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asin(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t asinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asinh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t atan(bfloat16_t y_over_x) { return static_cast<bfloat16_t>( __metal_atan(static_cast<float>(y_over_x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t atan2(bfloat16_t y, bfloat16_t x) { return static_cast<bfloat16_t>( __metal_atan2(static_cast<float>(y), static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t atanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_atanh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t ceil(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_ceil(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t cos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cos(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t cosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cosh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t cospi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cospi(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t divide(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_divide(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t exp(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t exp10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp10(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t exp2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp2(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fabs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fdim(bfloat16_t x, bfloat16_t y) { float t = static_cast<float>(x - y); return static_cast<bfloat16_t>(select(t, float(0), t < float(0) || x == y)); } METAL_FUNC bfloat16_t floor(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_floor(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fma(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fma( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))); } METAL_FUNC bfloat16_t fmax(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmax3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmedian3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmin(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmin3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fmod(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmod(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t fract(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fract(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t frexp(bfloat16_t x, thread int& exp) { return static_cast<bfloat16_t>(__metal_frexp(static_cast<float>(x), &exp)); } METAL_FUNC bfloat16_t ldexp(bfloat16_t x, int k) { return static_cast<bfloat16_t>(__metal_ldexp(static_cast<float>(x), k, __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t log(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t log10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log10(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t log2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log2(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t max(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t max3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t median3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t min(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t min3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t nextafter(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_nextafter(static_cast<float>(x), static_cast<float>(y))); } METAL_FUNC bfloat16_t pow(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_pow(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t powr(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_powr(static_cast<float>(x), static_cast<float>(y), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t rint(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rint(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t round(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_round(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t rsqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rsqrt(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t sin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sin(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t sinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t sinpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinpi(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t sqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sqrt(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t tan(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tan(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t tanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanh(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t tanpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanpi(static_cast<float>(x), __METAL_FAST_MATH__)); } METAL_FUNC bfloat16_t trunc(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_trunc(static_cast<float>(x), __METAL_FAST_MATH__)); };
}
namespace precise {
METAL_FUNC bfloat16_t abs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t acos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acos(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t acosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_acosh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t asin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asin(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t asinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_asinh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t atan(bfloat16_t y_over_x) { return static_cast<bfloat16_t>( __metal_atan(static_cast<float>(y_over_x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t atan2(bfloat16_t y, bfloat16_t x) { return static_cast<bfloat16_t>( __metal_atan2(static_cast<float>(y), static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t atanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_atanh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t ceil(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_ceil(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t cos(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cos(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t cosh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cosh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t cospi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_cospi(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t divide(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_divide(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t exp(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t exp10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp10(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t exp2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_exp2(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fabs(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fdim(bfloat16_t x, bfloat16_t y) { float t = static_cast<float>(x - y); return static_cast<bfloat16_t>(select(t, float(0), t < float(0) || x == y)); } METAL_FUNC bfloat16_t floor(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_floor(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fma(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fma( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))); } METAL_FUNC bfloat16_t fmax(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmax3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmedian3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmin(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmin3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fmod(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmod(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t fract(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_fract(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t frexp(bfloat16_t x, thread int& exp) { return static_cast<bfloat16_t>(__metal_frexp(static_cast<float>(x), &exp)); } METAL_FUNC bfloat16_t ldexp(bfloat16_t x, int k) { return static_cast<bfloat16_t>(__metal_ldexp(static_cast<float>(x), k, __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t log(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t log10(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log10(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t log2(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_log2(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t max(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmax(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t max3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmax3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t median3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmedian3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t min(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_fmin(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t min3(bfloat16_t x, bfloat16_t y, bfloat16_t z) { return static_cast<bfloat16_t>(__metal_fmin3( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t nextafter(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_nextafter(static_cast<float>(x), static_cast<float>(y))); } METAL_FUNC bfloat16_t pow(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_pow(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t powr(bfloat16_t x, bfloat16_t y) { return static_cast<bfloat16_t>( __metal_powr(static_cast<float>(x), static_cast<float>(y), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t rint(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rint(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t round(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_round(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t rsqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_rsqrt(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t sin(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sin(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t sinh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t sinpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sinpi(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t sqrt(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_sqrt(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t tan(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tan(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t tanh(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanh(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t tanpi(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_tanpi(static_cast<float>(x), __METAL_PRECISE_MATH__)); } METAL_FUNC bfloat16_t trunc(bfloat16_t x) { return static_cast<bfloat16_t>(__metal_trunc(static_cast<float>(x), __METAL_PRECISE_MATH__)); };
}
}
namespace metal {
METAL_FUNC bfloat16_t simd_broadcast(bfloat16_t data, ushort broadcast_lane_id) { return _MLX_BFloat16(__metal_simd_broadcast(data.bits_, broadcast_lane_id), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle(bfloat16_t data, ushort simd_lane_id) { return _MLX_BFloat16(__metal_simd_shuffle(data.bits_, simd_lane_id), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_down( bfloat16_t data, bfloat16_t filling_data, ushort delta, ushort modulo) { return _MLX_BFloat16(__metal_simd_shuffle_and_fill_down( data.bits_, filling_data.bits_, delta, modulo), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_down( bfloat16_t data, bfloat16_t filling_data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_and_fill_down( data.bits_, filling_data.bits_, delta, __metal_get_simdgroup_size(ushort())), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_up( bfloat16_t data, bfloat16_t filling_data, ushort delta, ushort modulo) { return _MLX_BFloat16(__metal_simd_shuffle_and_fill_up( data.bits_, filling_data.bits_, delta, modulo), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_up( bfloat16_t data, bfloat16_t filling_data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_and_fill_up( data.bits_, filling_data.bits_, delta, __metal_get_simdgroup_size(ushort())), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_down(bfloat16_t data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_down(data.bits_, delta), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_rotate_down(bfloat16_t data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_rotate_down(data.bits_, delta), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_rotate_up(bfloat16_t data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_rotate_up(data.bits_, delta), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_up(bfloat16_t data, ushort delta) { return _MLX_BFloat16(__metal_simd_shuffle_up(data.bits_, delta), _MLX_BFloat16::bits_to_bfloat()); } METAL_FUNC bfloat16_t simd_shuffle_xor(bfloat16_t data, ushort mask) { return _MLX_BFloat16(__metal_simd_shuffle_xor(data.bits_, mask), _MLX_BFloat16::bits_to_bfloat()); };
METAL_FUNC bfloat16_t simd_max(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_max(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_min(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_min(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_prefix_exclusive_product(bfloat16_t data) { return static_cast<bfloat16_t>( __metal_simd_prefix_exclusive_product(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_prefix_exclusive_sum(bfloat16_t data) { return static_cast<bfloat16_t>( __metal_simd_prefix_exclusive_sum(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_prefix_inclusive_product(bfloat16_t data) { return static_cast<bfloat16_t>( __metal_simd_prefix_inclusive_product(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_prefix_inclusive_sum(bfloat16_t data) { return static_cast<bfloat16_t>( __metal_simd_prefix_inclusive_sum(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_product(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_product(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_sum(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_sum(static_cast<float>(data))); } METAL_FUNC bfloat16_t simd_xor(bfloat16_t data) { return static_cast<bfloat16_t>(__metal_simd_xor(static_cast<float>(data))); };
}
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
  constexpr complex64_t(float real, float imag) : real(real), imag(imag) {};
  constexpr complex64_t() : real(0), imag(0) {};
  constexpr complex64_t() threadgroup : real(0), imag(0) {};
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
static constant constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static constant constexpr int REDUCE_N_READS = 4;
static constant constexpr int REDUCE_N_WRITES = 4;
static constant constexpr int SOFTMAX_N_READS = 4;
static constant constexpr int RMS_N_READS = 4;
static constant constexpr int RMS_LOOPED_LIMIT = 4096;

typedef half float16_t;
template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};
template <> struct Limits<uint8_t> { static constexpr constant uint8_t max = metal::numeric_limits<uint8_t>::max(); static constexpr constant uint8_t min = metal::numeric_limits<uint8_t>::min(); static constexpr constant uint8_t finite_max = metal::numeric_limits<uint8_t>::max(); static constexpr constant uint8_t finite_min = metal::numeric_limits<uint8_t>::min(); };;
template <> struct Limits<uint16_t> { static constexpr constant uint16_t max = metal::numeric_limits<uint16_t>::max(); static constexpr constant uint16_t min = metal::numeric_limits<uint16_t>::min(); static constexpr constant uint16_t finite_max = metal::numeric_limits<uint16_t>::max(); static constexpr constant uint16_t finite_min = metal::numeric_limits<uint16_t>::min(); };;
template <> struct Limits<uint32_t> { static constexpr constant uint32_t max = metal::numeric_limits<uint32_t>::max(); static constexpr constant uint32_t min = metal::numeric_limits<uint32_t>::min(); static constexpr constant uint32_t finite_max = metal::numeric_limits<uint32_t>::max(); static constexpr constant uint32_t finite_min = metal::numeric_limits<uint32_t>::min(); };;
template <> struct Limits<uint64_t> { static constexpr constant uint64_t max = metal::numeric_limits<uint64_t>::max(); static constexpr constant uint64_t min = metal::numeric_limits<uint64_t>::min(); static constexpr constant uint64_t finite_max = metal::numeric_limits<uint64_t>::max(); static constexpr constant uint64_t finite_min = metal::numeric_limits<uint64_t>::min(); };;
template <> struct Limits<int8_t> { static constexpr constant int8_t max = metal::numeric_limits<int8_t>::max(); static constexpr constant int8_t min = metal::numeric_limits<int8_t>::min(); static constexpr constant int8_t finite_max = metal::numeric_limits<int8_t>::max(); static constexpr constant int8_t finite_min = metal::numeric_limits<int8_t>::min(); };;
template <> struct Limits<int16_t> { static constexpr constant int16_t max = metal::numeric_limits<int16_t>::max(); static constexpr constant int16_t min = metal::numeric_limits<int16_t>::min(); static constexpr constant int16_t finite_max = metal::numeric_limits<int16_t>::max(); static constexpr constant int16_t finite_min = metal::numeric_limits<int16_t>::min(); };;
template <> struct Limits<int32_t> { static constexpr constant int32_t max = metal::numeric_limits<int32_t>::max(); static constexpr constant int32_t min = metal::numeric_limits<int32_t>::min(); static constexpr constant int32_t finite_max = metal::numeric_limits<int32_t>::max(); static constexpr constant int32_t finite_min = metal::numeric_limits<int32_t>::min(); };;
template <> struct Limits<int64_t> { static constexpr constant int64_t max = metal::numeric_limits<int64_t>::max(); static constexpr constant int64_t min = metal::numeric_limits<int64_t>::min(); static constexpr constant int64_t finite_max = metal::numeric_limits<int64_t>::max(); static constexpr constant int64_t finite_min = metal::numeric_limits<int64_t>::min(); };;
template <> struct Limits<half> { static constexpr constant half max = metal::numeric_limits<half>::infinity(); static constexpr constant half min = -metal::numeric_limits<half>::infinity(); static constexpr constant half finite_max = metal::numeric_limits<half>::max(); static constexpr constant half finite_min = -metal::numeric_limits<half>::max(); };;
template <> struct Limits<float> { static constexpr constant float max = metal::numeric_limits<float>::infinity(); static constexpr constant float min = -metal::numeric_limits<float>::infinity(); static constexpr constant float finite_max = metal::numeric_limits<float>::max(); static constexpr constant float finite_min = -metal::numeric_limits<float>::max(); };;
template <> struct Limits<bfloat16_t> { static constexpr constant bfloat16_t max = metal::numeric_limits<bfloat16_t>::infinity(); static constexpr constant bfloat16_t min = -metal::numeric_limits<bfloat16_t>::infinity(); static constexpr constant bfloat16_t finite_max = metal::numeric_limits<bfloat16_t>::max(); static constexpr constant bfloat16_t finite_min = -metal::numeric_limits<bfloat16_t>::max(); };;
template <>
struct Limits<bool> {
  static constexpr constant bool max = true;
  static constexpr constant bool min = false;
};
template <>
struct Limits<complex64_t> {
  static constexpr constant complex64_t max = complex64_t(
      metal::numeric_limits<float>::infinity(),
      metal::numeric_limits<float>::infinity());
  static constexpr constant complex64_t min = complex64_t(
      -metal::numeric_limits<float>::infinity(),
      -metal::numeric_limits<float>::infinity());
};
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
    stride_t elem,
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
template <typename stride_t>
METAL_FUNC ulong2 elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const stride_t* a_strides,
    constant const stride_t* b_strides,
    int ndim) {
  ulong2 loc = {
      ulong(elem.x * a_strides[ndim - 1] + elem.y * a_strides[ndim - 2]),
      ulong(elem.x * b_strides[ndim - 1] + elem.y * b_strides[ndim - 2])};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}
METAL_FUNC ulong3 elem_to_loc_3_nd(
    uint3 elem,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    int ndim) {
  ulong3 loc = {
      elem.x * a_strides[ndim - 1] + elem.y * a_strides[ndim - 2],
      elem.x * b_strides[ndim - 1] + elem.y * b_strides[ndim - 2],
      elem.x * c_strides[ndim - 1] + elem.y * c_strides[ndim - 2]};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * a_strides[d];
    loc.y += l * b_strides[d];
    loc.z += l * c_strides[d];
    elem.z /= shape[d];
  }
  return loc;
}
template <int dim, typename offset_t = size_t>
struct looped_elem_to_loc {
  looped_elem_to_loc<dim - 1, offset_t> inner_looper;
  offset_t offset{0};
  int index{0};
  void next(const constant int* shape, const constant size_t* strides) {
    index++;
    offset += strides[dim - 1];
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }
  void next(int n, const constant int* shape, const constant size_t* strides) {
    index += n;
    offset += n * strides[dim - 1];
    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }
  offset_t
  location(offset_t, const constant int*, const constant size_t*, int) {
    return offset;
  }
};
template <typename offset_t>
struct looped_elem_to_loc<1, offset_t> {
  offset_t offset{0};
  void next(const constant int*, const constant size_t* strides) {
    offset += strides[0];
  }
  void next(int n, const constant int*, const constant size_t* strides) {
    offset += n * strides[0];
  }
  offset_t
  location(offset_t, const constant int*, const constant size_t*, int) {
    return offset;
  }
};
template <typename offset_t>
struct looped_elem_to_loc<0, offset_t> {
  void next(const constant int*, const constant size_t*) {}
  void next(int, const constant int*, const constant size_t*) {}
  offset_t location(
      offset_t idx,
      const constant int* shape,
      const constant size_t* strides,
      int ndim) {
    return elem_to_loc(idx, shape, strides, ndim);
  }
};
template <typename T, typename U>
inline T ceildiv(T N, U M) {
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
inline complex64_t simd_shuffle_down(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_down(data.real, delta), simd_shuffle_down(data.imag, delta));
}
inline uint64_t simd_shuffle_up(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}
inline int64_t simd_shuffle_up(int64_t data, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}
inline bool simd_shuffle_up(bool data, uint16_t delta) {
  return simd_shuffle_up(static_cast<uint32_t>(data), delta);
}
inline complex64_t simd_shuffle_up(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_up(data.real, delta), simd_shuffle_up(data.imag, delta));
}
inline uint64_t
simd_shuffle_and_fill_up(uint64_t data, uint64_t filling, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}
inline int64_t
simd_shuffle_and_fill_up(int64_t data, int64_t filling, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}
inline bool simd_shuffle_and_fill_up(bool data, bool filling, uint16_t delta) {
  return simd_shuffle_and_fill_up(
      static_cast<uint32_t>(data), static_cast<uint32_t>(filling), delta);
}
inline complex64_t simd_shuffle_and_fill_up(
    complex64_t data,
    complex64_t filling,
    uint16_t delta) {
  return complex64_t(
      simd_shuffle_and_fill_up(data.real, filling.real, delta),
      simd_shuffle_and_fill_up(data.imag, filling.imag, delta));
}
inline uint64_t simd_shuffle(uint64_t data, uint16_t lane) {
  return as_type<uint64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}
inline int64_t simd_shuffle(int64_t data, uint16_t lane) {
  return as_type<int64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}
inline bool simd_shuffle(bool data, uint16_t lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}
inline complex64_t simd_shuffle(complex64_t data, uint16_t lane) {
  return complex64_t(
      simd_shuffle(data.real, lane), simd_shuffle(data.imag, lane));
}
)preamble";
}

} // namespace mlx::core::metal
