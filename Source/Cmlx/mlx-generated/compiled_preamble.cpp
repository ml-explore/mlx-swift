const char* get_kernel_preamble() {
return R"preamble(
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
#include <arm_fp16.h>
#endif
namespace mlx::core {
using ::float16_t;
}
namespace mlx::core {
namespace {
union float_bits_bf16 {
  float f;
  uint32_t u;
};
}
struct _MLX_BFloat16 {
  uint16_t bits_;
  _MLX_BFloat16() = default;
  _MLX_BFloat16(_MLX_BFloat16 const&) = default;
  _MLX_BFloat16& operator=(std::vector<bool>::reference x) {
    bits_ = x;
    return *this;
  }
  _MLX_BFloat16& operator=(const float& x) {
    return (*this = _MLX_BFloat16(x));
  }
  _MLX_BFloat16(const float& x) {
    if (std::isnan(x)) {
      bits_ = 0x7FC0;
    } else {
      float_bits_bf16 in;
      in.f = x;
      in.u += (in.u >> 16 & 1) + uint32_t(0x7FFF);
      bits_ = in.u >> 16;
    }
  }
  operator float() const {
    float_bits_bf16 out;
    out.u = ((uint32_t)bits_) << 16;
    return out.f;
  }
};
inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline float operator+(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline float operator+(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline double operator+(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) + static_cast<double>(rhs); } inline double operator+(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) + static_cast<double>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, bool rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(bool lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); };;
inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline float operator-(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline float operator-(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline double operator-(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) - static_cast<double>(rhs); } inline double operator-(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) - static_cast<double>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, bool rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(bool lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); };;
inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline float operator*(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline float operator*(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline double operator*(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) * static_cast<double>(rhs); } inline double operator*(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) * static_cast<double>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, bool rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(bool lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); };;
inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline float operator/(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline float operator/(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline double operator/(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) / static_cast<double>(rhs); } inline double operator/(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) / static_cast<double>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, bool rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(bool lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); };;
inline bool operator>(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) > static_cast<double>(rhs); } inline bool operator>(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) > static_cast<double>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); };;
inline bool operator<(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) < static_cast<double>(rhs); } inline bool operator<(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) < static_cast<double>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); };;
inline bool operator>=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) >= static_cast<double>(rhs); } inline bool operator>=(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) >= static_cast<double>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); };;
inline bool operator<=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) <= static_cast<double>(rhs); } inline bool operator<=(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) <= static_cast<double>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); };;
inline bool operator==(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) == static_cast<double>(rhs); } inline bool operator==(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) == static_cast<double>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); };;
inline bool operator!=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) != static_cast<double>(rhs); } inline bool operator!=(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) != static_cast<double>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); };;
inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs) {
  return -static_cast<float>(lhs);
}
inline _MLX_BFloat16& operator+=(_MLX_BFloat16& lhs, const float& rhs) { lhs = lhs + rhs; return lhs; } inline float& operator+=(float& lhs, _MLX_BFloat16 rhs) { lhs = lhs + rhs; return lhs; };
inline _MLX_BFloat16& operator-=(_MLX_BFloat16& lhs, const float& rhs) { lhs = lhs - rhs; return lhs; } inline float& operator-=(float& lhs, _MLX_BFloat16 rhs) { lhs = lhs - rhs; return lhs; };
inline _MLX_BFloat16& operator*=(_MLX_BFloat16& lhs, const float& rhs) { lhs = lhs * rhs; return lhs; } inline float& operator*=(float& lhs, _MLX_BFloat16 rhs) { lhs = lhs * rhs; return lhs; };
inline _MLX_BFloat16& operator/=(_MLX_BFloat16& lhs, const float& rhs) { lhs = lhs / rhs; return lhs; } inline float& operator/=(float& lhs, _MLX_BFloat16 rhs) { lhs = lhs / rhs; return lhs; };
inline _MLX_BFloat16 operator|(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ | rhs.bits_; return out; } inline _MLX_BFloat16 operator|(_MLX_BFloat16 lhs, uint16_t rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ | rhs; return out; } inline _MLX_BFloat16 operator|(uint16_t lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs | rhs.bits_; return out; };
inline _MLX_BFloat16 operator&(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ & rhs.bits_; return out; } inline _MLX_BFloat16 operator&(_MLX_BFloat16 lhs, uint16_t rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ & rhs; return out; } inline _MLX_BFloat16 operator&(uint16_t lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs & rhs.bits_; return out; };
inline _MLX_BFloat16 operator^(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ ^ rhs.bits_; return out; } inline _MLX_BFloat16 operator^(_MLX_BFloat16 lhs, uint16_t rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ ^ rhs; return out; } inline _MLX_BFloat16 operator^(uint16_t lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs ^ rhs.bits_; return out; };
inline _MLX_BFloat16& operator|=(_MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs.bits_ = lhs.bits_ | rhs.bits_; return lhs; } inline _MLX_BFloat16& operator|=(_MLX_BFloat16& lhs, uint16_t rhs) { lhs.bits_ = lhs.bits_ | rhs; return lhs; };
inline _MLX_BFloat16& operator&=(_MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs.bits_ = lhs.bits_ & rhs.bits_; return lhs; } inline _MLX_BFloat16& operator&=(_MLX_BFloat16& lhs, uint16_t rhs) { lhs.bits_ = lhs.bits_ & rhs; return lhs; };
inline _MLX_BFloat16& operator^=(_MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs.bits_ = lhs.bits_ ^ rhs.bits_; return lhs; } inline _MLX_BFloat16& operator^=(_MLX_BFloat16& lhs, uint16_t rhs) { lhs.bits_ = lhs.bits_ ^ rhs; return lhs; };
}
namespace mlx::core {
typedef struct _MLX_BFloat16 bfloat16_t;
}
namespace mlx::core {
inline float operator+(float16_t lhs, bfloat16_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline float operator+(bfloat16_t lhs, float16_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }
inline float operator-(float16_t lhs, bfloat16_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline float operator-(bfloat16_t lhs, float16_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }
inline float operator*(float16_t lhs, bfloat16_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline float operator*(bfloat16_t lhs, float16_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }
inline float operator/(float16_t lhs, bfloat16_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline float operator/(bfloat16_t lhs, float16_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }
}
namespace mlx::core {
struct complex64_t;
struct complex128_t;
template <typename T>
inline constexpr bool can_convert_to_complex128 =
    !std::is_same_v<T, complex128_t> && std::is_convertible_v<T, double>;
struct complex128_t : public std::complex<double> {
  complex128_t() : std::complex<double>() {};
  complex128_t(double v, double u) : std::complex<double>(v, u) {};
  complex128_t(std::complex<double> v) : std::complex<double>(v) {};
  template <
      typename T,
      typename = typename std::enable_if<can_convert_to_complex128<T>>::type>
  complex128_t(T x) : std::complex<double>(x){};
  operator float() const {
    return real();
  };
};
template <typename T>
inline constexpr bool can_convert_to_complex64 =
    !std::is_same_v<T, complex64_t> && std::is_convertible_v<T, float>;
struct complex64_t : public std::complex<float> {
  complex64_t() : std::complex<float>() {};
  complex64_t(float v, float u) : std::complex<float>(v, u) {};
  complex64_t(std::complex<float> v) : std::complex<float>(v) {};
  template <
      typename T,
      typename = typename std::enable_if<can_convert_to_complex64<T>>::type>
  complex64_t(T x) : std::complex<float>(x){};
  operator float() const {
    return real();
  };
};
inline bool operator>=(const complex64_t& a, const complex64_t& b) {
  return (a.real() > b.real()) ||
      (a.real() == b.real() && a.imag() >= b.imag());
}
inline bool operator>(const complex64_t& a, const complex64_t& b) {
  return (a.real() > b.real()) || (a.real() == b.real() && a.imag() > b.imag());
}
inline complex64_t operator%(complex64_t a, complex64_t b) {
  auto real = a.real() - (b.real() * static_cast<int64_t>(a.real() / b.real()));
  auto imag = a.imag() - (b.imag() * static_cast<int64_t>(a.imag() / b.imag()));
  if (real != 0 && ((real < 0) != (b.real() < 0)))
    real += b.real();
  if (imag != 0 && ((imag < 0) != (b.imag() < 0)))
    imag += b.imag();
  return {real, imag};
}
inline bool operator<=(const complex64_t& a, const complex64_t& b) {
  return operator>=(b, a);
}
inline bool operator<(const complex64_t& a, const complex64_t& b) {
  return operator>(b, a);
}
inline complex64_t operator-(const complex64_t& v) {
  return -static_cast<std::complex<float>>(v);
}
inline complex64_t operator+(const std::complex<float>& x, const complex64_t& y) { return x + static_cast<std::complex<float>>(y); } inline complex64_t operator+(const complex64_t& x, const std::complex<float>& y) { return static_cast<std::complex<float>>(x) + y; } inline complex64_t operator+(const complex64_t& x, const complex64_t& y) { return static_cast<std::complex<float>>(x) + static_cast<std::complex<float>>(y); } inline complex64_t operator+(bool x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, bool y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(uint32_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, uint32_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(uint64_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, uint64_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(int32_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, int32_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(int64_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, int64_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(float16_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, float16_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(bfloat16_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, bfloat16_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(float x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, float y) { return x + static_cast<complex64_t>(y); }
}
namespace mlx::core::simd {
template <typename T, int N>
struct Simd;
template <typename T>
static constexpr int max_size = 1;
template <typename T>
struct Simd<T, 1> {
  static constexpr int size = 1;
  T value;
  Simd() {}
  template <typename U>
  Simd(Simd<U, 1> v) : value(v.value) {}
  template <typename U>
  Simd(U v) : value(v) {}
  T operator[](int) const {
    return value;
  }
  T& operator[](int) {
    return value;
  }
};
template <typename T, int N>
Simd<T, N> load(const T* x) {
  return *(Simd<T, N>*)x;
}
template <typename T, int N>
void store(T* dst, Simd<T, N> x) {
  if constexpr (std::is_same_v<T, bool> && N > 1) {
    x = x & 1;
  }
  *(Simd<T, N>*)dst = x;
}
template <typename, typename = void>
constexpr bool is_complex = false;
template <typename T>
constexpr bool is_complex<T, std::void_t<decltype(std::declval<T>().real())>> =
    true;
template <typename T>
Simd<T, 1> rint(Simd<T, 1> in) {
  if constexpr (is_complex<T>) {
    return Simd<T, 1>{
        T{std::rint(in.value.real()), std::rint(in.value.imag())}};
  } else {
    return Simd<T, 1>{std::rint(in.value)};
  }
}
template <typename T>
Simd<T, 1> rsqrt(Simd<T, 1> in) {
  return T(1.0) / sqrt(in);
}
template <typename T>
Simd<T, 1> recip(Simd<T, 1> in) {
  return T(1.0) / in;
}
template <typename T> Simd<T, 1> operator-(Simd<T, 1> in) { return std::negate{}(in.value); }
template <typename T> Simd<T, 1> operator!(Simd<T, 1> in) { return std::logical_not{}(in.value); }
template <typename T> Simd<T, 1> abs(Simd<T, 1> in) { return std::abs(in.value); }
template <typename T> Simd<T, 1> acos(Simd<T, 1> in) { return std::acos(in.value); }
template <typename T> Simd<T, 1> acosh(Simd<T, 1> in) { return std::acosh(in.value); }
template <typename T> Simd<T, 1> asin(Simd<T, 1> in) { return std::asin(in.value); }
template <typename T> Simd<T, 1> asinh(Simd<T, 1> in) { return std::asinh(in.value); }
template <typename T> Simd<T, 1> atan(Simd<T, 1> in) { return std::atan(in.value); }
template <typename T> Simd<T, 1> atanh(Simd<T, 1> in) { return std::atanh(in.value); }
template <typename T> Simd<T, 1> ceil(Simd<T, 1> in) { return std::ceil(in.value); }
template <typename T> Simd<T, 1> conj(Simd<T, 1> in) { return std::conj(in.value); }
template <typename T> Simd<T, 1> cosh(Simd<T, 1> in) { return std::cosh(in.value); }
template <typename T> Simd<T, 1> expm1(Simd<T, 1> in) { return std::expm1(in.value); }
template <typename T> Simd<T, 1> floor(Simd<T, 1> in) { return std::floor(in.value); }
template <typename T> Simd<T, 1> log(Simd<T, 1> in) { return std::log(in.value); }
template <typename T> Simd<T, 1> log10(Simd<T, 1> in) { return std::log10(in.value); }
template <typename T> Simd<T, 1> sinh(Simd<T, 1> in) { return std::sinh(in.value); }
template <typename T> Simd<T, 1> sqrt(Simd<T, 1> in) { return std::sqrt(in.value); }
template <typename T> Simd<T, 1> tan(Simd<T, 1> in) { return std::tan(in.value); }
template <typename T> Simd<T, 1> tanh(Simd<T, 1> in) { return std::tanh(in.value); }
template <typename T>
Simd<T, 1> log1p(Simd<T, 1> in) {
  if constexpr (is_complex<T>) {
    auto x = in.value.real();
    auto y = in.value.imag();
    auto zabs = std::abs(in.value);
    auto theta = std::atan2(y, x + 1);
    if (zabs < 0.5) {
      auto r = x * (2 + x) + y * y;
      if (r == 0) {
        return Simd<T, 1>{T{x, theta}};
      }
      return Simd<T, 1>{T{((decltype(x))(0.5)) * std::log1p(r), theta}};
    } else {
      auto z0 = std::hypot(x + 1, y);
      return Simd<T, 1>{T{std::log(z0), theta}};
    }
  } else {
    return Simd<T, 1>{std::log1p(in.value)};
  }
}
template <typename T>
Simd<T, 1> log2(Simd<T, 1> in) {
  if constexpr (is_complex<T>) {
    auto out = std::log(in.value);
    auto scale = decltype(out.real())(M_LN2);
    return Simd<T, 1>{T{out.real() / scale, out.imag() / scale}};
  } else {
    return Simd<T, 1>{std::log2(in.value)};
  }
}
template <typename T>
Simd<T, 1> operator~(Simd<T, 1> in) {
  return ~in.value;
}
template <typename T>
auto real(Simd<T, 1> in) -> Simd<decltype(std::real(in.value)), 1> {
  return std::real(in.value);
}
template <typename T>
auto imag(Simd<T, 1> in) -> Simd<decltype(std::imag(in.value)), 1> {
  return std::imag(in.value);
}
template <typename T>
Simd<bool, 1> isnan(Simd<T, 1> in) {
  return std::isnan(in.value);
}
template <typename T1, typename T2> auto operator +(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value + b.value), 1> { return a.value + b.value; } template <typename T1, typename T2> auto operator +(T1 a, Simd<T2, 1> b)->Simd<decltype(a + b.value), 1> { return a + b.value; } template <typename T1, typename T2> auto operator +(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value + b), 1> { return a.value + b; }
template <typename T1, typename T2> auto operator -(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value - b.value), 1> { return a.value - b.value; } template <typename T1, typename T2> auto operator -(T1 a, Simd<T2, 1> b)->Simd<decltype(a - b.value), 1> { return a - b.value; } template <typename T1, typename T2> auto operator -(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value - b), 1> { return a.value - b; }
template <typename T1, typename T2> auto operator *(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value * b.value), 1> { return a.value * b.value; } template <typename T1, typename T2> auto operator *(T1 a, Simd<T2, 1> b)->Simd<decltype(a * b.value), 1> { return a * b.value; } template <typename T1, typename T2> auto operator *(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value * b), 1> { return a.value * b; }
template <typename T1, typename T2> auto operator /(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value / b.value), 1> { return a.value / b.value; } template <typename T1, typename T2> auto operator /(T1 a, Simd<T2, 1> b)->Simd<decltype(a / b.value), 1> { return a / b.value; } template <typename T1, typename T2> auto operator /(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value / b), 1> { return a.value / b; }
template <typename T1, typename T2> auto operator <<(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value << b.value), 1> { return a.value << b.value; } template <typename T1, typename T2> auto operator <<(T1 a, Simd<T2, 1> b)->Simd<decltype(a << b.value), 1> { return a << b.value; } template <typename T1, typename T2> auto operator <<(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value << b), 1> { return a.value << b; }
template <typename T1, typename T2> auto operator >>(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value >> b.value), 1> { return a.value >> b.value; } template <typename T1, typename T2> auto operator >>(T1 a, Simd<T2, 1> b)->Simd<decltype(a >> b.value), 1> { return a >> b.value; } template <typename T1, typename T2> auto operator >>(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value >> b), 1> { return a.value >> b; }
template <typename T1, typename T2> auto operator |(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value | b.value), 1> { return a.value | b.value; } template <typename T1, typename T2> auto operator |(T1 a, Simd<T2, 1> b)->Simd<decltype(a | b.value), 1> { return a | b.value; } template <typename T1, typename T2> auto operator |(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value | b), 1> { return a.value | b; }
template <typename T1, typename T2> auto operator ^(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value ^ b.value), 1> { return a.value ^ b.value; } template <typename T1, typename T2> auto operator ^(T1 a, Simd<T2, 1> b)->Simd<decltype(a ^ b.value), 1> { return a ^ b.value; } template <typename T1, typename T2> auto operator ^(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value ^ b), 1> { return a.value ^ b; }
template <typename T1, typename T2> auto operator &(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value & b.value), 1> { return a.value & b.value; } template <typename T1, typename T2> auto operator &(T1 a, Simd<T2, 1> b)->Simd<decltype(a & b.value), 1> { return a & b.value; } template <typename T1, typename T2> auto operator &(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value & b), 1> { return a.value & b; }
template <typename T1, typename T2> auto operator &&(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value && b.value), 1> { return a.value && b.value; } template <typename T1, typename T2> auto operator &&(T1 a, Simd<T2, 1> b)->Simd<decltype(a && b.value), 1> { return a && b.value; } template <typename T1, typename T2> auto operator &&(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value && b), 1> { return a.value && b; }
template <typename T1, typename T2> auto operator ||(Simd<T1, 1> a, Simd<T2, 1> b) ->Simd<decltype(a.value || b.value), 1> { return a.value || b.value; } template <typename T1, typename T2> auto operator ||(T1 a, Simd<T2, 1> b)->Simd<decltype(a || b.value), 1> { return a || b.value; } template <typename T1, typename T2> auto operator ||(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value || b), 1> { return a.value || b; }
template <typename T>
Simd<T, 1> clz(Simd<T, 1> x_) {
  return __builtin_clz(x_.value);
}
template <typename T>
Simd<T, 1> remainder(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  T r;
  if constexpr (std::is_integral_v<T>) {
    r = a % b;
  } else {
    r = std::remainder(a, b);
  }
  if constexpr (std::is_signed_v<T>) {
    if (r != 0 && (r < 0 != b < 0)) {
      r += b;
    }
  }
  return r;
}
template <typename T>
Simd<T, 1> maximum(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  if constexpr (!std::is_integral_v<T>) {
    if (std::isnan(a)) {
      return a;
    }
  }
  return (a > b) ? a : b;
}
template <typename T>
Simd<T, 1> minimum(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  if constexpr (!std::is_integral_v<T>) {
    if (std::isnan(a)) {
      return a;
    }
  }
  return (a < b) ? a : b;
}
template <typename T>
Simd<T, 1> pow(Simd<T, 1> a, Simd<T, 1> b) {
  T base = a.value;
  T exp = b.value;
  if constexpr (!std::is_integral_v<T>) {
    return std::pow(base, exp);
  } else {
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
}
template <typename T>
Simd<T, 1> atan2(Simd<T, 1> a, Simd<T, 1> b) {
  return std::atan2(a.value, b.value);
}
template <typename T1, typename T2> Simd<bool, 1> operator >(Simd<T1, 1> a, Simd<T2, 1> b) { return a.value > b.value; } template <typename T1, typename T2> Simd<bool, 1> operator >(T1 a, Simd<T2, 1> b) { return a > b.value; } template <typename T1, typename T2> Simd<bool, 1> operator >(Simd<T1, 1> a, T2 b) { return a.value > b; }
template <typename T1, typename T2> Simd<bool, 1> operator <(Simd<T1, 1> a, Simd<T2, 1> b) { return a.value < b.value; } template <typename T1, typename T2> Simd<bool, 1> operator <(T1 a, Simd<T2, 1> b) { return a < b.value; } template <typename T1, typename T2> Simd<bool, 1> operator <(Simd<T1, 1> a, T2 b) { return a.value < b; }
template <typename T1, typename T2> Simd<bool, 1> operator >=(Simd<T1, 1> a, Simd<T2, 1> b) { return a.value >= b.value; } template <typename T1, typename T2> Simd<bool, 1> operator >=(T1 a, Simd<T2, 1> b) { return a >= b.value; } template <typename T1, typename T2> Simd<bool, 1> operator >=(Simd<T1, 1> a, T2 b) { return a.value >= b; }
template <typename T1, typename T2> Simd<bool, 1> operator <=(Simd<T1, 1> a, Simd<T2, 1> b) { return a.value <= b.value; } template <typename T1, typename T2> Simd<bool, 1> operator <=(T1 a, Simd<T2, 1> b) { return a <= b.value; } template <typename T1, typename T2> Simd<bool, 1> operator <=(Simd<T1, 1> a, T2 b) { return a.value <= b; }
template <typename T1, typename T2> Simd<bool, 1> operator ==(Simd<T1, 1> a, Simd<T2, 1> b) { return a.value == b.value; } template <typename T1, typename T2> Simd<bool, 1> operator ==(T1 a, Simd<T2, 1> b) { return a == b.value; } template <typename T1, typename T2> Simd<bool, 1> operator ==(Simd<T1, 1> a, T2 b) { return a.value == b; }
template <typename T1, typename T2> Simd<bool, 1> operator !=(Simd<T1, 1> a, Simd<T2, 1> b) { return a.value != b.value; } template <typename T1, typename T2> Simd<bool, 1> operator !=(T1 a, Simd<T2, 1> b) { return a != b.value; } template <typename T1, typename T2> Simd<bool, 1> operator !=(Simd<T1, 1> a, T2 b) { return a.value != b; }
template <typename MaskT, typename T>
Simd<T, 1> select(Simd<MaskT, 1> mask, Simd<T, 1> x, Simd<T, 1> y) {
  return mask.value ? x.value : y.value;
}
template <typename T>
Simd<T, 1> clamp(Simd<T, 1> v, Simd<T, 1> min, Simd<T, 1> max) {
  return std::clamp(v.value, min.value, max.value);
}
template <typename T, typename U>
Simd<T, 1> fma(Simd<T, 1> x, Simd<T, 1> y, U z) {
  return std::fma(x.value, y.value, Simd<T, 1>(z).value);
}
template <typename T> T max(Simd<T, 1> x) { return x.value; }
template <typename T> T min(Simd<T, 1> x) { return x.value; }
template <typename T> T sum(Simd<T, 1> x) { return x.value; }
template <typename T> T prod(Simd<T, 1> x) { return x.value; }
template <typename T> bool any(Simd<T, 1> x) { return x.value; }
template <typename T> bool all(Simd<T, 1> x) { return x.value; }
}

namespace mlx::core::simd {
constexpr float inf = std::numeric_limits<float>::infinity();
template <typename T, int N>
Simd<T, N> exp(Simd<T, N> in) {
  if constexpr (is_complex<T>) {
    return Simd<T, 1>{std::exp(in.value)};
  } else {
    Simd<float, N> x_init = in;
    auto x = x_init * 1.442695f;
    Simd<float, N> ipart, fpart;
    ipart = floor(x + 0.5);
    fpart = x - ipart;
    x = 1.535336188319500e-4f;
    x = fma(x, fpart, 1.339887440266574e-3f);
    x = fma(x, fpart, 9.618437357674640e-3f);
    x = fma(x, fpart, 5.550332471162809e-2f);
    x = fma(x, fpart, 2.402264791363012e-1f);
    x = fma(x, fpart, 6.931472028550421e-1f);
    x = fma(x, fpart, 1.000000000000000f);
    Simd<int, N> epart = (Simd<int, N>(ipart) + 127) << 23;
    auto result = select(isnan(x_init), x_init, (*(Simd<float, N>*)&epart) * x);
    result = select(x_init > 88.0f, Simd<float, N>(inf), result);
    result = select(x_init < -88.0f, Simd<float, N>(0), result);
    return Simd<T, N>(result);
  }
}
template <bool Sine, typename T, int N>
Simd<T, N> sincos(Simd<T, N> in) {
  auto sign_mask_sin = in < 0;
  in = abs(in);
  Simd<float, N> x = in;
  auto y = x * 1.27323954473516f;
  Simd<uint32_t, N> emm2 = y;
  emm2 = emm2 + 1;
  emm2 = emm2 & ~1;
  y = emm2;
  auto poly_mask = (emm2 & 2) != 0;
  x = fma(y, Simd<float, N>(-0.78515625f), x);
  x = fma(y, Simd<float, N>(-2.4187564849853515625e-4f), x);
  x = fma(y, Simd<float, N>(-3.77489497744594108e-8f), x);
  sign_mask_sin = sign_mask_sin ^ ((emm2 & 4) != 0);
  auto sign_mask_cos = ((emm2 - 2) & 4) != 0;
  auto z = x * x;
  auto y1 =
      fma(z, Simd<float, N>(2.443315711809948e-5f), -1.388731625493765e-3f);
  auto y2 = fma(z, Simd<float, N>(-1.9515295891e-4f), 8.3321608736e-3f);
  y1 = fma(y1, z, 4.166664568298827e-2f);
  y2 = fma(y2, z, -1.6666654611e-1f);
  y1 = y1 * z;
  y2 = y2 * z;
  y1 = y1 * z;
  y2 = fma(x, y2, x);
  y1 = fma(z, Simd<float, N>(-0.5f), y1);
  y1 = y1 + 1.0f;
  if constexpr (Sine) {
    auto ys = select(poly_mask, y1, y2);
    return select(sign_mask_sin, -ys, ys);
  } else {
    auto yc = select(poly_mask, y2, y1);
    return select(sign_mask_cos, yc, -yc);
  }
}
template <typename T, int N>
Simd<T, N> sin(Simd<T, N> x) {
  if constexpr (is_complex<T>) {
    return std::sin(x.value);
  } else {
    return sincos<true>(x);
  }
}
template <typename T, int N>
Simd<T, N> cos(Simd<T, N> x) {
  if constexpr (is_complex<T>) {
    return std::cos(x.value);
  } else {
    return sincos<false>(x);
  }
}
template <typename T, int N>
Simd<T, N> erf(Simd<T, N> x) {
  Simd<float, N> v = x;
  auto t = recip(fma(Simd<float, N>(0.3275911f), abs(v), 1.0f));
  auto r = fma(Simd<float, N>(1.061405429f), t, -1.453152027f);
  r = fma(r, t, 1.421413741f);
  r = fma(r, t, -0.284496736f);
  r = fma(r, t, 0.254829592f);
  auto e = -exp(-v * v);
  auto result = Simd<T, N>(fma(e * t, r, 1.0f));
  return select(x > 0, result, -result);
}
template <typename T, int N>
Simd<T, N> erfinv(Simd<T, N> a_) {
  Simd<float, N> a = a_;
  auto t = fma(a, 0.0f - a, 1.0f);
  t = log(t);
  auto lhs = [](auto t) {
    Simd<float, N> p;
    p = 3.03697567e-10f;
    p = fma(p, t, 2.93243101e-8f);
    p = fma(p, t, 1.22150334e-6f);
    p = fma(p, t, 2.84108955e-5f);
    p = fma(p, t, 3.93552968e-4f);
    p = fma(p, t, 3.02698812e-3f);
    p = fma(p, t, 4.83185798e-3f);
    p = fma(p, t, -2.64646143e-1f);
    return fma(p, t, 8.40016484e-1f);
  };
  auto rhs = [](auto t) {
    Simd<float, N> p;
    p = 5.43877832e-9f;
    p = fma(p, t, 1.43285448e-7f);
    p = fma(p, t, 1.22774793e-6f);
    p = fma(p, t, 1.12963626e-7f);
    p = fma(p, t, -5.61530760e-5f);
    p = fma(p, t, -1.47697632e-4f);
    p = fma(p, t, 2.31468678e-3f);
    p = fma(p, t, 1.15392581e-2f);
    p = fma(p, t, -2.32015476e-1f);
    return fma(p, t, 8.86226892e-1f);
  };
  auto thresh = 6.125f;
  if constexpr (N == 1) {
    if ((abs(t) > thresh).value) {
      return a * lhs(t);
    } else {
      return a * rhs(t);
    }
  } else {
    return a * select(abs(t) > thresh, lhs(t), rhs(t));
  }
}
}

namespace mlx::core::detail {
using namespace mlx::core::simd;
struct Abs { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::abs(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct ArcCos { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::acos(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct ArcCosh { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::acosh(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct ArcSin { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::asin(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct ArcSinh { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::asinh(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct ArcTan { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::atan(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct ArcTanh { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::atanh(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct BitwiseInvert { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::operator~(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Ceil { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::ceil(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Conjugate { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::conj(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Cos { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::cos(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Cosh { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::cosh(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Erf { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::erf(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct ErfInv { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::erfinv(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Exp { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::exp(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Expm1 { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::expm1(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Floor { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::floor(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };;
struct Log { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::log(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };;
struct Log2 { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::log2(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };;
struct Log10 { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::log10(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };;
struct Log1p { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::log1p(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };;
struct LogicalNot { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::operator!(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Negative { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::operator-(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Round { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::rint(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };;
struct Sin { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::sin(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Sinh { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::sinh(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Sqrt { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::sqrt(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Rsqrt { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::rsqrt(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Tan { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::tan(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Tanh { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x) { return simd::tanh(x); } template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; } };
struct Imag {
  template <int N>
  Simd<float, N> operator()(Simd<complex64_t, N> x) {
    return simd::imag(x);
  }
  template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; }
};
struct Real {
  template <int N>
  Simd<float, N> operator()(Simd<complex64_t, N> x) {
    return simd::real(x);
  }
  template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; }
};
struct Sigmoid {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x) {
    auto y = 1.0f / (1.0f + simd::exp(simd::abs(x)));
    return simd::select(x < Simd<T, N>{0}, y, Simd<T, N>{1} - y);
  }
  template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; }
};
struct Sign {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x) {
    auto z = Simd<T, N>{0};
    auto o = Simd<T, N>{1};
    auto m = Simd<T, N>{-1};
    if constexpr (std::is_unsigned_v<T>) {
      return simd::select(x == z, z, o);
    } else if constexpr (std::is_same_v<T, complex64_t>) {
      return simd::select(x == z, x, Simd<T, N>(x / simd::abs(x)));
    } else {
      return simd::select(x < z, m, simd::select(x > z, o, z));
    }
  }
  template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; }
};
struct Square {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x) {
    return x * x;
  }
  template <typename T> T operator()(T x) { return (*this)(Simd<T, 1>(x)).value; }
};
template <int N>
Simd<float, N> fp32_from_bits(Simd<uint32_t, N> x) {
  return *(Simd<float, N>*)(&x);
}
template <int N>
Simd<uint32_t, N> fp32_to_bits(Simd<float, N> x) {
  return *(Simd<uint32_t, N>*)(&x);
}
struct ToFP8 {
  template <typename T, int N>
  Simd<uint8_t, N> operator()(Simd<T, N> f) {
    uint32_t fp8_max = 543 << 21;
    auto denorm_mask = Simd<uint32_t, N>(141 << 23);
    Simd<uint32_t, N> f_bits;
    Simd<float, N> f32 = f;
    f_bits = fp32_to_bits(f32);
    Simd<uint8_t, N> result = 0u;
    auto sign = f_bits & 0x80000000;
    f_bits = f_bits ^ sign;
    auto f_bits_low =
        fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    auto result_low = Simd<uint8_t, N>(f_bits_low - denorm_mask);
    auto mant_odd = Simd<uint8_t, N>((f_bits >> 20) & 1);
    auto f_bits_high = f_bits + (((uint32_t)(7 - 127) << 23) + 0x7FFFF);
    f_bits_high = f_bits_high + Simd<uint32_t, N>(mant_odd);
    auto result_high = Simd<uint8_t, N>(f_bits_high >> 20);
    result = select(f_bits < (121 << 23), result_low, result_high);
    auto result_sat = Simd<uint8_t, N>(0x7E);
    result = select(f_bits >= fp8_max, result_sat, result);
    return result | Simd<uint8_t, N>(sign >> 24);
  }
  template <typename T>
  uint8_t operator()(T x) {
    return (*this)(Simd<T, 1>(x)).value;
  }
};
struct FromFP8 {
  template <int N>
  Simd<float, N> operator()(Simd<uint8_t, N> x) {
    auto v = Simd<uint16_t, N>(x & 127) << 7;
    Simd<float, N> out;
    if constexpr (simd::max_size<float16_t> >= N) {
      auto converted = *(Simd<float16_t, N>*)(&v);
      out = converted * 256.0;
    } else {
      for (int i = 0; i < N; ++i) {
        auto converted = *(float16_t*)(&v[i]);
        out[i] = converted * 256.0;
      }
    }
    auto sign = Simd<bool, N>(x & 128);
    return select(sign, -out, out);
  }
  float operator()(uint8_t x) {
    return (*this)(Simd<uint8_t, 1>(x)).value;
  }
};
}
namespace mlx::core::detail {
using namespace mlx::core::simd;
struct Add { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator+(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct ArcTan2 { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return atan2(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Divide { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator/(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Multiply { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator*(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Subtract { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator-(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct LogicalAnd { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator&&(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct LogicalOr { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator||(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct BitwiseAnd { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator&(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct BitwiseOr { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator|(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct BitwiseXor { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator^(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct LeftShift { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator<<(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct RightShift { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator>>(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Remainder { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return remainder(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Maximum { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return maximum(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Minimum { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return minimum(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Power { template <int N, typename T> Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { return pow(x, y); } template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Equal { template <int N, typename T> Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator==(x, y); } template <typename T> bool operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Greater { template <int N, typename T> Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator>(x, y); } template <typename T> bool operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct GreaterEqual { template <int N, typename T> Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator>=(x, y); } template <typename T> bool operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct Less { template <int N, typename T> Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator<(x, y); } template <typename T> bool operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct LessEqual { template <int N, typename T> Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator<=(x, y); } template <typename T> bool operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct NotEqual { template <int N, typename T> Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) { return operator!=(x, y); } template <typename T> bool operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; } };
struct NaNEqual {
  template <int N, typename T>
  Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) {
    return x == y || (isnan(x) && isnan(y));
  }
  template <typename T>
  bool operator()(T x, T y) {
    return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value;
  }
};
struct LogAddExp {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) {
    auto maxval = maximum(x, y);
    auto minval = minimum(x, y);
    auto mask = minval == -inf || maxval == inf;
    auto out = maxval + log1p(exp(minval - maxval));
    return select(mask, Simd<T, N>(maxval), Simd<T, N>(out));
  }
  template <typename T> T operator()(T x, T y) { return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; }
};
struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return (*this)(Simd<bool, 1>(condition), Simd<T, 1>(x), Simd<T, 1>(y))
        .value;
  }
  template <int N, typename T>
  Simd<T, N> operator()(Simd<bool, N> condition, Simd<T, N> x, Simd<T, N> y) {
    return select(condition, x, y);
  }
};
}
const char* get_kernel_preamble();
using namespace mlx::core;
using namespace mlx::core::detail;
)preamble";
}
