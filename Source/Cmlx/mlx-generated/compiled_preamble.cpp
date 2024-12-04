const char* get_kernel_preamble() {
return R"preamble(
#include <cmath>
  #include <complex>
  #include <cstdint>
  #include <vector>
# 1 "Source/Cmlx/mlx/mlx/backend/common/compiled_preamble.h"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 424 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "Source/Cmlx/mlx/mlx/backend/common/compiled_preamble.h" 2





# 1 "Source/Cmlx/mlx/mlx/types/half_types.h" 1





# 1 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/arm_fp16.h" 1 3
# 27 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/arm_fp16.h" 3
# 1 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/stdint.h" 1 3
# 96 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/stdint.h" 3
typedef long long int int64_t;

typedef long long unsigned int uint64_t;
# 118 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/stdint.h" 3
typedef int64_t int_least64_t;
typedef uint64_t uint_least64_t;
typedef int64_t int_fast64_t;
typedef uint64_t uint_fast64_t;
# 193 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/stdint.h" 3
typedef int int32_t;




typedef unsigned int uint32_t;
# 216 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/stdint.h" 3
typedef int32_t int_least32_t;
typedef uint32_t uint_least32_t;
typedef int32_t int_fast32_t;
typedef uint32_t uint_fast32_t;
# 241 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/stdint.h" 3
typedef short int16_t;

typedef unsigned short uint16_t;
# 255 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/stdint.h" 3
typedef int16_t int_least16_t;
typedef uint16_t uint_least16_t;
typedef int16_t int_fast16_t;
typedef uint16_t uint_fast16_t;





typedef signed char int8_t;

typedef unsigned char uint8_t;







typedef int8_t int_least8_t;
typedef uint8_t uint_least8_t;
typedef int8_t int_fast8_t;
typedef uint8_t uint_fast8_t;
# 291 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/stdint.h" 3
typedef long int intptr_t;






typedef long unsigned int uintptr_t;





typedef long int intmax_t;
typedef long unsigned int uintmax_t;
# 28 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/16/include/arm_fp16.h" 2 3

typedef __fp16 float16_t;
# 7 "Source/Cmlx/mlx/mlx/types/half_types.h" 2
namespace mlx::core {
typedef __fp16 float16_t;
}
# 30 "Source/Cmlx/mlx/mlx/types/half_types.h"
# 1 "Source/Cmlx/mlx/mlx/types/bf16.h" 1
# 12 "Source/Cmlx/mlx/mlx/types/bf16.h"
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
# 96 "Source/Cmlx/mlx/mlx/types/bf16.h"
inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline float operator+(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline float operator+(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline double operator+(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) + static_cast<double>(rhs); } inline double operator+(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) + static_cast<double>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, bool rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(bool lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }; inline _MLX_BFloat16 operator+(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline _MLX_BFloat16 operator+(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); };;
inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline float operator-(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline float operator-(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline double operator-(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) - static_cast<double>(rhs); } inline double operator-(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) - static_cast<double>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, bool rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(bool lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }; inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline _MLX_BFloat16 operator-(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); };;
inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline float operator*(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline float operator*(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline double operator*(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) * static_cast<double>(rhs); } inline double operator*(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) * static_cast<double>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, bool rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(bool lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }; inline _MLX_BFloat16 operator*(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline _MLX_BFloat16 operator*(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); };;
inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline float operator/(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline float operator/(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline double operator/(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) / static_cast<double>(rhs); } inline double operator/(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) / static_cast<double>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, bool rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(bool lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }; inline _MLX_BFloat16 operator/(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline _MLX_BFloat16 operator/(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); };;
# 114 "Source/Cmlx/mlx/mlx/types/bf16.h"
inline bool operator>(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) > static_cast<double>(rhs); } inline bool operator>(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) > static_cast<double>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); }; inline bool operator>(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); } inline bool operator>(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) > static_cast<float>(rhs); };;
inline bool operator<(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) < static_cast<double>(rhs); } inline bool operator<(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) < static_cast<double>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); }; inline bool operator<(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); } inline bool operator<(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) < static_cast<float>(rhs); };;
inline bool operator>=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) >= static_cast<double>(rhs); } inline bool operator>=(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) >= static_cast<double>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); }; inline bool operator>=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); } inline bool operator>=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) >= static_cast<float>(rhs); };;
inline bool operator<=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) <= static_cast<double>(rhs); } inline bool operator<=(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) <= static_cast<double>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); }; inline bool operator<=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); } inline bool operator<=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) <= static_cast<float>(rhs); };;
inline bool operator==(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) == static_cast<double>(rhs); } inline bool operator==(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) == static_cast<double>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); }; inline bool operator==(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); } inline bool operator==(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) == static_cast<float>(rhs); };;
inline bool operator!=(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, float rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(float lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, double rhs) { return static_cast<double>(lhs) != static_cast<double>(rhs); } inline bool operator!=(double lhs, _MLX_BFloat16 rhs) { return static_cast<double>(lhs) != static_cast<double>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, int32_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(int32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, uint32_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(uint32_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, int64_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(int64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); }; inline bool operator!=(_MLX_BFloat16 lhs, uint64_t rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); } inline bool operator!=(uint64_t lhs, _MLX_BFloat16 rhs) { return static_cast<float>(lhs) != static_cast<float>(rhs); };;




inline _MLX_BFloat16 operator-(_MLX_BFloat16 lhs) {
  return -static_cast<float>(lhs);
}
# 139 "Source/Cmlx/mlx/mlx/types/bf16.h"
inline _MLX_BFloat16& operator+=(_MLX_BFloat16& lhs, const float& rhs) { lhs = lhs + rhs; return lhs; } inline float& operator+=(float& lhs, _MLX_BFloat16 rhs) { lhs = lhs + rhs; return lhs; };
inline _MLX_BFloat16& operator-=(_MLX_BFloat16& lhs, const float& rhs) { lhs = lhs - rhs; return lhs; } inline float& operator-=(float& lhs, _MLX_BFloat16 rhs) { lhs = lhs - rhs; return lhs; };
inline _MLX_BFloat16& operator*=(_MLX_BFloat16& lhs, const float& rhs) { lhs = lhs * rhs; return lhs; } inline float& operator*=(float& lhs, _MLX_BFloat16 rhs) { lhs = lhs * rhs; return lhs; };
inline _MLX_BFloat16& operator/=(_MLX_BFloat16& lhs, const float& rhs) { lhs = lhs / rhs; return lhs; } inline float& operator/=(float& lhs, _MLX_BFloat16 rhs) { lhs = lhs / rhs; return lhs; };
# 165 "Source/Cmlx/mlx/mlx/types/bf16.h"
inline _MLX_BFloat16 operator|(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ | rhs.bits_; return out; } inline _MLX_BFloat16 operator|(_MLX_BFloat16 lhs, uint16_t rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ | rhs; return out; } inline _MLX_BFloat16 operator|(uint16_t lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs | rhs.bits_; return out; };
inline _MLX_BFloat16 operator&(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ & rhs.bits_; return out; } inline _MLX_BFloat16 operator&(_MLX_BFloat16 lhs, uint16_t rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ & rhs; return out; } inline _MLX_BFloat16 operator&(uint16_t lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs & rhs.bits_; return out; };
inline _MLX_BFloat16 operator^(_MLX_BFloat16 lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ ^ rhs.bits_; return out; } inline _MLX_BFloat16 operator^(_MLX_BFloat16 lhs, uint16_t rhs) { _MLX_BFloat16 out; out.bits_ = lhs.bits_ ^ rhs; return out; } inline _MLX_BFloat16 operator^(uint16_t lhs, _MLX_BFloat16 rhs) { _MLX_BFloat16 out; out.bits_ = lhs ^ rhs.bits_; return out; };
# 181 "Source/Cmlx/mlx/mlx/types/bf16.h"
inline _MLX_BFloat16& operator|=(_MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs.bits_ = lhs.bits_ | rhs.bits_; return lhs; } inline _MLX_BFloat16& operator|=(_MLX_BFloat16& lhs, uint16_t rhs) { lhs.bits_ = lhs.bits_ | rhs; return lhs; };
inline _MLX_BFloat16& operator&=(_MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs.bits_ = lhs.bits_ & rhs.bits_; return lhs; } inline _MLX_BFloat16& operator&=(_MLX_BFloat16& lhs, uint16_t rhs) { lhs.bits_ = lhs.bits_ & rhs; return lhs; };
inline _MLX_BFloat16& operator^=(_MLX_BFloat16& lhs, _MLX_BFloat16 rhs) { lhs.bits_ = lhs.bits_ ^ rhs.bits_; return lhs; } inline _MLX_BFloat16& operator^=(_MLX_BFloat16& lhs, uint16_t rhs) { lhs.bits_ = lhs.bits_ ^ rhs; return lhs; };



}
# 31 "Source/Cmlx/mlx/mlx/types/half_types.h" 2
namespace mlx::core {
typedef struct _MLX_BFloat16 bfloat16_t;
}




namespace mlx::core {
# 49 "Source/Cmlx/mlx/mlx/types/half_types.h"
inline float operator+(float16_t lhs, bfloat16_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); } inline float operator+(bfloat16_t lhs, float16_t rhs) { return static_cast<float>(lhs) + static_cast<float>(rhs); }
inline float operator-(float16_t lhs, bfloat16_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); } inline float operator-(bfloat16_t lhs, float16_t rhs) { return static_cast<float>(lhs) - static_cast<float>(rhs); }
inline float operator*(float16_t lhs, bfloat16_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); } inline float operator*(bfloat16_t lhs, float16_t rhs) { return static_cast<float>(lhs) * static_cast<float>(rhs); }
inline float operator/(float16_t lhs, bfloat16_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); } inline float operator/(bfloat16_t lhs, float16_t rhs) { return static_cast<float>(lhs) / static_cast<float>(rhs); }


}
# 7 "Source/Cmlx/mlx/mlx/backend/common/compiled_preamble.h" 2
# 1 "Source/Cmlx/mlx/mlx/types/complex.h" 1






namespace mlx::core {

struct complex64_t;
struct complex128_t;

template <typename T>
inline constexpr bool can_convert_to_complex128 =
    !std::is_same_v<T, complex128_t> && std::is_convertible_v<T, double>;

struct complex128_t : public std::complex<double> {
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
# 109 "Source/Cmlx/mlx/mlx/types/complex.h"
inline complex64_t operator+(const std::complex<float>& x, const complex64_t& y) { return x + static_cast<std::complex<float>>(y); } inline complex64_t operator+(const complex64_t& x, const std::complex<float>& y) { return static_cast<std::complex<float>>(x) + y; } inline complex64_t operator+(const complex64_t& x, const complex64_t& y) { return static_cast<std::complex<float>>(x) + static_cast<std::complex<float>>(y); } inline complex64_t operator+(bool x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, bool y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(uint32_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, uint32_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(uint64_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, uint64_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(int32_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, int32_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(int64_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, int64_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(float16_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, float16_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(bfloat16_t x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, bfloat16_t y) { return x + static_cast<complex64_t>(y); } inline complex64_t operator+(float x, const complex64_t& y) { return static_cast<complex64_t>(x) + y; } inline complex64_t operator+(const complex64_t& x, float y) { return x + static_cast<complex64_t>(y); }

}
# 8 "Source/Cmlx/mlx/mlx/backend/common/compiled_preamble.h" 2
# 1 "Source/Cmlx/mlx/mlx/backend/common/ops.h" 1







namespace mlx::core::detail {

namespace {
constexpr float inf = std::numeric_limits<float>::infinity();
}

typedef union {
  int i;
  float f;
} IntOrFloat;

inline float fast_exp(float x) {
  if (x == -std::numeric_limits<float>::infinity()) {
    return 0.0f;
  } else if (x == std::numeric_limits<float>::infinity() || std::isnan(x)) {
    return x;
  }
  x *= 1.442695;
  float ipart, fpart;
  IntOrFloat epart;
  x = std::max(-80.f, std::min(x, 80.f));
  ipart = std::floor(x + 0.5);
  fpart = x - ipart;

  x = 1.535336188319500e-4f;
  x = x * fpart + 1.339887440266574e-3f;
  x = x * fpart + 9.618437357674640e-3f;
  x = x * fpart + 5.550332471162809e-2f;
  x = x * fpart + 2.402264791363012e-1f;
  x = x * fpart + 6.931472028550421e-1f;
  x = x * fpart + 1.000000000000000f;



  epart.i = (int(ipart) + 127) << 23;

  return epart.f * x;
}

inline float fast_erf(float a) {
  float r, s, t, u;
  t = std::abs(a);
  s = a * a;
  if (t > 0.927734375f) {

    r = std::fma(
        -1.72853470e-5f, t, 3.83197126e-4f);
    u = std::fma(
        -3.88396438e-3f, t, 2.42546219e-2f);
    r = std::fma(r, s, u);
    r = std::fma(r, t, -1.06777877e-1f);
    r = std::fma(r, t, -6.34846687e-1f);
    r = std::fma(r, t, -1.28717512e-1f);
    r = std::fma(r, t, -t);

    r = 1.0f - std::exp(r);
    r = std::copysign(r, a);
  } else {

    r = -5.96761703e-4f;
    r = std::fma(r, s, 4.99119423e-3f);
    r = std::fma(r, s, -2.67681349e-2f);
    r = std::fma(r, s, 1.12819925e-1f);
    r = std::fma(r, s, -3.76125336e-1f);
    r = std::fma(r, s, 1.28379166e-1f);
    r = std::fma(r, a, a);
  }
  return r;
}

inline float fast_erfinv(float a) {
  auto t = std::fma(a, 0.0f - a, 1.0f);
  t = std::log(t);
  float p;
  if (std::abs(t) > 6.125f) {
    p = 3.03697567e-10f;
    p = std::fma(p, t, 2.93243101e-8f);
    p = std::fma(p, t, 1.22150334e-6f);
    p = std::fma(p, t, 2.84108955e-5f);
    p = std::fma(p, t, 3.93552968e-4f);
    p = std::fma(p, t, 3.02698812e-3f);
    p = std::fma(p, t, 4.83185798e-3f);
    p = std::fma(p, t, -2.64646143e-1f);
    p = std::fma(p, t, 8.40016484e-1f);
  } else {
    p = 5.43877832e-9f;
    p = std::fma(p, t, 1.43285448e-7f);
    p = std::fma(p, t, 1.22774793e-6f);
    p = std::fma(p, t, 1.12963626e-7f);
    p = std::fma(p, t, -5.61530760e-5f);
    p = std::fma(p, t, -1.47697632e-4f);
    p = std::fma(p, t, 2.31468678e-3f);
    p = std::fma(p, t, 1.15392581e-2f);
    p = std::fma(p, t, -2.32015476e-1f);
    p = std::fma(p, t, 8.86226892e-1f);
  }
  return a * p;
}

struct Abs {
  template <typename T>
  T operator()(T x) {
    return std::abs(x);
  }
  uint8_t operator()(uint8_t x) {
    return x;
  }
  uint16_t operator()(uint16_t x) {
    return x;
  }
  uint32_t operator()(uint32_t x) {
    return x;
  }
  uint64_t operator()(uint64_t x) {
    return x;
  }
  bool operator()(bool x) {
    return x;
  }
};

struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return std::acos(x);
  }
};

struct ArcCosh {
  template <typename T>
  T operator()(T x) {
    return std::acosh(x);
  }
};

struct ArcSin {
  template <typename T>
  T operator()(T x) {
    return std::asin(x);
  }
};

struct ArcSinh {
  template <typename T>
  T operator()(T x) {
    return std::asinh(x);
  }
};

struct ArcTan {
  template <typename T>
  T operator()(T x) {
    return std::atan(x);
  }
};

struct ArcTan2 {
  template <typename T>
  T operator()(T y, T x) {
    return std::atan2(y, x);
  }
};

struct ArcTanh {
  template <typename T>
  T operator()(T x) {
    return std::atanh(x);
  }
};

struct Ceil {
  template <typename T>
  T operator()(T x) {
    return std::ceil(x);
  }
  int8_t operator()(int8_t x) {
    return x;
  }
  int16_t operator()(int16_t x) {
    return x;
  }
  int32_t operator()(int32_t x) {
    return x;
  }
  int64_t operator()(int64_t x) {
    return x;
  }
  uint8_t operator()(uint8_t x) {
    return x;
  }
  uint16_t operator()(uint16_t x) {
    return x;
  }
  uint32_t operator()(uint32_t x) {
    return x;
  }
  uint64_t operator()(uint64_t x) {
    return x;
  }
  bool operator()(bool x) {
    return x;
  }
};

struct Conjugate {
  complex64_t operator()(complex64_t x) {
    return std::conj(x);
  }
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return std::cos(x);
  }
};

struct Cosh {
  template <typename T>
  T operator()(T x) {
    return std::cosh(x);
  }
};

struct Erf {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(fast_erf(static_cast<float>(x)));
  }
};

struct ErfInv {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(fast_erfinv(static_cast<float>(x)));
  }
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    return fast_exp(x);
  }

  complex64_t operator()(complex64_t x) {
    return std::exp(x);
  }
};

struct Expm1 {
  template <typename T>
  T operator()(T x) {
    return expm1(x);
  }
};

struct Floor {
  template <typename T>
  T operator()(T x) {
    return std::floor(x);
  }
  int8_t operator()(int8_t x) {
    return x;
  }
  int16_t operator()(int16_t x) {
    return x;
  }
  int32_t operator()(int32_t x) {
    return x;
  }
  int64_t operator()(int64_t x) {
    return x;
  }
  uint8_t operator()(uint8_t x) {
    return x;
  }
  uint16_t operator()(uint16_t x) {
    return x;
  }
  uint32_t operator()(uint32_t x) {
    return x;
  }
  uint64_t operator()(uint64_t x) {
    return x;
  }
  bool operator()(bool x) {
    return x;
  }
};

struct Imag {
  template <typename T>
  T operator()(T x) {
    return std::imag(x);
  }
};

struct Log {
  template <typename T>
  T operator()(T x) {
    return std::log(x);
  }
};

struct Log2 {
  template <typename T>
  T operator()(T x) {
    return std::log2(x);
  }
};

struct Log10 {
  template <typename T>
  T operator()(T x) {
    return std::log10(x);
  }
};

struct Log1p {
  template <typename T>
  T operator()(T x) {
    return log1p(x);
  }
};

struct LogicalNot {
  template <typename T>
  T operator()(T x) {
    return !x;
  }
};

struct Negative {
  template <typename T>
  T operator()(T x) {
    return -x;
  }
};

struct Real {
  template <typename T>
  T operator()(T x) {
    return std::real(x);
  }
};

struct Round {
  template <typename T>
  T operator()(T x) {
    return std::rint(x);
  }

  complex64_t operator()(complex64_t x) {
    return {std::rint(x.real()), std::rint(x.imag())};
  }
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto one = static_cast<decltype(x)>(1.0);
    return one / (one + fast_exp(-x));
  }
};

struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  }
  uint8_t operator()(uint8_t x) {
    return x != 0;
  }
  uint16_t operator()(uint16_t x) {
    return x != 0;
  }
  uint32_t operator()(uint32_t x) {
    return x != 0;
  }
  uint64_t operator()(uint64_t x) {
    return x != 0;
  }

  complex64_t operator()(complex64_t x) {
    return x == complex64_t(0) ? x : x / std::abs(x);
  }
};

struct Sin {
  template <typename T>
  T operator()(T x) {
    return std::sin(x);
  }
};

struct Sinh {
  template <typename T>
  T operator()(T x) {
    return std::sinh(x);
  }
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  }
};

struct Sqrt {
  template <typename T>
  T operator()(T x) {
    return std::sqrt(x);
  }
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return static_cast<decltype(x)>(1.0) / std::sqrt(x);
  }
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return std::tan(x);
  }
};

struct Tanh {
  template <typename T>
  T operator()(T x) {
    return std::tanh(x);
  }
};

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
  std::enable_if_t<std::is_integral_v<T> & !std::is_signed_v<T>, T> operator()(
      T numerator,
      T denominator) {
    return numerator % denominator;
  }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T> & std::is_signed_v<T>, T> operator()(
      T numerator,
      T denominator) {
    auto r = numerator % denominator;
    if (r != 0 && (r < 0 != denominator < 0))
      r += denominator;
    return r;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(
      T numerator,
      T denominator) {
    auto r = std::fmod(numerator, denominator);
    if (r != 0 && (r < 0 != denominator < 0)) {
      r += denominator;
    }
    return r;
  }

  complex64_t operator()(complex64_t numerator, complex64_t denominator) {
    return numerator % denominator;
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
    return x == y || (std::isnan(x) && std::isnan(y));
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

struct Maximum {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T x, T y) {
    return (x > y) ? x : y;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T x, T y) {
    if (std::isnan(x)) {
      return x;
    }
    return (x > y) ? x : y;
  }
};

struct Minimum {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T x, T y) {
    return x < y ? x : y;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T x, T y) {
    if (std::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }
};

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto maxval = Maximum()(x, y);
    auto minval = Minimum()(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : static_cast<decltype(x)>(
              maxval + std::log1p(fast_exp(minval - maxval)));
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
};

struct Power {
  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T base, T exp) {
    return std::pow(base, exp);
  }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T base, T exp) {
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
  }
};

struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  }
};

struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};

struct BitwiseAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x & y;
  }
};

struct BitwiseOr {
  template <typename T>
  T operator()(T x, T y) {
    return x | y;
  }
};

struct BitwiseXor {
  template <typename T>
  T operator()(T x, T y) {
    return x ^ y;
  }
};

struct LeftShift {
  template <typename T>
  T operator()(T x, T y) {
    return x << y;
  }
};

struct RightShift {
  template <typename T>
  T operator()(T x, T y) {
    return x >> y;
  }
};

}
# 9 "Source/Cmlx/mlx/mlx/backend/common/compiled_preamble.h" 2


const char* get_kernel_preamble();
using namespace mlx::core;
using namespace mlx::core::detail;
)preamble";
}
