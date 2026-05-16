namespace mlx::core::metal {

const char* embedded_default_library() {
  return R"MLXEMB(

// ---- embedded from Source/Cmlx/mlx-generated/metal/arg_reduce.metal ----
// Copyright © 2023 Apple Inc.

#include <metal_simdgroup>


// ---- embedded from Source/Cmlx/mlx-generated/metal/utils.h ----
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_math>


// ---- embedded from Source/Cmlx/mlx-generated/metal/bf16.h ----
// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}

// ---- embedded from Source/Cmlx/mlx-generated/metal/bf16_math.h ----
// Copyright © 2023 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// Metal math for bfloat16
///////////////////////////////////////////////////////////////////////////////

/*

Following the Metal Shading Language Specification (Metal 3.1)

"bfloat is an extended itypeing point type that only allows implicit conversion
 to a type of greater itypeing point rank. While bfloat can be implicitly
 converted to itype, it cannot be implicitly converted to half, and neither
 itype nor half can be implicitly converted to bfloat."

Further, as far as I can tell, the stdlib math/simd functions are not defined
for bfloat and calling with an argument of type bfloat will result in that
argument getting implicitly converted to itype which then returns an output
that is (likely) a itype which cannot be implicitly converted into a bfloat

This leads to situations where
bfloat a = 5.0bf;
bfloat b = metal::abs(a); // this will throw an error since abs return itype
bfloat c = static_cast<bfloat>(metal::abs(a)); // this is fine

For the moment, I will be adding overloaded instantiations of the math
functions to accordingly automatically handle the casting

*/

#define instantiate_metal_math_funcs(itype, otype, ctype, mfast)               \
                                                                               \
  METAL_FUNC otype abs(itype x) {                                              \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acos(itype x) {                                             \
    return static_cast<otype>(__metal_acos(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acosh(itype x) {                                            \
    return static_cast<otype>(__metal_acosh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype asin(itype x) {                                             \
    return static_cast<otype>(__metal_asin(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype asinh(itype x) {                                            \
    return static_cast<otype>(__metal_asinh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype atan(itype y_over_x) {                                      \
    return static_cast<otype>(                                                 \
        __metal_atan(static_cast<ctype>(y_over_x), mfast));                    \
  }                                                                            \
  METAL_FUNC otype atan2(itype y, itype x) {                                   \
    return static_cast<otype>(                                                 \
        __metal_atan2(static_cast<ctype>(y), static_cast<ctype>(x), mfast));   \
  }                                                                            \
  METAL_FUNC otype atanh(itype x) {                                            \
    return static_cast<otype>(__metal_atanh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype ceil(itype x) {                                             \
    return static_cast<otype>(__metal_ceil(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cos(itype x) {                                              \
    return static_cast<otype>(__metal_cos(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype cosh(itype x) {                                             \
    return static_cast<otype>(__metal_cosh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cospi(itype x) {                                            \
    return static_cast<otype>(__metal_cospi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype divide(itype x, itype y) {                                  \
    return static_cast<otype>(                                                 \
        __metal_divide(static_cast<ctype>(x), static_cast<ctype>(y), mfast));  \
  }                                                                            \
  METAL_FUNC otype exp(itype x) {                                              \
    return static_cast<otype>(__metal_exp(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype exp10(itype x) {                                            \
    return static_cast<otype>(__metal_exp10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype exp2(itype x) {                                             \
    return static_cast<otype>(__metal_exp2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fabs(itype x) {                                             \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fdim(itype x, itype y) {                                    \
    ctype t = static_cast<ctype>(x - y);                                       \
    return static_cast<otype>(select(t, ctype(0), t < ctype(0) || x == y));    \
  }                                                                            \
  METAL_FUNC otype floor(itype x) {                                            \
    return static_cast<otype>(__metal_floor(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype fma(itype x, itype y, itype z) {                            \
    return static_cast<otype>(__metal_fma(                                     \
        static_cast<ctype>(x), static_cast<ctype>(y), static_cast<ctype>(z))); \
  }                                                                            \
  METAL_FUNC otype fmax(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmax3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmedian3(itype x, itype y, itype z) {                       \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmin(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmin3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmod(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmod(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fract(itype x) {                                            \
    return static_cast<otype>(__metal_fract(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype frexp(itype x, thread int& exp) {                           \
    return static_cast<otype>(__metal_frexp(static_cast<ctype>(x), &exp));     \
  }                                                                            \
  METAL_FUNC otype ldexp(itype x, int k) {                                     \
    return static_cast<otype>(__metal_ldexp(static_cast<ctype>(x), k, mfast)); \
  }                                                                            \
  METAL_FUNC otype log(itype x) {                                              \
    return static_cast<otype>(__metal_log(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype log10(itype x) {                                            \
    return static_cast<otype>(__metal_log10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype log2(itype x) {                                             \
    return static_cast<otype>(__metal_log2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype max(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype max3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype median3(itype x, itype y, itype z) {                        \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype min(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype min3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype nextafter(itype x, itype y) {                               \
    return static_cast<otype>(                                                 \
        __metal_nextafter(static_cast<ctype>(x), static_cast<ctype>(y)));      \
  }                                                                            \
  METAL_FUNC otype pow(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_pow(static_cast<ctype>(x), static_cast<ctype>(y), mfast));     \
  }                                                                            \
  METAL_FUNC otype powr(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_powr(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype rint(itype x) {                                             \
    return static_cast<otype>(__metal_rint(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype round(itype x) {                                            \
    return static_cast<otype>(__metal_round(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype rsqrt(itype x) {                                            \
    return static_cast<otype>(__metal_rsqrt(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sin(itype x) {                                              \
    return static_cast<otype>(__metal_sin(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype sinh(itype x) {                                             \
    return static_cast<otype>(__metal_sinh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype sinpi(itype x) {                                            \
    return static_cast<otype>(__metal_sinpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sqrt(itype x) {                                             \
    return static_cast<otype>(__metal_sqrt(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tan(itype x) {                                              \
    return static_cast<otype>(__metal_tan(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype tanh(itype x) {                                             \
    return static_cast<otype>(__metal_tanh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tanpi(itype x) {                                            \
    return static_cast<otype>(__metal_tanpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype trunc(itype x) {                                            \
    return static_cast<otype>(__metal_trunc(static_cast<ctype>(x), mfast));    \
  }

namespace metal {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_MAYBE_FAST_MATH__);

namespace fast {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_FAST_MATH__);

} // namespace fast

namespace precise {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_PRECISE_MATH__);

} // namespace precise

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Metal simd for bfloat16
///////////////////////////////////////////////////////////////////////////////

#define instantiate_metal_simd_comm_funcs(                                   \
    itype, otype, ctype, itype_to_ctype, ctype_to_otype)                     \
                                                                             \
  METAL_FUNC otype simd_broadcast(itype data, ushort broadcast_lane_id) {    \
    return ctype_to_otype(                                                   \
        __metal_simd_broadcast(itype_to_ctype(data), broadcast_lane_id));    \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle(itype data, ushort simd_lane_id) {           \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle(itype_to_ctype(data), simd_lane_id));           \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_down(itype data, ushort delta) {             \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_down(itype_to_ctype(data), delta));             \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_down(itype data, ushort delta) {      \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_down(itype_to_ctype(data), delta));      \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_up(itype data, ushort delta) {        \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_up(itype_to_ctype(data), delta));        \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_up(itype data, ushort delta) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_up(itype_to_ctype(data), delta));               \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_xor(itype data, ushort mask) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_xor(itype_to_ctype(data), mask));               \
  }

#define instantiate_metal_simd_reduction_funcs(itype, otype, ctype)            \
                                                                               \
  METAL_FUNC otype simd_max(itype data) {                                      \
    return static_cast<otype>(__metal_simd_max(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_min(itype data) {                                      \
    return static_cast<otype>(__metal_simd_min(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_product(itype data) {                                  \
    return static_cast<otype>(__metal_simd_product(static_cast<ctype>(data))); \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_sum(itype data) {                                      \
    return static_cast<otype>(__metal_simd_sum(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_xor(itype data) {                                      \
    return static_cast<otype>(__metal_simd_xor(static_cast<ctype>(data)));     \
  }

namespace metal {

instantiate_metal_simd_comm_funcs(
    bfloat16_t,
    bfloat16_t,
    uint16_t,
    bfloat16_to_uint16,
    uint16_to_bfloat16);
instantiate_metal_simd_reduction_funcs(bfloat16_t, bfloat16_t, float);

} // namespace metal

// ---- embedded from Source/Cmlx/mlx-generated/metal/complex.h ----
// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_stdlib>

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

  // Constructors
  constexpr complex64_t(float real, float imag) : real(real), imag(imag) {};
  constexpr complex64_t() : real(0), imag(0) {};
  constexpr complex64_t() threadgroup : real(0), imag(0) {};

  // Conversions to complex64_t
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

  // Conversions from complex64_t
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

constexpr thread complex64_t& operator+=(thread complex64_t& a, complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr threadgroup complex64_t& operator+=(
    threadgroup complex64_t& a,
    complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr device complex64_t& operator+=(device complex64_t& a, complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr complex64_t operator+(float a, complex64_t b) {
  return {a + b.real, b.imag};
}
constexpr complex64_t operator+(complex64_t a, float b) {
  return {a.real + b, a.imag};
}

constexpr complex64_t operator-(complex64_t a, complex64_t b) {
  return {a.real - b.real, a.imag - b.imag};
}
constexpr complex64_t operator-(float a, complex64_t b) {
  return {a - b.real, -b.imag};
}
constexpr complex64_t operator-(complex64_t a, float b) {
  return {a.real - b, a.imag};
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

constexpr complex64_t operator/(float a, complex64_t b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a * b.real;
  auto y = -a * b.imag;
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

// ---- embedded from Source/Cmlx/mlx-generated/metal/defines.h ----
// Copyright © 2023 Apple Inc.

#pragma once

#if defined __METAL__ || defined MLX_METAL_JIT
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// ---- embedded from Source/Cmlx/mlx-generated/metal/logging.h ----
// Copyright © 2025 Apple Inc.

#pragma once

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
#include <metal_logging>

namespace mlx {
using os_log = metal::os_log;
} // namespace mlx

#else

namespace mlx {
struct os_log {
  constexpr os_log(constant char*, constant char*) constant {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const constant {}
};
} // namespace mlx

#endif

typedef half float16_t;

// Work per thread values for different types. The values here are expected to
// match get_work_per_thread in mlx/backend/metal/utils.h
template <typename U>
struct WorkPerThread {
  static_assert(sizeof(U) <= 8, "Type too large");
  static constexpr int constant n = 8 / sizeof(U);
};

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};

#define instantiate_default_limit(type)                                      \
  template <>                                                                \
  struct Limits<type> {                                                      \
    static constexpr constant type max = metal::numeric_limits<type>::max(); \
    static constexpr constant type min = metal::numeric_limits<type>::min(); \
    static constexpr constant type finite_max =                              \
        metal::numeric_limits<type>::max();                                  \
    static constexpr constant type finite_min =                              \
        metal::numeric_limits<type>::min();                                  \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type)             \
  template <>                                     \
  struct Limits<type> {                           \
    static constexpr constant type max =          \
        metal::numeric_limits<type>::infinity();  \
    static constexpr constant type min =          \
        -metal::numeric_limits<type>::infinity(); \
    static constexpr constant type finite_max =   \
        metal::numeric_limits<type>::max();       \
    static constexpr constant type finite_min =   \
        -metal::numeric_limits<type>::max();      \
  };

instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);

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

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    IdxT elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc =
      elem.x * IdxT(strides[ndim - 1]) + elem.y * IdxT(strides[ndim - 2]);
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * IdxT(strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Single Array with fixed N dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_1(uint elem, constant const int64_t& stride) {
  return elem * IdxT(stride);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_2(uint2 elem, constant const int64_t strides[2]) {
  return elem.x * IdxT(strides[1]) + elem.y * IdxT(strides[0]);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_3(uint3 elem, constant const int64_t strides[3]) {
  return elem.x * IdxT(strides[2]) + elem.y * IdxT(strides[1]) +
      elem.z * IdxT(strides[0]);
}

///////////////////////////////////////////////////////////////////////////////
// Multiple Arrays with generic dims

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 2> elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    int ndim) {
  vec<IdxT, 2> loc = {
      IdxT(
          elem.x * IdxT(a_strides[ndim - 1]) +
          IdxT(elem.y) * IdxT(a_strides[ndim - 2])),
      IdxT(
          elem.x * IdxT(b_strides[ndim - 1]) +
          elem.y * IdxT(b_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 3> elem_to_loc_3_nd(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  vec<IdxT, 3> loc = {
      IdxT(elem.x * IdxT(a_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(a_strides[ndim - 2])),
      IdxT(elem.x * IdxT(b_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(b_strides[ndim - 2])),
      IdxT(elem.x * IdxT(c_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(c_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    loc.z += l * IdxT(c_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Elem to loc in a loop utils
///////////////////////////////////////////////////////////////////////////////

template <int DIM, typename OffsetT = size_t, bool General = true>
struct LoopedElemToLoc {
  int dim;
  LoopedElemToLoc<DIM - 1, OffsetT, General> inner_looper;
  OffsetT offset{0};
  int index{0};

  LoopedElemToLoc(int dim) : dim(dim), inner_looper(dim - 1) {}

  void next(const constant int* shape, const constant int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index++;
    offset += OffsetT(strides[dim - 1]);
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index += n;
    offset += n * OffsetT(strides[dim - 1]);

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      if (extra >= shape[dim - 1]) {
        inner_looper.next(1 + extra / shape[dim - 1], shape, strides);
        extra = extra % shape[dim - 1];
      } else {
        inner_looper.next(shape, strides);
      }
      index = 0;
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, true> {
  int dim;
  OffsetT offset{0};
  uint index{0};

  LoopedElemToLoc(int dim) : dim(dim) {}

  void next(const constant int* shape, const constant int64_t* strides) {
    index++;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset += OffsetT(strides[0]);
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides) {
    index += n;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset = index * OffsetT(strides[0]);
    }
  }

  OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, false> {
  OffsetT offset{0};

  LoopedElemToLoc(int) {}

  void next(const constant int*, const constant int64_t* strides) {
    offset += OffsetT(strides[0]);
  }

  void next(int n, const constant int*, const constant int64_t* strides) {
    offset += n * OffsetT(strides[0]);
  }

  OffsetT location() {
    return offset;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Calculation utils
///////////////////////////////////////////////////////////////////////////////

/** Compute ceil((float)N/(float)M) */
template <typename T, typename U>
inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
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

inline complex64_t log1p(complex64_t in) {
  float x = in.real;
  float y = in.imag;
  float zabs = metal::precise::sqrt(x * x + y * y);
  float theta = metal::atan2(y, x + 1);
  if (zabs < 0.5f) {
    float r = x * (2 + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {0.5f * log1p(r), theta};
  } else {
    auto z0 = metal::sqrt((x + 1) * (x + 1) + y * y);
    return {metal::log(z0), theta};
  }
}

///////////////////////////////////////////////////////////////////////////////
// SIMD shuffle ops
///////////////////////////////////////////////////////////////////////////////

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

// std::conditional is not included with Metal
template <bool condition, typename T, typename U>
struct ConditionalType {
  using type = U;
};

template <typename T, typename U>
struct ConditionalType<true, T, U> {
  using type = T;
};

using namespace metal;

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

template <typename U>
struct ArgMin {
  static constexpr constant U init = Limits<U>::max;

  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val > current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U>
  reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] < best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
struct ArgMax {
  static constexpr constant U init = Limits<U>::min;

  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) {
    if (best.val < current.val ||
        (best.val == current.val && best.index > current.index)) {
      return current;
    } else {
      return best;
    }
  }

  template <int N>
  IndexValPair<U>
  reduce_many(IndexValPair<U> best, thread U* vals, uint32_t offset) {
    for (int i = 0; i < N; i++) {
      if (vals[i] > best.val) {
        best.val = vals[i];
        best.index = offset + i;
      }
    }
    return best;
  }
};

template <typename U>
IndexValPair<U> simd_shuffle_down(IndexValPair<U> data, uint16_t delta) {
  return IndexValPair<U>{
      simd_shuffle_down(data.index, delta), simd_shuffle_down(data.val, delta)};
}

template <typename T, typename Op, int N_READS = 4>
[[kernel]] void arg_reduce_general(
    const device T* in [[buffer(0)]],
    device uint32_t* out [[buffer(1)]],
    const constant int* shape [[buffer(2)]],
    const constant int64_t* in_strides [[buffer(3)]],
    const constant int64_t* out_strides [[buffer(4)]],
    const constant size_t& ndim [[buffer(5)]],
    const constant int64_t& axis_stride [[buffer(6)]],
    const constant size_t& axis_size [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 gsize [[threads_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Shapes and strides *do not* contain the reduction axis. The reduction size
  // and stride are provided in axis_stride and axis_size.
  //
  // Note: in shape == out shape with this convention.
  //
  // The sketch of the kernel is as follows.
  //    1. Launch prod(shape) * thread_group_size threads.
  //    2. Loop ceildiv(axis_size / lsize) times
  //    3. Read input values
  //    4. Reduce among them and go to 3
  //    4. Reduce in each simd_group
  //    6. Write in the thread local memory
  //    6. Reduce them across thread group
  //    7. Write the output without need for atomic
  Op op;

  // Compute the input/output index. There is one beginning and one output for
  // the whole threadgroup.
  int64_t row_idx = gid.y + static_cast<int64_t>(gsize.y) * gid.z;
  auto in_idx = elem_to_loc(row_idx, shape, in_strides, ndim);
  auto out_idx = elem_to_loc(row_idx, shape, out_strides, ndim);

  IndexValPair<T> best{0, Op::init};

  threadgroup IndexValPair<T> local_data[32];

  // Loop over the reduction axis in lsize*N_READS buckets
  for (uint r = 0; r < ceildiv(axis_size, N_READS * lsize.x); r++) {
    // Read the current value
    uint32_t current_index = r * lsize.x * N_READS + lid.x * N_READS;
    uint32_t offset = current_index;
    const device T* current_in = in + in_idx + current_index * axis_stride;
    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (current_index < axis_size) ? *current_in : T(Op::init);
      current_index++;
      current_in += axis_stride;
    }
    best = op.template reduce_many<N_READS>(best, vals, offset);
  }
  // At this point we have reduced the axis into thread group best values so we
  // need to reduce across the thread group.

  // First per simd reduction.
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    IndexValPair<T> neighbor = simd_shuffle_down(best, offset);
    best = op.reduce(best, neighbor);
  }

  // Write to the threadgroup memory
  if (simd_lane_id == 0) {
    local_data[simd_group_id] = best;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id != 0) {
    return;
  }

  // Read the appropriate value from local data and perform one simd reduction
  uint simd_groups = ceildiv(lsize.x, simd_size);
  if (simd_lane_id < simd_groups) {
    best = local_data[simd_lane_id];
  }
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    IndexValPair<T> neighbor = simd_shuffle_down(best, offset);
    best = op.reduce(best, neighbor);
  }

  // Finally write the output
  if (lid.x == 0) {
    out[out_idx] = best.index;
  }
}

// clang-format off
#define instantiate_arg_reduce(name, itype)                      \
  instantiate_kernel(                                            \
      "argmin_" #name, arg_reduce_general, itype, ArgMin<itype>) \
  instantiate_kernel(                                            \
      "argmax_" #name, arg_reduce_general, itype, ArgMax<itype>)

instantiate_arg_reduce(bool_, bool)
instantiate_arg_reduce(uint8, uint8_t)
instantiate_arg_reduce(uint16, uint16_t)
instantiate_arg_reduce(uint32, uint32_t)
instantiate_arg_reduce(uint64, uint64_t)
instantiate_arg_reduce(int8, int8_t)
instantiate_arg_reduce(int16, int16_t)
instantiate_arg_reduce(int32, int32_t)
instantiate_arg_reduce(int64, int64_t)
instantiate_arg_reduce(float16, half)
instantiate_arg_reduce(float32, float)
instantiate_arg_reduce(bfloat16, bfloat16_t) // clang-format on

// ---- embedded from Source/Cmlx/mlx-generated/metal/conv.metal ----
// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/conv/params.h ----
// Copyright © 2024 Apple Inc.

#pragma once

template <int NDIM>
struct MLXConvParams {
  int N; // Batch size
  int C; // In channels
  int O; // Out channels
  int iS[NDIM]; // Input spatial dim
  int wS[NDIM]; // Weight spatial dim
  int oS[NDIM]; // Output spatial dim
  int str[NDIM]; // Kernel strides
  int pad[NDIM]; // Input padding
  int kdil[NDIM]; // Kernel dilation
  int idil[NDIM]; // Input dilation
  int64_t in_strides[NDIM + 2]; // In strides
  int64_t wt_strides[NDIM + 2]; // Wt strides
  int64_t out_strides[NDIM + 2]; // Out strides
  int groups; // Input channel groups
  bool flip;

  static MLXConvParams<NDIM>
  with_padded_channels(MLXConvParams<NDIM> other, int pad_out, int pad_in) {
    MLXConvParams<NDIM> params = other;

    // Update strides
    for (int i = 0; i < NDIM + 1; i++) {
      params.in_strides[i] =
          (params.in_strides[i] / params.C) * (params.C + pad_in);
      params.wt_strides[i] =
          (params.wt_strides[i] / params.C) * (params.C + pad_in);
      params.out_strides[i] =
          (params.out_strides[i] / params.O) * (params.O + pad_out);
    }
    params.in_strides[NDIM + 1] = 1;
    params.wt_strides[NDIM + 1] = 1;
    params.out_strides[NDIM + 1] = 1;

    // Update channels
    params.C += pad_in;
    params.O += pad_out;

    return params;
  };
};

namespace mlx {
namespace steel {

struct ImplicitGemmConv2DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_c;

  const int tiles_n;
  const int tiles_m;
  const int swizzle_log;
};

struct ImplicitGemmConv3DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_d;
  const int inp_jump_c;

  const int tiles_n;
  const int tiles_m;
  const int swizzle_log;
};

struct Conv2DGeneralJumpParams {
  const int f_wgt_jump_h;
  const int f_wgt_jump_w;

  const int f_out_jump_h;
  const int f_out_jump_w;

  const int adj_out_h;
  const int adj_out_w;
  const int adj_out_hw;
  const int adj_implicit_m;
};

struct Conv2DGeneralBaseInfo {
  int weight_base;
  int weight_size;
};

} // namespace steel
} // namespace mlx

#define MLX_MTL_CONST static constant constexpr const

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// Naive unfold with dilation
///////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
[[kernel]] void naive_unfold_Nd(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant MLXConvParams<N>* params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  int filter_size = params->C;
  for (short i = 0; i < N; i++)
    filter_size *= params->wS[i];

  int out_pixels = 1;
  for (short i = 0; i < N; i++)
    out_pixels *= params->oS[i];

  // Set out
  out += (size_t)gid.z * filter_size + (size_t)gid.y * (params->C);

  // Coordinates in input
  int is[N] = {0};

  // gid.z: N oS (Batch and row in unfolded output)
  // gid.y: wS (Filter location to unfold input)
  // gid.x: C (channel)

  int n = (gid.z) / out_pixels;
  int oS = (gid.z) % out_pixels;
  int wS = gid.y;

  bool valid = n < params->N;

  // Unroll dimensions
  for (int i = N - 1; i >= 0; --i) {
    int os_ = (oS % params->oS[i]);
    int ws_ = (wS % params->wS[i]);

    ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;

    int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
    int is_max = 1 + params->idil[i] * (params->iS[i] - 1);

    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[i] == 0);

    is[i] = is_ / params->idil[i];

    oS /= params->oS[i];
    wS /= params->wS[i];
  }

  if (valid) {
    size_t in_offset = n * params->in_strides[0];

    for (int i = 0; i < N; ++i) {
      in_offset += is[i] * params->in_strides[i + 1];
    }

    out[gid.x] = in[in_offset + gid.x];
  } else {
    out[gid.x] = T(0);
  }
}

// This kernel unfolds the input array of size (N, *spatial_dims, C)
// into an array of size (N x *spatial_dims, C x *kernel_dims).
template <typename T, int N>
[[kernel]] void naive_unfold_transpose_Nd(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant MLXConvParams<N>* params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  int filter_size = params->C;
  for (short i = 0; i < N; i++)
    filter_size *= params->wS[i];

  int out_pixels = 1;
  for (short i = 0; i < N; i++)
    out_pixels *= params->oS[i];

  // Set out
  out +=
      (size_t)gid.z * filter_size + (size_t)gid.x * (filter_size / params->C);

  // Coordinates in input
  int is[N] = {0};

  // gid.z: N oS (Batch and row in unfolded output)
  // gid.y: wS (Filter location to unfold input)
  // gid.x: C (channel)

  int n = (gid.z) / out_pixels;
  int oS = (gid.z) % out_pixels;
  int wS = gid.y;

  bool valid = n < params->N;

  // Unroll dimensions
  int kernel_stride = 1;
  for (int i = N - 1; i >= 0; --i) {
    int os_ = (oS % params->oS[i]);
    int ws_ = (wS % params->wS[i]);
    out += ws_ * kernel_stride;

    ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;

    int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
    int is_max = 1 + params->idil[i] * (params->iS[i] - 1);

    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[i] == 0);

    is[i] = is_ / params->idil[i];

    oS /= params->oS[i];
    wS /= params->wS[i];

    kernel_stride *= params->wS[i];
  }

  if (valid) {
    size_t in_offset = n * params->in_strides[0];

    for (int i = 0; i < N; ++i) {
      in_offset += is[i] * params->in_strides[i + 1];
    }

    out[0] = in[in_offset + gid.x];
  } else {
    out[0] = T(0);
  }
}

#define instantiate_naive_unfold_nd(name, itype, n)                            \
  template [[host_name("naive_unfold_nd_" #name "_" #n)]] [[kernel]] void      \
  naive_unfold_Nd(                                                             \
      const device itype* in [[buffer(0)]],                                    \
      device itype* out [[buffer(1)]],                                         \
      const constant MLXConvParams<n>* params [[buffer(2)]],                   \
      uint3 gid [[thread_position_in_grid]]);                                  \
  template                                                                     \
      [[host_name("naive_unfold_transpose_nd_" #name "_" #n)]] [[kernel]] void \
      naive_unfold_transpose_Nd(                                               \
          const device itype* in [[buffer(0)]],                                \
          device itype* out [[buffer(1)]],                                     \
          const constant MLXConvParams<n>* params [[buffer(2)]],               \
          uint3 gid [[thread_position_in_grid]]);

#define instantiate_naive_unfold_nd_dims(name, itype)                      \
  instantiate_naive_unfold_nd(name, itype, 1) instantiate_naive_unfold_nd( \
      name, itype, 2) instantiate_naive_unfold_nd(name, itype, 3)

instantiate_naive_unfold_nd_dims(float32, float);
instantiate_naive_unfold_nd_dims(float16, half);
instantiate_naive_unfold_nd_dims(bfloat16, bfloat16_t);

///////////////////////////////////////////////////////////////////////////////
/// Depthwise convolution kernels
///////////////////////////////////////////////////////////////////////////////

constant int ker_h [[function_constant(00)]];
constant int ker_w [[function_constant(01)]];
constant int str_h [[function_constant(10)]];
constant int str_w [[function_constant(11)]];
constant int tgp_h [[function_constant(100)]];
constant int tgp_w [[function_constant(101)]];
constant bool do_flip [[function_constant(200)]];

constant int span_h = tgp_h * str_h + ker_h - 1;
constant int span_w = tgp_w * str_w + ker_w - 1;
constant int span_hw = span_h * span_w;

template <typename T>
[[kernel]] void depthwise_conv_2d(
    const device T* in [[buffer(0)]],
    const device T* wt [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant MLXConvParams<2>& params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int tc = 8;
  constexpr int tw = 8;
  constexpr int th = 4;

  constexpr int c_per_thr = 8;

  constexpr int TGH = th * 2 + 6;
  constexpr int TGW = tw * 2 + 6;
  constexpr int TGC = tc;

  threadgroup T ins[TGH * TGW * TGC];

  const int n_tgblocks_h = params.oS[0] / th;
  const int n = tid.z / n_tgblocks_h;
  const int tghid = tid.z % n_tgblocks_h;
  const int oh = tghid * th + lid.z;
  const int ow = gid.y;
  const int c = gid.x;

  in += n * params.in_strides[0];

  // Load in
  {
    constexpr int n_threads = th * tw * tc;
    const int tg_oh = (tghid * th) * str_h - params.pad[0];
    const int tg_ow = (tid.y * tw) * str_w - params.pad[1];
    const int tg_c = tid.x * tc;

    const int thread_idx = simd_gid * 32 + simd_lid;
    constexpr int thr_per_hw = tc / c_per_thr;
    constexpr int hw_per_group = n_threads / thr_per_hw;

    const int thr_c = thread_idx % thr_per_hw;
    const int thr_hw = thread_idx / thr_per_hw;

    for (int hw = thr_hw; hw < span_hw; hw += hw_per_group) {
      const int h = hw / span_w;
      const int w = hw % span_w;

      const int ih = tg_oh + h;
      const int iw = tg_ow + w;

      const int in_s_offset = h * span_w * TGC + w * TGC;

      if (ih >= 0 && ih < params.iS[0] && iw >= 0 && iw < params.iS[1]) {
        const auto in_load =
            in + ih * params.in_strides[1] + iw * params.in_strides[2] + tg_c;

        MLX_MTL_PRAGMA_UNROLL
        for (int cc = 0; cc < c_per_thr; ++cc) {
          ins[in_s_offset + c_per_thr * thr_c + cc] =
              in_load[c_per_thr * thr_c + cc];
        }
      } else {
        MLX_MTL_PRAGMA_UNROLL
        for (int cc = 0; cc < c_per_thr; ++cc) {
          ins[in_s_offset + c_per_thr * thr_c + cc] = T(0);
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  wt += c * params.wt_strides[0];

  const auto ins_ptr =
      &ins[lid.z * str_h * span_w * TGC + lid.y * str_w * TGC + lid.x];
  float o = 0.;
  for (int h = 0; h < ker_h; ++h) {
    for (int w = 0; w < ker_w; ++w) {
      int wt_h = h;
      int wt_w = w;
      if (do_flip) {
        wt_h = ker_h - h - 1;
        wt_w = ker_w - w - 1;
      }
      auto inv = ins_ptr[h * span_w * TGC + w * TGC];
      auto wtv = wt[wt_h * ker_w + wt_w];
      o += inv * wtv;
    }
  }
  threadgroup_barrier(mem_flags::mem_none);

  out += n * params.out_strides[0] + oh * params.out_strides[1] +
      ow * params.out_strides[2];
  out[c] = static_cast<T>(o);
}

#define instantiate_depthconv2d(iname, itype) \
  instantiate_kernel("depthwise_conv_2d_" #iname, depthwise_conv_2d, itype)

instantiate_depthconv2d(float32, float);
instantiate_depthconv2d(float16, half);
instantiate_depthconv2d(bfloat16, bfloat16_t);

template <typename T, typename IdxT>
[[kernel]] void depthwise_conv_1d(
    const device T* in [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const IdxT strides[3],
    constant const int& kernel_size,
    uint3 tid [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  out += (tid.z * static_cast<IdxT>(grid_dim.y) + tid.y) * grid_dim.x + tid.x;
  in += tid.z * strides[0] + tid.y * strides[1] + tid.x * strides[2];
  w += tid.x * kernel_size;

  float acc = 0.0;
  for (int i = 0; i < kernel_size; ++i) {
    acc += static_cast<float>(in[0]) * w[i];
    in += strides[1];
  }
  *out = static_cast<T>(acc);
}

#define instantiate_depthconv1d(iname, itype)                         \
  instantiate_kernel(                                                 \
      "depthwise_conv_1d_" #iname, depthwise_conv_1d, itype, int32_t) \
      instantiate_kernel(                                             \
          "depthwise_conv_1d_" #iname "_large",                       \
          depthwise_conv_1d,                                          \
          itype,                                                      \
          int64_t)

instantiate_depthconv1d(float32, float);
instantiate_depthconv1d(float16, half);
instantiate_depthconv1d(bfloat16, bfloat16_t);

///////////////////////////////////////////////////////////////////////////////
/// Winograd kernels
///////////////////////////////////////////////////////////////////////////////

template <int M, int R, int S>
struct WinogradTransforms {};

template <>
struct WinogradTransforms<6, 3, 8> {
  MLX_MTL_CONST int OUT_TILE_SIZE = 6;
  MLX_MTL_CONST int FILTER_SIZE = 3;
  MLX_MTL_CONST int IN_TILE_SIZE = OUT_TILE_SIZE + FILTER_SIZE - 1;
  MLX_MTL_CONST int SIMD_MATRIX_SIZE = 8;
  MLX_MTL_CONST float in_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
      {0.00f, 1.00f, -1.00f, 0.50f, -0.50f, 2.00f, -2.00f, -1.00f},
      {-5.25f, 1.00f, 1.00f, 0.25f, 0.25f, 4.00f, 4.00f, 0.00f},
      {0.00f, -4.25f, 4.25f, -2.50f, 2.50f, -2.50f, 2.50f, 5.25f},
      {5.25f, -4.25f, -4.25f, -1.25f, -1.25f, -5.00f, -5.00f, 0.00f},
      {0.00f, 1.00f, -1.00f, 2.00f, -2.00f, 0.50f, -0.50f, -5.25f},
      {-1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 0.00f},
      {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
  };

  MLX_MTL_CONST float out_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
      {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
      {1.00f, -1.00f, 1.00f, -1.00f, 1.00f, -1.00f},
      {1.00f, 2.00f, 4.00f, 8.00f, 16.00f, 32.00f},
      {1.00f, -2.00f, 4.00f, -8.00f, 16.00f, -32.00f},
      {1.00f, 0.50f, 0.25f, 0.125f, 0.0625f, 0.03125f},
      {1.00f, -0.50f, 0.25f, -0.125f, 0.0625f, -0.03125f},
      {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
  };

  MLX_MTL_CONST float wt_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00, 0.00, 0.00},
      {-2.0 / 9.00, -2.0 / 9.00, -2.0 / 9.00},
      {-2.0 / 9.00, 2.0 / 9.00, -2.0 / 9.00},
      {1.0 / 90.0, 1.0 / 45.0, 2.0 / 45.0},
      {1.0 / 90.0, -1.0 / 45.0, 2.0 / 45.0},
      {32.0 / 45.0, 16.0 / 45.0, 8.0 / 45.0},
      {32.0 / 45.0, -16.0 / 45.0, 8.0 / 45.0},
      {0.00, 0.00, 1.00},
  };
};

constant constexpr const float WinogradTransforms<6, 3, 8>::wt_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::in_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::out_transform[8][8];

template <typename T, int BC = 32, int BO = 4, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(BO * 32)]] void
winograd_conv_2d_weight_transform(
    const device T* wt_in [[buffer(0)]],
    device T* wt_out [[buffer(1)]],
    const constant int& C [[buffer(2)]],
    const constant int& O [[buffer(3)]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  using WGT = WinogradTransforms<M, R, 8>;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize G matrix
  simdgroup_matrix<float, 8, 8> G;
  G.thread_elements()[0] = WGT::wt_transform[sm][sn];
  G.thread_elements()[1] = WGT::wt_transform[sm][sn + 1];

  // Initialize Gt matrix
  simdgroup_matrix<float, 8, 8> Gt;
  Gt.thread_elements()[0] = WGT::wt_transform[sn][sm];
  Gt.thread_elements()[1] = WGT::wt_transform[sn + 1][sm];

  // Move to the correct output filter
  size_t ko = BO * tid + simd_group_id;
  wt_in += ko * R * R * C;

  // wt_out is stored transposed (A x A x C x O)
  short ohw_0 = sm * 8 + sn;
  short ohw_1 = sm * 8 + sn + 1;
  device T* wt_out_0 = wt_out + ohw_0 * C * O + ko;
  device T* wt_out_1 = wt_out + ohw_1 * C * O + ko;

  // Prepare shared memory
  threadgroup T Ws[BO][R][R][BC];

  // Loop over C
  for (int bc = 0; bc < C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for (int kh = 0; kh < R; ++kh) {
      for (int kw = 0; kw < R; ++kw) {
        for (int kc = simd_lane_id; kc < BC; kc += 32) {
          Ws[simd_group_id][kh][kw][kc] = wt_in[kh * R * C + kw * C + kc];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = 0; c < BC; ++c) {
      simdgroup_matrix<float, 8, 8> g;
      g.thread_elements()[0] =
          sm < R && sn < R ? Ws[simd_group_id][sm][sn][c] : T(0);
      g.thread_elements()[1] =
          sm < R && sn + 1 < R ? Ws[simd_group_id][sm][sn + 1][c] : T(0);

      simdgroup_matrix<float, 8, 8> g_out = (G * g) * Gt;
      wt_out_0[c * O] = static_cast<T>(g_out.thread_elements()[0]);
      wt_out_1[c * O] = static_cast<T>(g_out.thread_elements()[1]);
    }

    wt_in += BC;
    wt_out_0 += BC * O;
    wt_out_1 += BC * O;
  }
}

#define instantiate_winograd_conv_2d_weight_transform_base(name, itype, bc)   \
  template [[host_name(                                                       \
      "winograd_conv_2d_weight_transform_" #name "_bc" #bc)]] [[kernel]] void \
  winograd_conv_2d_weight_transform<itype, bc>(                               \
      const device itype* wt_in [[buffer(0)]],                                \
      device itype* wt_out [[buffer(1)]],                                     \
      const constant int& C [[buffer(2)]],                                    \
      const constant int& O [[buffer(3)]],                                    \
      uint tid [[threadgroup_position_in_grid]],                              \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T, int BC, int WM, int WN, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
winograd_conv_2d_input_transform(
    const device T* inp_in [[buffer(0)]],
    device T* inp_out [[buffer(1)]],
    const constant MLXConvParams<2>& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)lid;

  using WGT = WinogradTransforms<M, R, 8>;
  constexpr int A = WGT::IN_TILE_SIZE;
  constexpr int N_SIMD_GROUPS = WM * WN;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize B matrix
  simdgroup_matrix<float, 8, 8> B;
  B.thread_elements()[0] = WGT::in_transform[sm][sn];
  B.thread_elements()[1] = WGT::in_transform[sm][sn + 1];

  // Initialize Bt matrix
  simdgroup_matrix<float, 8, 8> Bt;
  Bt.thread_elements()[0] = WGT::in_transform[sn][sm];
  Bt.thread_elements()[1] = WGT::in_transform[sn + 1][sm];

  // Resolve input tile
  constexpr int TH = (A / WM);
  constexpr int TW = (A / WN);
  int kh = TH * (simd_group_id / WN);
  int kw = TW * (simd_group_id % WN);
  int bh = M * tid.y + kh;
  int bw = M * tid.x + kw;

  // Move to the correct input tile
  inp_in += tid.z * params.in_strides[0] + bh * params.in_strides[1] +
      bw * params.in_strides[2];

  // Pre compute strides
  int jump_in[TH][TW];

  for (int h = 0; h < TH; h++) {
    for (int w = 0; w < TW; w++) {
      jump_in[h][w] = h * params.in_strides[1] + w * params.in_strides[2];
    }
  }

  // inp_out is stored interleaved (A x A x tiles x C)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id =
      tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  device T* inp_out_0 =
      inp_out + ohw_0 * N_TILES * params.C + tile_id * params.C;
  device T* inp_out_1 =
      inp_out + ohw_1 * N_TILES * params.C + tile_id * params.C;

  // Prepare shared memory
  threadgroup T Is[A][A][BC];

  // Loop over C
  for (int bc = 0; bc < params.C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for (int h = 0; h < TH; h++) {
      for (int w = 0; w < TW; w++) {
        const device T* in_ptr = inp_in + jump_in[h][w];
        for (int c = simd_lane_id; c < BC; c += 32) {
          Is[kh + h][kw + w][c] = in_ptr[c];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = simd_group_id; c < BC; c += N_SIMD_GROUPS) {
      simdgroup_matrix<float, 8, 8> I;
      I.thread_elements()[0] = Is[sm][sn][c];
      I.thread_elements()[1] = Is[sm][sn + 1][c];

      simdgroup_matrix<float, 8, 8> I_out = (Bt * I) * B;
      inp_out_0[c] = static_cast<T>(I_out.thread_elements()[0]);
      inp_out_1[c] = static_cast<T>(I_out.thread_elements()[1]);
    }

    inp_in += BC;
    inp_out_0 += BC;
    inp_out_1 += BC;
  }
}

#define instantiate_winograd_conv_2d_input_transform(name, itype, bc)        \
  template [[host_name(                                                      \
      "winograd_conv_2d_input_transform_" #name "_bc" #bc)]] [[kernel]] void \
  winograd_conv_2d_input_transform<itype, bc, 2, 2>(                         \
      const device itype* inp_in [[buffer(0)]],                              \
      device itype* inp_out [[buffer(1)]],                                   \
      const constant MLXConvParams<2>& params [[buffer(2)]],                 \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                          \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                 \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T, int BO, int WM, int WN, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
winograd_conv_2d_output_transform(
    const device T* out_in [[buffer(0)]],
    device T* out_out [[buffer(1)]],
    const constant MLXConvParams<2>& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)lid;

  using WGT = WinogradTransforms<M, R, 8>;
  constexpr int N_SIMD_GROUPS = WM * WN;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize A matrix
  simdgroup_matrix<float, 8, 8> B;
  B.thread_elements()[0] = WGT::out_transform[sm][sn];
  B.thread_elements()[1] = WGT::out_transform[sm][sn + 1];

  // Initialize At matrix
  simdgroup_matrix<float, 8, 8> Bt;
  Bt.thread_elements()[0] = WGT::out_transform[sn][sm];
  Bt.thread_elements()[1] = WGT::out_transform[sn + 1][sm];

  // Out_in comes in shape (A x A x tiles x O)
  // We do transform and then write out to out_out in shape (N, H, W, O)

  // Resolve output tile
  constexpr int TH = (M / WM);
  constexpr int TW = (M / WN);
  int kh = TH * (simd_group_id / WN);
  int kw = TW * (simd_group_id % WN);
  int bh = M * tid.y + kh;
  int bw = M * tid.x + kw;

  // Move to the correct input tile
  out_out += tid.z * params.out_strides[0] + bh * params.out_strides[1] +
      bw * params.out_strides[2];

  // Pre compute strides
  int jump_in[TH][TW];

  for (int h = 0; h < TH; h++) {
    for (int w = 0; w < TW; w++) {
      bool valid = ((bh + h) < params.oS[0]) && ((bw + w) < params.oS[1]);
      jump_in[h][w] =
          valid ? h * params.out_strides[1] + w * params.out_strides[2] : -1;
    }
  }

  // out_in is stored interleaved (A x A x tiles x O)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id =
      tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  const device T* out_in_0 =
      out_in + ohw_0 * N_TILES * params.O + tile_id * params.O;
  const device T* out_in_1 =
      out_in + ohw_1 * N_TILES * params.O + tile_id * params.O;

  // Prepare shared memory
  threadgroup T Os[M][M][BO];

  // Loop over O
  for (int bo = 0; bo < params.O; bo += BO) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = simd_group_id; c < BO; c += N_SIMD_GROUPS) {
      simdgroup_matrix<float, 8, 8> O_mat;
      O_mat.thread_elements()[0] = out_in_0[c];
      O_mat.thread_elements()[1] = out_in_1[c];

      simdgroup_matrix<float, 8, 8> O_out = (Bt * (O_mat * B));
      if ((sm < M) && (sn < M)) {
        Os[sm][sn][c] = static_cast<T>(O_out.thread_elements()[0]);
      }
      if ((sm < M) && ((sn + 1) < M)) {
        Os[sm][sn + 1][c] = static_cast<T>(O_out.thread_elements()[1]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read out from shared memory
    for (int h = 0; h < TH; h++) {
      for (int w = 0; w < TW; w++) {
        if (jump_in[h][w] >= 0) {
          device T* out_ptr = out_out + jump_in[h][w];
          for (int c = simd_lane_id; c < BO; c += 32) {
            out_ptr[c] = Os[kh + h][kw + w][c];
          }
        }
      }
    }

    out_out += BO;
    out_in_0 += BO;
    out_in_1 += BO;
  }
}

#define instantiate_winograd_conv_2d_output_transform(name, itype, bo)        \
  template [[host_name(                                                       \
      "winograd_conv_2d_output_transform_" #name "_bo" #bo)]] [[kernel]] void \
  winograd_conv_2d_output_transform<itype, bo, 2, 2>(                         \
      const device itype* out_in [[buffer(0)]],                               \
      device itype* out_out [[buffer(1)]],                                    \
      const constant MLXConvParams<2>& params [[buffer(2)]],                  \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint3 lid [[thread_position_in_threadgroup]],                           \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                           \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

// clang-format off
#define instantiate_winograd_conv_2d(name, itype)                     \
  instantiate_winograd_conv_2d_weight_transform_base(name, itype, 32) \
  instantiate_winograd_conv_2d_input_transform(name, itype, 32)       \
  instantiate_winograd_conv_2d_output_transform(name, itype, 32) // clang-format on

// clang-format off
instantiate_winograd_conv_2d(float32, float);
instantiate_winograd_conv_2d(bfloat16, bfloat16_t);
instantiate_winograd_conv_2d(float16, half); // clang-format on

// ---- embedded from Source/Cmlx/mlx-generated/metal/gemv.metal ----
// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>



// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/utils.h ----
// Copyright © 2024 Apple Inc.

#pragma once

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

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_CONST static constant constexpr const

template <typename U>
struct DefaultAccT {
  using type = float;
};
template <>
struct DefaultAccT<complex64_t> {
  using type = complex64_t;
};

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
    typename AccT = typename DefaultAccT<T>::type>
struct GEMVKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  static_assert(
      SN == 4 || SN == 8 || SN == 16 || SN == 32,
      "gemv block must have a width of 4, 8, 16, or 32");

  // - The matrix of size (M = out_vec_size, K = in_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM rows
  //    and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for
  //    the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated blockM outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? BN*(blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  template <typename U = T>
  static METAL_FUNC void
  load_unsafe(const device T* src, thread U dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = static_cast<U>(src[src_offset + tn]);
    }
  }

  template <typename U = T>
  static METAL_FUNC void load_safe(
      const device T* src,
      thread U dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else { // Edgecase
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      const device T* bias [[buffer(2)]],
      device T* out_vec [[buffer(3)]],
      const constant int& in_vec_size [[buffer(4)]],
      const constant int& out_vec_size [[buffer(5)]],
      const constant int& matrix_ld [[buffer(6)]],
      const constant float& alpha [[buffer(7)]],
      const constant float& beta [[buffer(8)]],
      const constant int& bias_stride [[buffer(14)]],
      threadgroup AccT* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    thread AccT result[TM] = {0};
    thread T inter[TN];
    thread AccT v_coeff[TN];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position
    int out_row = tid.x * blockM + bm;

    // Exit simdgroup if rows out of bound
    if (out_row >= out_vec_size)
      return;

    // Adjust tail simdgroup to ensure in bound reads
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    // Advance matrix
    mat += out_row * matrix_ld;

    constexpr const uniform<int> loop_stride = make_uniform(blockN);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Loop over in_vec in blocks of blockN
    for (int i = 0; i < n_iter; ++i) {
      load_unsafe<AccT>(in_vec, v_coeff, bn);

      // Per thread work loop
      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_unsafe(mat, inter, mat_offset + bn);

        // Accumulate results
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }

        mat_offset += matrix_ld;
      }

      bn += blockN;
    }

    if (leftover > 0) {
      load_safe<AccT>(in_vec, v_coeff, bn, in_size);

      // Per thread work loop
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_safe(&mat[tm * matrix_ld], inter, bn, in_size);

        // Accumulate results
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results[tm] = result[tm];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    // Write outputs
    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        if (kDoAxpby) {
          out_vec[out_row + tm] =
              static_cast<T>(alpha) * static_cast<T>(result[tm]) +
              static_cast<T>(beta) * bias[(out_row + tm) * bias_stride];
        } else {
          out_vec[out_row + tm] = static_cast<T>(result[tm]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoAxpby, /* Do out = alpha * out + beta * bias */
    typename AccT = typename DefaultAccT<T>::type>
struct GEMVTKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  // - The matrix of size (M = in_vec_size, N = out_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows
  //    and the corresponding scalar from the vector
  // 2. The thread then accumulates its local result for the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BM > 1 ? BM*(blockN + TN) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BM > 1;

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      const device T* bias [[buffer(2)]],
      device T* out_vec [[buffer(3)]],
      const constant int& in_vec_size [[buffer(4)]],
      const constant int& out_vec_size [[buffer(5)]],
      const constant int& marix_ld [[buffer(6)]],
      const constant float& alpha [[buffer(7)]],
      const constant float& beta [[buffer(8)]],
      const constant int& bias_stride [[buffer(14)]],
      threadgroup AccT* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    AccT result[TN] = {0};
    T inter[TN];
    AccT v_coeff[TM];
    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = SM * sgM;
    const int simdN = SN * sgN;

    int cm = (simdM + thrM);
    int cn = (simdN + thrN);

    int bm = cm * TM;
    int bn = cn * TN;

    int out_col = tid.x * blockN + bn;

    constexpr const uniform<int> loop_stride = make_uniform(blockM);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Edgecase handling
    if (out_col < out_vec_size) {
      out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;

      // Per thread accumulation main loop
      for (int i = 0; i < n_iter; ++i) {
        // Adding a threadgroup_barrier improves performance slightly
        // This is possibly it may help exploit cache better
        threadgroup_barrier(mem_flags::mem_none);

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
        }

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          auto vc = static_cast<AccT>(v_coeff[tm]);
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
          }
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += vc * inter[tn];
          }
        }

        bm += blockM;
      }

      if (leftover > 0) {
        for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);

          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
          }

          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += v_coeff[tm] * inter[tn];
          }
        }
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
      if (thrM == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          tgp_results[tn] = result[tn];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgM == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgm = 1; sgm < BM; sgm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += tgp_results[sgm * (blockN + TN) + tn];
            }
          }
        }
      }
    }

    // Threadgroup accumulation and writing out results
    if (cm == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        if (kDoAxpby) {
          out_vec[out_col + j] =
              static_cast<T>(alpha) * static_cast<T>(result[j]) +
              static_cast<T>(beta) * bias[(out_col + j) * bias_stride];
        } else {
          out_vec[out_col + j] = static_cast<T>(result[j]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch, /* Batch ndim > 1 */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  threadgroup typename gemv_kernel::acc_type tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  // Update batch offsets
  if (kDoNCBatch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (kDoAxpby) {
      bias += elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
    }

  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (kDoAxpby) {
      bias += tid.z * bias_batch_stride[0];
    }
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      bias_stride,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

#define instantiate_gemv_helper(                                      \
    name, itype, bm, bn, sm, sn, tm, tn, nc, axpby)                   \
  instantiate_kernel(                                                 \
      "gemv_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm \
      "_tn" #tn "_nc" #nc "_axpby" #axpby,                            \
      gemv,                                                           \
      itype,                                                          \
      bm,                                                             \
      bn,                                                             \
      sm,                                                             \
      sn,                                                             \
      tm,                                                             \
      tn,                                                             \
      nc,                                                             \
      axpby)

// clang-format off
#define instantiate_gemv(name, itype, bm, bn, sm, sn, tm, tn)        \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 1) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 0) \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 1) // clang-format on

// clang-format off
#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 1,  8, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 1,  8, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 1,  1, 8,  4, 4, 4) \
  instantiate_gemv(name, itype, 1,  1, 8,  4, 1, 4) \
  instantiate_gemv(name, itype, 4,  1, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 4,  1, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 8,  1, 1, 32, 4, 4) // clang-format on

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat16_t);
instantiate_gemv_blocks(complex64, complex64_t);

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void gemv_gather(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* index_batch_strides [[buffer(11)]],
    const constant int& vector_batch_ndim [[buffer(12)]],
    const constant int* vector_batch_shape [[buffer(13)]],
    const constant int64_t* vector_batch_stride [[buffer(14)]],
    const constant int& matrix_batch_ndim [[buffer(15)]],
    const constant int* matrix_batch_shape [[buffer(16)]],
    const constant int64_t* matrix_batch_stride [[buffer(17)]],
    const constant uint32_t* vec_indices [[buffer(18)]],
    const constant uint32_t* mat_indices [[buffer(19)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVKernel<T, BM, BN, SM, SN, TM, TN, false>;
  threadgroup typename gemv_kernel::acc_type tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  uint32_t indx_vec;
  uint32_t indx_mat;

  // Update batch offsets
  if (batch_ndim > 1) {
    const constant auto* veci_bstrides = index_batch_strides;
    const constant auto* mati_bstrides = index_batch_strides + batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, veci_bstrides, mati_bstrides, batch_ndim);

    indx_vec = vec_indices[batch_offsets.x];
    indx_mat = mat_indices[batch_offsets.y];

  } else {
    indx_vec = vec_indices[index_batch_strides[0] * tid.z];
    indx_mat = mat_indices[index_batch_strides[batch_ndim] * tid.z];
  }

  if (vector_batch_ndim > 1) {
    in_vec += elem_to_loc(
        indx_vec, vector_batch_shape, vector_batch_stride, vector_batch_ndim);
  } else {
    in_vec += indx_vec * vector_batch_stride[0];
  }

  if (matrix_batch_ndim > 1) {
    mat += elem_to_loc(
        indx_mat, matrix_batch_shape, matrix_batch_stride, matrix_batch_ndim);
  } else {
    mat += indx_mat * matrix_batch_stride[0];
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      batch_ndim, // Not used
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

// clang-format off
#define instantiate_gemv_bs_helper(nm, itype, bm, bn, sm, sn, tm, tn) \
  instantiate_kernel(                                                 \
    "gemv_gather_" #nm "_bm" #bm "_bn" #bn "_sm" #sm                  \
                       "_sn" #sn "_tm" #tm "_tn" #tn,                 \
    gemv_gather, itype, bm, bn, sm, sn, tm, tn)

#define instantiate_gemv_bs_blocks(name, itype)              \
  instantiate_gemv_bs_helper(name, itype, 4, 1, 1, 32, 1, 4) \
  instantiate_gemv_bs_helper(name, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv_bs_helper(name, itype, 8, 1, 1, 32, 4, 4) // clang-format on

instantiate_gemv_bs_blocks(float32, float);
instantiate_gemv_bs_blocks(float16, half);
instantiate_gemv_bs_blocks(bfloat16, bfloat16_t);
instantiate_gemv_bs_blocks(complex64, complex64_t);

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN, /* Thread cols (in elements) */
    const bool kDoNCBatch, /* Batch ndim > 1 */
    const bool kDoAxpby> /* Do out = alpha * out + beta * bias */
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  threadgroup typename gemv_kernel::acc_type tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  // Update batch offsets
  if (kDoNCBatch) {
    in_vec += elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat += elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);

    if (kDoAxpby) {
      bias += elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
    }

  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat += tid.z * matrix_batch_stride[0];

    if (kDoAxpby) {
      bias += tid.z * bias_batch_stride[0];
    }
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      bias_stride,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

// clang-format off
#define instantiate_gemv_t_helper(                          \
    name, itype, bm, bn, sm, sn, tm, tn, nc, axpby)         \
  instantiate_kernel(                                       \
    "gemv_t_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn \
       "_tm" #tm "_tn" #tn "_nc" #nc "_axpby" #axpby,       \
  gemv_t, itype, bm, bn, sm, sn, tm, tn, nc, axpby)

#define instantiate_gemv_t(name, itype, bm, bn, sm, sn, tm, tn)        \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 0) \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 0, 1) \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 0) \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 1, 1) // clang-format on

// clang-format off
#define instantiate_gemv_t_blocks(name, itype) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 4, 1) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 4,  8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 4, 8, 4, 4) // clang-format on

// clang-format off
instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t);
instantiate_gemv_t_blocks(complex64, complex64_t); // clang-format on

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]] void gemv_t_gather(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* index_batch_strides [[buffer(11)]],
    const constant int& vector_batch_ndim [[buffer(12)]],
    const constant int* vector_batch_shape [[buffer(13)]],
    const constant int64_t* vector_batch_stride [[buffer(14)]],
    const constant int& matrix_batch_ndim [[buffer(15)]],
    const constant int* matrix_batch_shape [[buffer(16)]],
    const constant int64_t* matrix_batch_stride [[buffer(17)]],
    const constant uint32_t* vec_indices [[buffer(18)]],
    const constant uint32_t* mat_indices [[buffer(19)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, false>;
  threadgroup typename gemv_kernel::acc_type tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  uint32_t indx_vec;
  uint32_t indx_mat;

  // Update batch offsets
  if (batch_ndim > 1) {
    const constant auto* veci_bstrides = index_batch_strides;
    const constant auto* mati_bstrides = index_batch_strides + batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, veci_bstrides, mati_bstrides, batch_ndim);

    indx_vec = vec_indices[batch_offsets.x];
    indx_mat = mat_indices[batch_offsets.y];

  } else {
    indx_vec = vec_indices[index_batch_strides[0] * tid.z];
    indx_mat = mat_indices[index_batch_strides[batch_ndim] * tid.z];
  }

  if (vector_batch_ndim > 1) {
    in_vec += elem_to_loc(
        indx_vec, vector_batch_shape, vector_batch_stride, vector_batch_ndim);
  } else {
    in_vec += indx_vec * vector_batch_stride[0];
  }

  if (matrix_batch_ndim > 1) {
    mat += elem_to_loc(
        indx_mat, matrix_batch_shape, matrix_batch_stride, matrix_batch_ndim);
  } else {
    mat += indx_mat * matrix_batch_stride[0];
  }

  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      bias,
      out_vec,
      in_vec_size,
      out_vec_size,
      marix_ld,
      alpha,
      beta,
      batch_ndim, // Not used,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

// clang-format off
#define instantiate_gemv_t_bs_helper(                  \
    nm, itype, bm, bn, sm, sn, tm, tn)                 \
  instantiate_kernel(                                  \
    "gemv_t_gather_" #nm "_bm" #bm "_bn" #bn "_sm" #sm \
       "_sn" #sn "_tm" #tm "_tn" #tn,                  \
  gemv_t_gather, itype, bm, bn, sm, sn, tm, tn)

#define instantiate_gemv_t_bs_blocks(name, itype)              \
  instantiate_gemv_t_bs_helper(name, itype, 1,  2, 8, 4, 4, 1) \
  instantiate_gemv_t_bs_helper(name, itype, 1,  2, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1,  4, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1, 16, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1, 16, 4, 8, 4, 4) // clang-format on

// clang-format off
instantiate_gemv_t_bs_blocks(float32, float);
instantiate_gemv_t_bs_blocks(float16, half);
instantiate_gemv_t_bs_blocks(bfloat16, bfloat16_t);
instantiate_gemv_t_bs_blocks(complex64, complex64_t); // clang-format on

// ---- embedded from Source/Cmlx/mlx-generated/metal/layer_norm.metal ----
// Copyright © 2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>


using namespace metal;

constant bool has_w [[function_constant(20)]];

template <int N = 1>
inline void initialize_buffer(
    threadgroup float* xs,
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  if (simd_group_id == 0) {
    for (int i = 0; i < N; i++) {
      xs[N * simd_lane_id + i] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <int N = 1>
inline void threadgroup_sum(
    thread float* x,
    threadgroup float* xs,
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  for (int i = 0; i < N; i++) {
    x[i] = simd_sum(x[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    for (int i = 0; i < N; i++) {
      xs[N * simd_group_id + i] = x[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < N; i++) {
    x[i] = xs[N * simd_lane_id + i];
    x[i] = simd_sum(x[i]);
  }
}

template <typename T, int N_READS = 8>
[[kernel]] void layer_norm_single_row(
    const device T* x,
    const device T* w,
    const device T* b,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    constant uint& b_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  // Initialize the registers and threadgroup memory
  float thread_x[N_READS] = {0};
  threadgroup float local_buffer[SIMD_SIZE] = {0};
  initialize_buffer(local_buffer, simd_lane_id, simd_group_id);

  // Advance the pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  b += b_stride * lid * N_READS;
  out += gid * size_t(axis_size) + lid * N_READS;

  // Compute some variables for reading writing etc
  const bool safe = lid * N_READS + N_READS <= axis_size;
  const int n = axis_size - lid * N_READS;

  // Read the inputs
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] = x[i];
    }
  }

  // Compute the mean
  float mean = 0;
  for (int i = 0; i < N_READS; i++) {
    mean += thread_x[i];
  }
  threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  // Compute the normalizer
  float normalizer = 0;
  if (!safe) {
    for (int i = n; i < N_READS; i++) {
      thread_x[i] = mean;
    }
  }
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] -= mean;
    normalizer += thread_x[i] * thread_x[i];
  }
  threadgroup_sum(&normalizer, local_buffer, simd_lane_id, simd_group_id);
  normalizer = metal::precise::rsqrt(normalizer / axis_size + eps);

  // Write the outputs
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] *= normalizer;
      out[i] = w[w_stride * i] * static_cast<T>(thread_x[i]) + b[b_stride * i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] *= normalizer;
      out[i] = w[w_stride * i] * static_cast<T>(thread_x[i]) + b[b_stride * i];
    }
  }
}

template <typename T, int N_READS = 4>
[[kernel]] void layer_norm_looped(
    const device T* x,
    const device T* w,
    const device T* b,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    constant uint& b_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_buffer[SIMD_SIZE];
  initialize_buffer(local_buffer, simd_lane_id, simd_group_id);

  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  b += b_stride * lid * N_READS;

  // Compute the mean
  float mean = 0;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        mean += x[i + r];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          mean += x[i + r];
        }
      }
    }
  }
  threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  // Compute the normalizer
  float normalizer = 0;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float t = x[i + r] - mean;
        normalizer += t * t;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float t = x[i + r] - mean;
          normalizer += t * t;
        }
      }
    }
  }
  threadgroup_sum(&normalizer, local_buffer, simd_lane_id, simd_group_id);
  normalizer = metal::precise::rsqrt(normalizer / axis_size + eps);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = (x[r + i] - mean) * normalizer;
        out[r + i] =
            w[w_stride * (i + r)] * static_cast<T>(xi) + b[b_stride * (i + r)];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = (x[r + i] - mean) * normalizer;
          out[r + i] = w[w_stride * (i + r)] * static_cast<T>(xi) +
              b[b_stride * (i + r)];
        }
      }
    }
  }
}

template <typename T, int N_READS = 8>
[[kernel]] void vjp_layer_norm_single_row(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Initialize the registers and threadgroup memory
  float thread_x[N_READS] = {0};
  float thread_w[N_READS] = {0};
  float thread_g[N_READS] = {0};
  threadgroup float local_buffer[3 * SIMD_SIZE];
  initialize_buffer<3>(local_buffer, simd_lane_id, simd_group_id);

  // Compute some variables for reading writing etc
  const bool safe = lid * N_READS + N_READS <= axis_size;
  const int n = axis_size - lid * N_READS;

  // Read the inputs
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      thread_g[i] = g[i];
      thread_w[i] = w[i * w_stride];
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] = x[i];
      thread_g[i] = g[i];
      thread_w[i] = w[i * w_stride];
    }
  }

  // Compute the mean
  float mean = 0;
  for (int i = 0; i < N_READS; i++) {
    mean += thread_x[i];
  }
  threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  // Compute the neccesary scaling factors using the mean
  if (!safe) {
    for (int i = n; i < N_READS; i++) {
      thread_x[i] = mean;
    }
  }
  float factors[3] = {0};
  constexpr int meanwg = 0;
  constexpr int meanwgxc = 1;
  constexpr int normalizer2 = 2;
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] -= mean;
    factors[meanwg] += thread_w[i] * thread_g[i];
    factors[meanwgxc] += thread_w[i] * thread_g[i] * thread_x[i];
    factors[normalizer2] += thread_x[i] * thread_x[i];
  }
  threadgroup_sum<3>(factors, local_buffer, simd_lane_id, simd_group_id);
  factors[meanwg] /= axis_size;
  factors[meanwgxc] /= axis_size;
  factors[normalizer2] = 1 / (factors[normalizer2] / axis_size + eps);
  float normalizer = metal::precise::sqrt(factors[normalizer2]);

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] *= normalizer;
      gx[i] = static_cast<T>(
          normalizer * (thread_w[i] * thread_g[i] - factors[meanwg]) -
          thread_x[i] * factors[meanwgxc] * factors[normalizer2]);
      if (has_w) {
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i]);
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] *= normalizer;
      gx[i] = static_cast<T>(
          normalizer * (thread_w[i] * thread_g[i] - factors[meanwg]) -
          thread_x[i] * factors[meanwgxc] * factors[normalizer2]);
      if (has_w) {
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i]);
      }
    }
  }
}

template <typename T, int N_READS = 4>
[[kernel]] void vjp_layer_norm_looped(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  threadgroup float local_buffer[3 * SIMD_SIZE];
  initialize_buffer<3>(local_buffer, simd_lane_id, simd_group_id);

  // Compute the mean
  float mean = 0;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        mean += x[i + r];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          mean += x[i + r];
        }
      }
    }
  }
  threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  // Compute the neccesary scaling factors using the mean
  float factors[3] = {0};
  constexpr int meanwg = 0;
  constexpr int meanwgxc = 1;
  constexpr int normalizer2 = 2;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float t = x[i + r] - mean;
        float wi = w[(i + r) * w_stride];
        float gi = g[i + r];
        float wg = wi * gi;
        factors[meanwg] += wg;
        factors[meanwgxc] += wg * t;
        factors[normalizer2] += t * t;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float t = x[i + r] - mean;
          float wi = w[(i + r) * w_stride];
          float gi = g[i + r];
          float wg = wi * gi;
          factors[meanwg] += wg;
          factors[meanwgxc] += wg * t;
          factors[normalizer2] += t * t;
        }
      }
    }
  }
  threadgroup_sum<3>(factors, local_buffer, simd_lane_id, simd_group_id);
  factors[meanwg] /= axis_size;
  factors[meanwgxc] /= axis_size;
  factors[normalizer2] = 1 / (factors[normalizer2] / axis_size + eps);
  float normalizer = metal::precise::sqrt(factors[normalizer2]);

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = (x[i + r] - mean) * normalizer;
        float wi = w[(i + r) * w_stride];
        float gi = g[i + r];
        gx[i + r] = static_cast<T>(
            normalizer * (wi * gi - factors[meanwg]) -
            xi * factors[meanwgxc] * factors[normalizer2]);
        if (has_w) {
          gw[i + r] = static_cast<T>(gi * xi);
        }
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = (x[i + r] - mean) * normalizer;
          float wi = w[(i + r) * w_stride];
          float gi = g[i + r];
          gx[i + r] = static_cast<T>(
              normalizer * (wi * gi - factors[meanwg]) -
              xi * factors[meanwgxc] * factors[normalizer2]);
          if (has_w) {
            gw[i + r] = static_cast<T>(gi * xi);
          }
        }
      }
    }
  }
}

// clang-format off
#define instantiate_layer_norm(name, itype)                                       \
  instantiate_kernel("layer_norm" #name, layer_norm_single_row, itype)            \
  instantiate_kernel("vjp_layer_norm" #name, vjp_layer_norm_single_row, itype)    \
  instantiate_kernel("layer_norm_looped" #name, layer_norm_looped, itype)         \
  instantiate_kernel("vjp_layer_norm_looped" #name, vjp_layer_norm_looped, itype)

instantiate_layer_norm(float32, float)
instantiate_layer_norm(float16, half)
instantiate_layer_norm(bfloat16, bfloat16_t) // clang-format on

// ---- embedded from Source/Cmlx/mlx-generated/metal/random.metal ----
// Copyright © 2023 Apple Inc.


static constexpr constant uint32_t rotations[2][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24}};

union rbits {
  uint2 val;
  uchar4 bytes[2];
};

rbits threefry2x32_hash(const thread uint2& key, uint2 count) {
  uint4 ks = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDA};

  rbits v;
  v.val.x = count.x + ks[0];
  v.val.y = count.y + ks[1];

  for (int i = 0; i < 5; ++i) {
    for (auto r : rotations[i % 2]) {
      v.val.x += v.val.y;
      v.val.y = (v.val.y << r) | (v.val.y >> (32 - r));
      v.val.y ^= v.val.x;
    }
    v.val.x += ks[(i + 1) % 3];
    v.val.y += ks[(i + 2) % 3] + i + 1;
  }

  return v;
}

[[kernel]] void rbitsc(
    device const uint32_t* keys,
    device char* out,
    constant const bool& odd,
    constant const uint& bytes_per_key,
    uint2 grid_dim [[threads_per_grid]],
    uint2 index [[thread_position_in_grid]]) {
  auto kidx = 2 * index.x;
  auto key = uint2(keys[kidx], keys[kidx + 1]);
  auto half_size = grid_dim.y - odd;
  out += index.x * bytes_per_key;
  bool drop_last = odd && (index.y == half_size);
  auto bits = threefry2x32_hash(
      key, uint2(index.y, drop_last ? 0 : index.y + grid_dim.y));
  size_t idx = size_t(index.y) << 2;
  for (int i = 0; i < 4; ++i) {
    out[idx + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    idx = (drop_last ? 0 : size_t(index.y) + grid_dim.y) << 2;
    if ((index.y + 1) == half_size && (bytes_per_key % 4) > 0) {
      int edge_bytes = (bytes_per_key % 4);
      for (int i = 0; i < edge_bytes; ++i) {
        out[idx + i] = bits.bytes[1][i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        out[idx + i] = bits.bytes[1][i];
      }
    }
  }
}

[[kernel]] void rbits(
    device const uint32_t* keys,
    device char* out,
    constant const bool& odd,
    constant const uint& bytes_per_key,
    constant const int& ndim,
    constant const int* key_shape,
    constant const int64_t* key_strides,
    uint2 grid_dim [[threads_per_grid]],
    uint2 index [[thread_position_in_grid]]) {
  auto kidx = 2 * index.x;
  auto k1_elem = elem_to_loc(kidx, key_shape, key_strides, ndim);
  auto k2_elem = elem_to_loc(kidx + 1, key_shape, key_strides, ndim);
  auto key = uint2(keys[k1_elem], keys[k2_elem]);
  auto half_size = grid_dim.y - odd;
  out += size_t(index.x) * bytes_per_key;
  bool drop_last = odd && (index.y == half_size);
  auto bits = threefry2x32_hash(
      key, uint2(index.y, drop_last ? 0 : index.y + grid_dim.y));
  size_t idx = size_t(index.y) << 2;
  for (int i = 0; i < 4; ++i) {
    out[idx + i] = bits.bytes[0][i];
  }
  if (!drop_last) {
    idx = (drop_last ? 0 : size_t(index.y) + grid_dim.y) << 2;
    if ((index.y + 1) == half_size && (bytes_per_key % 4) > 0) {
      int edge_bytes = (bytes_per_key % 4);
      for (int i = 0; i < edge_bytes; ++i) {
        out[idx + i] = bits.bytes[1][i];
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        out[idx + i] = bits.bytes[1][i];
      }
    }
  }
}

// ---- embedded from Source/Cmlx/mlx-generated/metal/rms_norm.metal ----
// Copyright © 2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>


using namespace metal;

constant bool has_w [[function_constant(20)]];

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_single_row(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      float xi = x[i];
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[i];
        acc += xi * xi;
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_looped(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];

  float acc = 0;
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          acc += xi * xi;
        }
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[r + i] = w[w_stride * (i + r)] *
            static_cast<T>(x[r + i] * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          out[r + i] = w[w_stride * (i + r)] *
              static_cast<T>(x[r + i] * local_inv_mean[0]);
        }
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_rms_single_row(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Allocate registers for the computation and accumulators
  float thread_x[N_READS];
  float thread_w[N_READS];
  float thread_g[N_READS];
  float sumx2 = 0;
  float sumgwx = 0;

  // Allocate shared memory to implement the reduction
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumgwx[SIMD_SIZE];
  threadgroup float local_normalizer[1];
  threadgroup float local_meangwx[1];

  // Read and accumulate locally
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      thread_w[i] = w[w_stride * i];
      thread_g[i] = g[i];

      sumx2 += thread_x[i] * thread_x[i];
      sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        thread_x[i] = x[i];
        thread_w[i] = w[w_stride * i];
        thread_g[i] = g[i];

        sumx2 += thread_x[i] * thread_x[i];
        sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
      }
    }
  }

  // Accumulate across threads
  sumx2 = simd_sum(sumx2);
  sumgwx = simd_sum(sumgwx);
  if (simd_group_id == 0) {
    local_sumx2[simd_lane_id] = 0;
    local_sumgwx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sumx2[simd_group_id] = sumx2;
    local_sumgwx[simd_group_id] = sumgwx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumgwx = simd_sum(local_sumgwx[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_meangwx[0] = sumgwx / axis_size;
      local_normalizer[0] = metal::precise::rsqrt(sumx2 / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meangwx = local_meangwx[0];
  float normalizer = local_normalizer[0];
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      gx[i] = static_cast<T>(
          thread_g[i] * thread_w[i] * normalizer -
          thread_x[i] * meangwx * normalizer3);
      if (has_w) {
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i] * normalizer);
      }
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        gx[i] = static_cast<T>(
            thread_g[i] * thread_w[i] * normalizer -
            thread_x[i] * meangwx * normalizer3);
        if (has_w) {
          gw[i] = static_cast<T>(thread_g[i] * thread_x[i] * normalizer);
        }
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_rms_looped(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Allocate registers for the accumulators
  float sumx2 = 0;
  float sumgwx = 0;

  // Allocate shared memory to implement the reduction
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumgwx[SIMD_SIZE];
  threadgroup float local_normalizer[1];
  threadgroup float local_meangwx[1];

  // Read and accumulate locally
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[w_stride * (i + r)];
        float gi = g[i + r];

        sumx2 += xi * xi;
        sumgwx += xi * wi * gi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[w_stride * (i + r)];
          float gi = g[i + r];

          sumx2 += xi * xi;
          sumgwx += xi * wi * gi;
        }
      }
    }
  }

  // Accumulate across threads
  sumx2 = simd_sum(sumx2);
  sumgwx = simd_sum(sumgwx);
  if (simd_group_id == 0) {
    local_sumx2[simd_lane_id] = 0;
    local_sumgwx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sumx2[simd_group_id] = sumx2;
    local_sumgwx[simd_group_id] = sumgwx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumgwx = simd_sum(local_sumgwx[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_meangwx[0] = sumgwx / axis_size;
      local_normalizer[0] = metal::precise::rsqrt(sumx2 / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meangwx = local_meangwx[0];
  float normalizer = local_normalizer[0];
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[w_stride * (i + r)];
        float gi = g[i + r];

        gx[i + r] =
            static_cast<T>(gi * wi * normalizer - xi * meangwx * normalizer3);
        if (has_w) {
          gw[i + r] = static_cast<T>(gi * xi * normalizer);
        }
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[w_stride * (i + r)];
          float gi = g[i + r];

          gx[i + r] =
              static_cast<T>(gi * wi * normalizer - xi * meangwx * normalizer3);
          if (has_w) {
            gw[i + r] = static_cast<T>(gi * xi * normalizer);
          }
        }
      }
    }
  }
}

// clang-format off
#define instantiate_rms(name, itype)                                \
  instantiate_kernel("rms" #name, rms_single_row, itype)            \
  instantiate_kernel("vjp_rms" #name, vjp_rms_single_row, itype)    \
  instantiate_kernel("rms_looped" #name, rms_looped, itype)         \
  instantiate_kernel("vjp_rms_looped" #name, vjp_rms_looped, itype)

instantiate_rms(float32, float)
instantiate_rms(float16, half)
instantiate_rms(bfloat16, bfloat16_t) // clang-format on

// ---- embedded from Source/Cmlx/mlx-generated/metal/rope.metal ----
// Copyright © 2023-2024 Apple Inc.

#include <metal_math>


constant bool forward [[function_constant(1)]];
constant bool traditional [[function_constant(2)]];
constant bool hs_transpose [[function_constant(3)]];

template <typename T>
void rope_single_impl(
    const device T* in,
    device T* out,
    constant const int& offset,
    const float inv_freq,
    constant const float& scale,
    constant const int64_t& stride,
    uint2 pos,
    uint2 grid) {
  float L = scale * static_cast<float>(offset);

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices
  uint index_1, index_2;
  if (traditional) {
    index_1 = 2 * pos.x + pos.y * stride;
    index_2 = index_1 + 1;
  } else {
    index_1 = pos.x + pos.y * stride;
    index_2 = index_1 + grid.x;
  }

  // Read and write the output
  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  float rx1;
  float rx2;
  if (forward) {
    rx1 = x1 * costheta - x2 * sintheta;
    rx2 = x1 * sintheta + x2 * costheta;
  } else {
    rx1 = x2 * sintheta + x1 * costheta;
    rx2 = x2 * costheta - x1 * sintheta;
  }
  out[index_1] = static_cast<T>(rx1);
  out[index_2] = static_cast<T>(rx2);
}

template <typename T>
[[kernel]] void rope_single(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const int64_t& stride,
    constant const float& base [[buffer(10)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_single_impl<T>(in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T>
[[kernel]] void rope_single_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const int64_t& stride,
    const device float* freqs [[buffer(10)]],
    constant const int64_t& freq_stride [[buffer(11)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_single_impl<T>(in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T, typename IdxT, int N = 4>
void rope_impl(
    const device T* in,
    device T* out,
    const device int* offset,
    const float inv_freq,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    uint3 pos,
    uint3 grid) {
  auto n_head_up = N * ((n_head + N - 1) / N);
  auto head_idx = static_cast<int>((pos.z * N) % n_head_up);
  auto batch_idx = (pos.z * N) / n_head_up;
  auto batch_offset = offset[batch_idx * offset_stride];
  float L = scale * static_cast<float>(pos.y + batch_offset);
  auto mat_idx = batch_idx * n_head + head_idx;

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);
  // Compute the input and output indices
  IdxT in_index_1;
  if (hs_transpose) {
    IdxT batch_stride = grid.y * IdxT(strides[1]);
    in_index_1 =
        batch_idx * batch_stride + pos.y * strides[1] + head_idx * strides[0];
  } else {
    in_index_1 = pos.y * IdxT(strides[1]) + mat_idx * IdxT(strides[0]);
  }
  IdxT in_index_2;
  IdxT out_index_1 =
      pos.y * IdxT(out_strides[1]) + mat_idx * IdxT(out_strides[0]);
  IdxT out_index_2;
  if (traditional) {
    out_index_1 += 2 * pos.x * IdxT(out_strides[2]);
    out_index_2 = out_index_1 + 1;
    in_index_1 += 2 * pos.x * IdxT(strides[2]);
    in_index_2 = in_index_1 + IdxT(strides[2]);
  } else {
    out_index_1 += pos.x * IdxT(out_strides[2]);
    out_index_2 = out_index_1 + grid.x * IdxT(out_strides[2]);
    in_index_1 += pos.x * IdxT(strides[2]);
    in_index_2 = in_index_1 + grid.x * IdxT(strides[2]);
  }
  for (int i = 0; i < N && head_idx + i < n_head; ++i) {
    // Read and write the output
    float x1 = static_cast<float>(in[in_index_1]);
    float x2 = static_cast<float>(in[in_index_2]);
    float rx1;
    float rx2;
    if (forward) {
      rx1 = x1 * costheta - x2 * sintheta;
      rx2 = x1 * sintheta + x2 * costheta;
    } else {
      rx1 = x2 * sintheta + x1 * costheta;
      rx2 = x2 * costheta - x1 * sintheta;
    }
    out[out_index_1] = static_cast<T>(rx1);
    out[out_index_2] = static_cast<T>(rx2);
    in_index_1 += IdxT(strides[0]);
    in_index_2 += IdxT(strides[0]);
    out_index_1 += IdxT(out_strides[0]);
    out_index_2 += IdxT(out_strides[0]);
  }
}

template <typename T, typename IdxT, int N = 4>
[[kernel]] void rope(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device int* offset,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    constant const float& base [[buffer(10)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_impl<T, IdxT, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      grid);
}

template <typename T, typename IdxT, int N = 4>
[[kernel]] void rope_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device int* offset,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    const device float* freqs [[buffer(10)]],
    constant const int64_t& freq_stride [[buffer(11)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_impl<T, IdxT, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      grid);
}

// clang-format off
#define instantiate_rope_g(name, type) \
  instantiate_kernel("rope_" #name, rope, type, int32_t) \
  instantiate_kernel("rope_freqs_" #name, rope_freqs, type, int32_t) \
  instantiate_kernel("rope_large_" #name, rope, type, int64_t) \
  instantiate_kernel("rope_freqs_large_" #name, rope_freqs, type, int64_t)

#define instantiate_rope_s(name, type) \
  instantiate_kernel("rope_single_" #name, rope_single, type) \
  instantiate_kernel("rope_single_freqs_" #name, rope_single_freqs, type)

#define instantiate_rope(name, type) \
  instantiate_rope_s(name, type)     \
  instantiate_rope_g(name, type)

instantiate_rope(float16, half)
instantiate_rope(bfloat16, bfloat16_t)
instantiate_rope(float32, float) // clang-format on

// ---- embedded from Source/Cmlx/mlx-generated/metal/scaled_dot_product_attention.metal ----
#include <metal_stdlib>

// clang-format off

// ---- embedded from Source/Cmlx/mlx-generated/metal/sdpa_vector.h ----
// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(has_mask)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

using namespace metal;

// SDPA vector instantiations
#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on

// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/attn/kernels/steel_attention.metal ----
// Copyright © 2024-25 Apple Inc.

// clang-format off


// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/attn/kernels/steel_attention.h ----
// Copyright © 2024-25 Apple Inc.


// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/attn/attn.h ----
// Copyright © 2024 Apple Inc.

#pragma once


// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/attn/loader.h ----
// Copyright © 2024 Apple Inc.

#pragma once


// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/defines.h ----
// Copyright © 2024 Apple Inc.

#pragma once

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

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

// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/attn/mma.h ----
// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/attn/transforms.h ----
// Copyright © 2024 Apple Inc.

#pragma once


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

// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/utils/integral_constant.h ----
// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_stdlib>

// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/utils/type_traits.h ----
// Copyright © 2024 Apple Inc.

#pragma once

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

// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/attn/params.h ----
// Copyright © 2024 Apple Inc.

#pragma once

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

// ---- embedded from Source/Cmlx/mlx-generated/metal/steel/gemm/params.h ----
// Copyright © 2024 Apple Inc.

#pragma once

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

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];

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

  const AccumType scale = params->scale * M_LOG2E_F;

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

  // Load Q blocks
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

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

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
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

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat16_t);

instantiate_attn_mask_helper(float32, float);
// clang-format on
)MLXEMB";
}

} // namespace mlx::core::metal
