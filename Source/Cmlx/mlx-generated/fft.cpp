namespace mlx::core::metal {

const char* fft() {
  return R"preamble(
// Copyright © 2025 Apple Inc.

// Auto generated source for mlx/backend/metal/kernels/fft.h

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/fft/radix.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/fft/radix.h"
// Copyright © 2024 Apple Inc.

/* Radix kernels

We provide optimized, single threaded Radix codelets
for n=2,3,4,5,6,7,8,10,11,12,13.

For n=2,3,4,5,6 we hand write the codelets.
For n=8,10,12 we combine smaller codelets.
For n=7,11,13 we use Rader's algorithm which decomposes
them into (n-1)=6,10,12 codelets. */


#include <metal_common>
#include <metal_math>
#include <metal_stdlib>

// The codelets are templated over the scalar lane type so reduced-precision
// complex paths can reuse them; complex values are plain two-lane vectors.

template <typename T>
METAL_FUNC metal::vec<T, 2> complex_mul(
    metal::vec<T, 2> a,
    metal::vec<T, 2> b) {
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

// Complex mul followed by conjugate
template <typename T>
METAL_FUNC metal::vec<T, 2> complex_mul_conj(
    metal::vec<T, 2> a,
    metal::vec<T, 2> b) {
  return {a.x * b.x - a.y * b.y, -a.x * b.y - a.y * b.x};
}

// Compute an FFT twiddle factor
template <typename T>
METAL_FUNC metal::vec<T, 2> get_twiddle(int k, int p) {
  // Derive phase evaluation precision from the scalar lane and Metal's pi
  // constant. Reduced lanes currently promote to float for fast trig, then
  // narrow the result to the lane type explicitly.
  using phase_T = decltype(M_PI_F * T(0));
  phase_T theta = -phase_T(2) * phase_T(k) * phase_T(M_PI_F) / phase_T(p);
  return {T(metal::fast::cos(theta)), T(metal::fast::sin(theta))};
}

template <typename T>
METAL_FUNC void radix2(thread metal::vec<T, 2>* x, thread metal::vec<T, 2>* y) {
  y[0] = x[0] + x[1];
  y[1] = x[0] - x[1];
}

template <typename T>
METAL_FUNC void radix3(thread metal::vec<T, 2>* x, thread metal::vec<T, 2>* y) {
  T pi_2_3 = -0.8660254037844387;

  metal::vec<T, 2> a_1 = x[1] + x[2];
  metal::vec<T, 2> a_2 = x[1] - x[2];

  y[0] = x[0] + a_1;
  metal::vec<T, 2> b_1 = x[0] - 0.5 * a_1;
  metal::vec<T, 2> b_2 = pi_2_3 * a_2;

  metal::vec<T, 2> b_2_j = {-b_2.y, b_2.x};
  y[1] = b_1 + b_2_j;
  y[2] = b_1 - b_2_j;
}

template <typename T>
METAL_FUNC void radix4(thread metal::vec<T, 2>* x, thread metal::vec<T, 2>* y) {
  metal::vec<T, 2> z_0 = x[0] + x[2];
  metal::vec<T, 2> z_1 = x[0] - x[2];
  metal::vec<T, 2> z_2 = x[1] + x[3];
  metal::vec<T, 2> z_3 = x[1] - x[3];
  metal::vec<T, 2> z_3_i = {z_3.y, -z_3.x};

  y[0] = z_0 + z_2;
  y[1] = z_1 + z_3_i;
  y[2] = z_0 - z_2;
  y[3] = z_1 - z_3_i;
}

template <typename T>
METAL_FUNC void radix5(thread metal::vec<T, 2>* x, thread metal::vec<T, 2>* y) {
  T root_5_4 = 0.5590169943749475;
  T sin_2pi_5 = 0.9510565162951535;
  T sin_1pi_5 = 0.5877852522924731;

  metal::vec<T, 2> a_1 = x[1] + x[4];
  metal::vec<T, 2> a_2 = x[2] + x[3];
  metal::vec<T, 2> a_3 = x[1] - x[4];
  metal::vec<T, 2> a_4 = x[2] - x[3];

  metal::vec<T, 2> a_5 = a_1 + a_2;
  metal::vec<T, 2> a_6 = root_5_4 * (a_1 - a_2);
  metal::vec<T, 2> a_7 = x[0] - a_5 / 4;
  metal::vec<T, 2> a_8 = a_7 + a_6;
  metal::vec<T, 2> a_9 = a_7 - a_6;
  metal::vec<T, 2> a_10 = sin_2pi_5 * a_3 + sin_1pi_5 * a_4;
  metal::vec<T, 2> a_11 = sin_1pi_5 * a_3 - sin_2pi_5 * a_4;
  metal::vec<T, 2> a_10_j = {a_10.y, -a_10.x};
  metal::vec<T, 2> a_11_j = {a_11.y, -a_11.x};

  y[0] = x[0] + a_5;
  y[1] = a_8 + a_10_j;
  y[2] = a_9 + a_11_j;
  y[3] = a_9 - a_11_j;
  y[4] = a_8 - a_10_j;
}

template <typename T>
METAL_FUNC void radix6(thread metal::vec<T, 2>* x, thread metal::vec<T, 2>* y) {
  T sin_pi_3 = 0.8660254037844387;
  metal::vec<T, 2> a_1 = x[2] + x[4];
  metal::vec<T, 2> a_2 = x[0] - a_1 / 2;
  metal::vec<T, 2> a_3 = sin_pi_3 * (x[2] - x[4]);
  metal::vec<T, 2> a_4 = x[5] + x[1];
  metal::vec<T, 2> a_5 = x[3] - a_4 / 2;
  metal::vec<T, 2> a_6 = sin_pi_3 * (x[5] - x[1]);
  metal::vec<T, 2> a_7 = x[0] + a_1;

  metal::vec<T, 2> a_3_i = {a_3.y, -a_3.x};
  metal::vec<T, 2> a_6_i = {a_6.y, -a_6.x};
  metal::vec<T, 2> a_8 = a_2 + a_3_i;
  metal::vec<T, 2> a_9 = a_2 - a_3_i;
  metal::vec<T, 2> a_10 = x[3] + a_4;
  metal::vec<T, 2> a_11 = a_5 + a_6_i;
  metal::vec<T, 2> a_12 = a_5 - a_6_i;

  y[0] = a_7 + a_10;
  y[1] = a_8 - a_11;
  y[2] = a_9 + a_12;
  y[3] = a_7 - a_10;
  y[4] = a_8 + a_11;
  y[5] = a_9 - a_12;
}

template <typename T>
METAL_FUNC void radix7(thread metal::vec<T, 2>* x, thread metal::vec<T, 2>* y) {
  // Rader's algorithm
  T inv_r = static_cast<T>(1 / 6.0);
  metal::vec<T, 2> inv = {inv_r, -inv_r};

  // fft
  metal::vec<T, 2> in1[6] = {x[1], x[3], x[2], x[6], x[4], x[5]};
  radix6<T>(in1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj<T>(y[1], metal::vec<T, 2>(-1, 0));
  y[2] = complex_mul_conj<T>(y[2], metal::vec<T, 2>(2.44013336, -1.02261879));
  y[3] = complex_mul_conj<T>(y[3], metal::vec<T, 2>(2.37046941, -1.17510629));
  y[4] = complex_mul_conj<T>(y[4], metal::vec<T, 2>(0, -2.64575131));
  y[5] = complex_mul_conj<T>(y[5], metal::vec<T, 2>(2.37046941, 1.17510629));
  y[6] = complex_mul_conj<T>(y[6], metal::vec<T, 2>(-2.44013336, -1.02261879));

  // ifft
  radix6<T>(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[5] = x[2] * inv + x[0];
  y[4] = x[3] * inv + x[0];
  y[6] = x[4] * inv + x[0];
  y[2] = x[5] * inv + x[0];
  y[3] = x[6] * inv + x[0];
}

template <typename T>
METAL_FUNC void radix8(thread metal::vec<T, 2>* x, thread metal::vec<T, 2>* y) {
  T cos_pi_4 = 0.7071067811865476;
  metal::vec<T, 2> w_0 = {cos_pi_4, -cos_pi_4};
  metal::vec<T, 2> w_1 = {-cos_pi_4, -cos_pi_4};
  metal::vec<T, 2> temp[8] = {x[0], x[2], x[4], x[6], x[1], x[3], x[5], x[7]};
  radix4<T>(temp, x);
  radix4<T>(temp + 4, x + 4);

  y[0] = x[0] + x[4];
  y[4] = x[0] - x[4];
  metal::vec<T, 2> x_5 = complex_mul<T>(x[5], w_0);
  y[1] = x[1] + x_5;
  y[5] = x[1] - x_5;
  metal::vec<T, 2> x_6 = {x[6].y, -x[6].x};
  y[2] = x[2] + x_6;
  y[6] = x[2] - x_6;
  metal::vec<T, 2> x_7 = complex_mul<T>(x[7], w_1);
  y[3] = x[3] + x_7;
  y[7] = x[3] - x_7;
}

template <typename T, bool raders_perm>
METAL_FUNC void radix10(
    thread metal::vec<T, 2>* x,
    thread metal::vec<T, 2>* y) {
  metal::vec<T, 2> w[4];
  w[0] = {T(0.8090169943749475), T(-0.5877852522924731)};
  w[1] = {T(0.30901699437494745), T(-0.9510565162951535)};
  w[2] = {-w[1].x, w[1].y};
  w[3] = {-w[0].x, w[0].y};

  if (raders_perm) {
    metal::vec<T, 2> temp[10] = {
        x[0], x[3], x[4], x[8], x[2], x[1], x[7], x[9], x[6], x[5]};
    radix5<T>(temp, x);
    radix5<T>(temp + 5, x + 5);
  } else {
    metal::vec<T, 2> temp[10] = {
        x[0], x[2], x[4], x[6], x[8], x[1], x[3], x[5], x[7], x[9]};
    radix5<T>(temp, x);
    radix5<T>(temp + 5, x + 5);
  }

  y[0] = x[0] + x[5];
  y[5] = x[0] - x[5];
  for (int t = 1; t < 5; t++) {
    metal::vec<T, 2> a = complex_mul<T>(x[t + 5], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 5] = x[t] - a;
  }
}

template <typename T>
METAL_FUNC void radix11(
    thread metal::vec<T, 2>* x,
    thread metal::vec<T, 2>* y) {
  // Rader's algorithm
  T inv_r = static_cast<T>(1 / 10.0);
  metal::vec<T, 2> inv = {inv_r, -inv_r};

  // fft
  radix10<T, true>(x + 1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj<T>(y[1], metal::vec<T, 2>(-1, 0));
  y[2] = complex_mul_conj<T>(y[2], metal::vec<T, 2>(0.955301878, -3.17606649));
  y[3] = complex_mul_conj<T>(y[3], metal::vec<T, 2>(2.63610556, 2.01269656));
  y[4] = complex_mul_conj<T>(y[4], metal::vec<T, 2>(2.54127802, 2.13117479));
  y[5] = complex_mul_conj<T>(y[5], metal::vec<T, 2>(2.07016210, 2.59122150));
  y[6] = complex_mul_conj<T>(y[6], metal::vec<T, 2>(0, -3.31662479));
  y[7] = complex_mul_conj<T>(y[7], metal::vec<T, 2>(2.07016210, -2.59122150));
  y[8] = complex_mul_conj<T>(y[8], metal::vec<T, 2>(-2.54127802, 2.13117479));
  y[9] = complex_mul_conj<T>(y[9], metal::vec<T, 2>(2.63610556, -2.01269656));
  y[10] =
      complex_mul_conj<T>(y[10], metal::vec<T, 2>(-0.955301878, -3.17606649));

  // ifft
  radix10<T, false>(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[6] = x[2] * inv + x[0];
  y[3] = x[3] * inv + x[0];
  y[7] = x[4] * inv + x[0];
  y[9] = x[5] * inv + x[0];
  y[10] = x[6] * inv + x[0];
  y[5] = x[7] * inv + x[0];
  y[8] = x[8] * inv + x[0];
  y[4] = x[9] * inv + x[0];
  y[2] = x[10] * inv + x[0];
}

template <typename T, bool raders_perm>
METAL_FUNC void radix12(
    thread metal::vec<T, 2>* x,
    thread metal::vec<T, 2>* y) {
  metal::vec<T, 2> w[6];
  T sin_pi_3 = 0.8660254037844387;
  w[0] = {sin_pi_3, -0.5};
  w[1] = {0.5, -sin_pi_3};
  w[2] = {0, -1};
  w[3] = {-0.5, -sin_pi_3};
  w[4] = {-sin_pi_3, -0.5};

  if (raders_perm) {
    metal::vec<T, 2> temp[12] = {
        x[0],
        x[3],
        x[2],
        x[11],
        x[8],
        x[9],
        x[1],
        x[7],
        x[5],
        x[10],
        x[4],
        x[6]};
    radix6<T>(temp, x);
    radix6<T>(temp + 6, x + 6);
  } else {
    metal::vec<T, 2> temp[12] = {
        x[0],
        x[2],
        x[4],
        x[6],
        x[8],
        x[10],
        x[1],
        x[3],
        x[5],
        x[7],
        x[9],
        x[11]};
    radix6<T>(temp, x);
    radix6<T>(temp + 6, x + 6);
  }

  y[0] = x[0] + x[6];
  y[6] = x[0] - x[6];
  for (int t = 1; t < 6; t++) {
    metal::vec<T, 2> a = complex_mul<T>(x[t + 6], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 6] = x[t] - a;
  }
}

template <typename T>
METAL_FUNC void radix13(
    thread metal::vec<T, 2>* x,
    thread metal::vec<T, 2>* y) {
  // Rader's algorithm
  T inv_r = static_cast<T>(1 / 12.0);
  metal::vec<T, 2> inv = {inv_r, -inv_r};

  // fft
  radix12<T, true>(x + 1, y + 1);

  y[0] = y[1] + x[0];

  // b_q
  y[1] = complex_mul_conj<T>(y[1], metal::vec<T, 2>(-1, 0));
  y[2] = complex_mul_conj<T>(y[2], metal::vec<T, 2>(3.07497206, -1.88269669));
  y[3] = complex_mul_conj<T>(y[3], metal::vec<T, 2>(3.09912468, 1.84266823));
  y[4] = complex_mul_conj<T>(y[4], metal::vec<T, 2>(3.45084438, -1.04483161));
  y[5] = complex_mul_conj<T>(y[5], metal::vec<T, 2>(0.91083583, 3.48860690));
  y[6] = complex_mul_conj<T>(y[6], metal::vec<T, 2>(-3.60286363, 0.139189267));
  y[7] = complex_mul_conj<T>(y[7], metal::vec<T, 2>(3.60555128, 0));
  y[8] = complex_mul_conj<T>(y[8], metal::vec<T, 2>(3.60286363, 0.139189267));
  y[9] = complex_mul_conj<T>(y[9], metal::vec<T, 2>(0.91083583, -3.48860690));
  y[10] =
      complex_mul_conj<T>(y[10], metal::vec<T, 2>(-3.45084438, -1.04483161));
  y[11] = complex_mul_conj<T>(y[11], metal::vec<T, 2>(3.09912468, -1.84266823));
  y[12] =
      complex_mul_conj<T>(y[12], metal::vec<T, 2>(-3.07497206, -1.88269669));

  // ifft
  radix12<T, false>(y + 1, x + 1);

  y[1] = x[1] * inv + x[0];
  y[7] = x[2] * inv + x[0];
  y[10] = x[3] * inv + x[0];
  y[5] = x[4] * inv + x[0];
  y[9] = x[5] * inv + x[0];
  y[11] = x[6] * inv + x[0];
  y[12] = x[7] * inv + x[0];
  y[6] = x[8] * inv + x[0];
  y[3] = x[9] * inv + x[0];
  y[8] = x[10] * inv + x[0];
  y[4] = x[11] * inv + x[0];
  y[2] = x[12] * inv + x[0];
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/bf16.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/bf16.h"
// Copyright © 2023 Apple Inc.


#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/complex.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/complex.h"
// Copyright © 2023 Apple Inc.


#include <metal_stdlib>


using namespace metal;

template <typename T>
struct complex_t;

template <typename T>
static constexpr constant bool is_complex_v = false;

template <typename T>
static constexpr constant bool is_complex_v<complex_t<T>> = true;

// Metal accepts explicit bfloat casts that is_convertible_v reports as false.
template <typename From, typename To>
static constexpr constant bool is_lane_convertible_v =
    is_convertible_v<From, To> ||
    (is_same_v<To, bfloat16_t> && is_convertible_v<From, float>) ||
    (is_same_v<From, bfloat16_t> && is_convertible_v<float, To>);

template <typename T>
struct complex_t {
  using value_type = T;

  T real;
  T imag;

  // Constructors
  constexpr complex_t(T real, T imag) thread : real(real), imag(imag) {};
  constexpr complex_t() thread : real(0), imag(0) {};
  constexpr complex_t() threadgroup : real(0), imag(0) {};

  // Conversions from scalar types
  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) thread : real(static_cast<T>(x)),
                                    imag(static_cast<T>(0)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) threadgroup : real(static_cast<T>(x)),
                                         imag(static_cast<T>(0)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) device : real(static_cast<T>(x)),
                                    imag(static_cast<T>(0)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) constant : real(static_cast<T>(x)),
                                      imag(static_cast<T>(0)) {}

  // Conversions between complex types
  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) thread : real(static_cast<T>(x.real)),
                                               imag(static_cast<T>(x.imag)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) threadgroup
      : real(static_cast<T>(x.real)),
        imag(static_cast<T>(x.imag)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) device : real(static_cast<T>(x.real)),
                                               imag(static_cast<T>(x.imag)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) constant : real(static_cast<T>(x.real)),
                                                 imag(static_cast<T>(x.imag)) {}

  // Conversions to and from two-lane vectors (the FFT lane representation)
  constexpr complex_t(vec<T, 2> v) thread : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) threadgroup : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) device : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) constant : real(v.x), imag(v.y) {};

  constexpr operator vec<T, 2>() const thread {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const threadgroup {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const device {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const constant {
    return vec<T, 2>(real, imag);
  }

  // Conversions to scalar types
  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const thread {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const threadgroup {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const device {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const constant {
    return static_cast<U>(real);
  }
};

using complex32_t = complex_t<half>;
using complex64_t = complex_t<float>;

static_assert(sizeof(complex32_t) == 2 * sizeof(half));
static_assert(sizeof(complex64_t) == 2 * sizeof(float));
static_assert(sizeof(complex_t<bfloat16_t>) == 2 * sizeof(bfloat16_t));

template <typename T>
constexpr complex_t<T> operator-(complex_t<T> x) {
  return {-x.real, -x.imag};
}

template <typename T>
constexpr bool operator>=(complex_t<T> a, complex_t<T> b) {
  return (a.real > b.real) || (a.real == b.real && a.imag >= b.imag);
}

template <typename T>
constexpr bool operator>(complex_t<T> a, complex_t<T> b) {
  return (a.real > b.real) || (a.real == b.real && a.imag > b.imag);
}

template <typename T>
constexpr bool operator<=(complex_t<T> a, complex_t<T> b) {
  return operator>=(b, a);
}

template <typename T>
constexpr bool operator<(complex_t<T> a, complex_t<T> b) {
  return operator>(b, a);
}

template <typename T>
constexpr bool operator==(complex_t<T> a, complex_t<T> b) {
  return a.real == b.real && a.imag == b.imag;
}

template <typename T>
constexpr complex_t<T> operator+(complex_t<T> a, complex_t<T> b) {
  return {a.real + b.real, a.imag + b.imag};
}

template <typename T>
constexpr thread complex_t<T>& operator+=(
    thread complex_t<T>& a,
    complex_t<T> b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

template <typename T>
constexpr threadgroup complex_t<T>& operator+=(
    threadgroup complex_t<T>& a,
    complex_t<T> b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

template <typename T>
constexpr device complex_t<T>& operator+=(
    device complex_t<T>& a,
    complex_t<T> b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator+(U a, complex_t<T> b) {
  return {static_cast<T>(a) + b.real, b.imag};
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator+(complex_t<T> a, U b) {
  return {a.real + static_cast<T>(b), a.imag};
}

template <typename T>
constexpr complex_t<T> operator-(complex_t<T> a, complex_t<T> b) {
  return {a.real - b.real, a.imag - b.imag};
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator-(U a, complex_t<T> b) {
  return {static_cast<T>(a) - b.real, -b.imag};
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator-(complex_t<T> a, U b) {
  return {a.real - static_cast<T>(b), a.imag};
}

template <typename T>
constexpr complex_t<T> operator*(complex_t<T> a, complex_t<T> b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

template <typename T>
constexpr complex_t<T> operator/(complex_t<T> a, complex_t<T> b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a.real * b.real + a.imag * b.imag;
  auto y = a.imag * b.real - a.real * b.imag;
  return {x / denom, y / denom};
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator/(U a, complex_t<T> b) {
  auto scalar = static_cast<T>(a);
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = scalar * b.real;
  auto y = -scalar * b.imag;
  return {x / denom, y / denom};
}

template <typename T>
constexpr complex_t<T> operator%(complex_t<T> a, complex_t<T> b) {
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

static_assert(
    (complex_t<half>{1.0h, 2.0h} * complex_t<half>{3.0h, 4.0h}).real == -5.0h);
static_assert(
    (complex_t<bfloat16_t>{bfloat16_t(1.0f), bfloat16_t(2.0f)} *
     complex_t<bfloat16_t>{bfloat16_t(3.0f), bfloat16_t(4.0f)})
        .real == bfloat16_t(-5.0f));

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/fft/readwrite.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/fft/readwrite.h"
// Copyright © 2024 Apple Inc.

#include <metal_common>


/* FFT helpers for reading and writing from/to device memory.

For many sizes, GPU FFTs are memory bandwidth bound so
read/write performance is important.

Where possible, we read 128 bits sequentially in each thread,
coalesced with accesses from adjacent threads for optimal performance.

We implement specialized reading/writing for:
  - FFT
  - RFFT
  - IRFFT

Each with support for:
  - Contiguous reads
  - Padded reads
  - Strided reads
*/

#define MAX_RADIX 13

using namespace metal;

// Derives the FFT scalar arithmetic type from a storage type.
template <typename storage_T>
struct FFTStorageTraits {
  using scalar_T = storage_T;
};

template <typename T>
struct FFTStorageTraits<vec<T, 2>> {
  using scalar_T = T;
};

template <typename T>
struct FFTStorageTraits<complex_t<T>> {
  using scalar_T = T;
};

template <typename in_T, typename out_T>
struct FFTIOTypeTraits {
  using scalar_T = typename FFTStorageTraits<in_T>::scalar_T;

  static_assert(
      metal::is_same_v<scalar_T, typename FFTStorageTraits<out_T>::scalar_T>,
      "FFT input and output storage must share a scalar arithmetic type");
  static_assert(
      sizeof(in_T) <= 16 && 16 % sizeof(in_T) == 0,
      "FFT input storage must divide the 128-bit sequential access width");
  static_assert(
      sizeof(out_T) <= 16 && 16 % sizeof(out_T) == 0,
      "FFT output storage must divide the 128-bit sequential access width");
};

template <
    typename in_T,
    typename out_T,
    int step = 0,
    bool four_step_real = false>
struct ReadWriter {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  using complex_T = vec<scalar_T, 2>;

  const device in_T* in;
  threadgroup complex_T* buf;
  device out_T* out;
  int n;
  int batch_size;
  int elems_per_thread;
  uint3 elem;
  uint3 grid;
  int threads_per_tg;
  bool inv;

  // Used for strided access
  int strided_device_idx = 0;
  int strided_shared_idx = 0;

  METAL_FUNC ReadWriter(
      const device in_T* in_,
      threadgroup complex_T* buf_,
      device out_T* out_,
      const short n_,
      const int batch_size_,
      const short elems_per_thread_,
      const uint3 elem_,
      const uint3 grid_,
      const bool inv_) thread : in(in_),
                                buf(buf_),
                                out(out_),
                                n(n_),
                                batch_size(batch_size_),
                                elems_per_thread(elems_per_thread_),
                                elem(elem_),
                                grid(grid_),
                                inv(inv_) {
    // Account for padding on last threadgroup
    threads_per_tg = elem.x == grid.x - 1
        ? (batch_size - (grid.x - 1) * grid.y) * grid.z
        : grid.y * grid.z;
  }

  // ifft(x) = 1/n * conj(fft(conj(x)))
  METAL_FUNC complex_T post_in(complex_T elem) const thread {
    return inv ? complex_T(elem.x, -elem.y) : elem;
  }

  // Handle packed complex_t storage
  METAL_FUNC complex_T post_in(complex_t<scalar_T> elem) const thread {
    return post_in(complex_T(elem.real, elem.imag));
  }

  // Handle float case for generic RFFT alg
  METAL_FUNC complex_T post_in(scalar_T elem) const thread {
    return complex_T(elem, 0);
  }

  METAL_FUNC complex_T pre_out(complex_T elem) const thread {
    return pre_out(elem, n);
  }

  METAL_FUNC complex_T pre_out(complex_T elem, int length) const thread {
    if (!inv) {
      return elem;
    }
    if constexpr (metal::is_same_v<scalar_T, float>) {
      return complex_T(elem.x / length, -elem.y / length);
    } else {
      // Compute the reciprocal in float before narrowing to the lane.
      scalar_T inv_length = static_cast<scalar_T>(1.0f / length);
      return complex_T(elem.x * inv_length, -elem.y * inv_length);
    }
  }

  METAL_FUNC bool out_of_bounds() const thread {
    // Account for possible extra threadgroups
    int grid_index = elem.x * grid.y + elem.y;
    return grid_index >= batch_size;
  }

  METAL_FUNC void load(scalar_T scale) const thread {
    size_t batch_idx = size_t(elem.x * grid.y) * n;
    short tg_idx = elem.y * grid.z + elem.z;
    // Keep each thread's sequential access at 128 bits where possible.
    constexpr int read_width = 16 / sizeof(in_T);
    short max_full_index = grid.y * n - read_width;
    short full_width_reads = elems_per_thread / read_width;
    for (short e = 0; e < full_width_reads; e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      index = metal::min(index, max_full_index);
      // vectorized reads
      for (short r = 0; r < read_width; r++) {
        buf[index + r] = post_in(in[batch_idx + index + r]) * scale;
      }
    }
    short max_index = grid.y * n - 1;
    for (short r = 0; r < elems_per_thread % read_width; r++) {
      short index = tg_idx + r * threads_per_tg +
          read_width * threads_per_tg * full_width_reads;
      index = metal::min(index, max_index);
      buf[index] = post_in(in[batch_idx + index]) * scale;
    }
  }

  METAL_FUNC void load() const thread {
    load(static_cast<scalar_T>(1));
  }

  METAL_FUNC void write() const thread {
    size_t batch_idx = size_t(elem.x * grid.y) * n;
    short tg_idx = elem.y * grid.z + elem.z;
    constexpr int read_width = 16 / sizeof(out_T);
    short max_full_index = grid.y * n - read_width;
    short full_width_reads = elems_per_thread / read_width;
    for (short e = 0; e < full_width_reads; e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      index = metal::min(index, max_full_index);
      // vectorized reads
      for (short r = 0; r < read_width; r++) {
        out[batch_idx + index + r] = pre_out(buf[index + r]);
      }
    }
    short max_index = grid.y * n - 1;
    for (short r = 0; r < elems_per_thread % read_width; r++) {
      short index = tg_idx + r * threads_per_tg +
          read_width * threads_per_tg * full_width_reads;
      index = metal::min(index, max_index);
      out[batch_idx + index] = pre_out(buf[index]);
    }
  }

  // Padded IO for Bluestein's algorithm
  METAL_FUNC void load_padded(int length, const device complex_T* w_k)
      const thread {
    size_t batch_idx = size_t(elem.x * grid.y) * length + elem.y * length;
    int fft_idx = elem.z;
    int m = grid.z;

    threadgroup complex_T* seq_buf = buf + elem.y * n;
    for (int e = 0; e < elems_per_thread; e++) {
      int index = metal::min(fft_idx + e * m, n - 1);
      if (index < length) {
        complex_T elem = post_in(in[batch_idx + index]);
        seq_buf[index] = complex_mul<scalar_T>(elem, w_k[index]);
      } else {
        seq_buf[index] = 0.0;
      }
    }
  }

  METAL_FUNC void write_padded(int length, const device complex_T* w_k)
      const thread {
    size_t batch_idx = size_t(elem.x * grid.y) * length + elem.y * length;
    int fft_idx = elem.z;
    int m = grid.z;
    scalar_T inv_n = static_cast<scalar_T>(1.0f / n);
    complex_T inv_factor = {inv_n, -inv_n};
    if constexpr (!metal::is_same_v<scalar_T, float>) {
      // Reduced lanes applied the 1/n scale with the convolution, so only
      // the conjugation remains here.
      inv_factor = {scalar_T(1), -scalar_T(1)};
    }

    threadgroup complex_T* seq_buf = buf + elem.y * n;
    for (int e = 0; e < elems_per_thread; e++) {
      int index = metal::min(fft_idx + e * m, n - 1);
      if (index < length) {
        complex_T elem = seq_buf[index + length - 1] * inv_factor;
        out[batch_idx + index] =
            pre_out(complex_mul<scalar_T>(elem, w_k[index]), length);
      }
    }
  }

  // Strided IO for four step FFT
  METAL_FUNC void compute_strided_indices(int stride, int overall_n) thread {
    // Use the batch threadgroup dimension to coalesce memory accesses:
    // e.g. stride = 12
    // device      | shared mem
    // 0  1  2  3  |  0 12 - -
    // -  -  -  -  |  1 13 - -
    // -  -  -  -  |  2 14 - -
    // 12 13 14 15 |  3 15 - -
    int coalesce_width = grid.y;
    int tg_idx = elem.y * grid.z + elem.z;
    int outer_batch_size = stride / coalesce_width;

    int strided_batch_idx = (elem.x % outer_batch_size) * coalesce_width +
        overall_n * (elem.x / outer_batch_size);
    strided_device_idx = strided_batch_idx +
        tg_idx / coalesce_width * elems_per_thread * stride +
        tg_idx % coalesce_width;
    strided_shared_idx = (tg_idx % coalesce_width) * n +
        tg_idx / coalesce_width * elems_per_thread;
  }

  // Four-step FFT I/O
  METAL_FUNC void load_strided(int stride, int overall_n) thread {
    if constexpr (step == 1 && !four_step_real) {
      // Do not invert between C2C four-step passes.
      (void)stride;
      (void)overall_n;
      bool default_inv = inv;
      inv = false;
      if constexpr (!metal::is_same_v<scalar_T, float>) {
        // Reduced lanes normalize inverse transforms on input so
        // intermediates stay in range; see write_strided.
        if (default_inv) {
          load(static_cast<scalar_T>(1.0f / n));
        } else {
          load();
        }
      } else {
        load();
      }
      inv = default_inv;
    } else {
      compute_strided_indices(stride, overall_n);
      for (int e = 0; e < elems_per_thread; e++) {
        complex_T input = post_in(in[strided_device_idx + e * stride]);
        if constexpr (!metal::is_same_v<scalar_T, float>) {
          if (step == 0 && !four_step_real && inv) {
            input *= static_cast<scalar_T>(1.0f / n);
          }
        }
        buf[strided_shared_idx + e] = input;
      }
    }
  }

  METAL_FUNC void write_strided(int stride, int overall_n) thread {
    if constexpr (step == 1 && !four_step_real) {
      compute_strided_indices(stride, overall_n);
      for (int e = 0; e < elems_per_thread; e++) {
        if constexpr (!metal::is_same_v<scalar_T, float>) {
          // Inverse C2C passes were normalized on input, so only the
          // conjugation of the inverse transform remains here.
          complex_T value = buf[strided_shared_idx + e];
          out[strided_device_idx + e * stride] =
              inv ? complex_T(value.x, -value.y) : value;
        } else {
          out[strided_device_idx + e * stride] =
              pre_out(buf[strided_shared_idx + e], overall_n);
        }
      }
    } else {
      for (int e = 0; e < elems_per_thread; e++) {
        complex_T output = buf[strided_shared_idx + e];
        int combined_idx = (strided_device_idx + e * stride) % overall_n;
        int ij = (combined_idx / stride) * (combined_idx % stride);
        // Apply four step twiddles at end of first step
        complex_T twiddle = get_twiddle<scalar_T>(ij, overall_n);
        out[strided_device_idx + e * stride] =
            complex_mul<scalar_T>(output, twiddle);
      }
    }
  }
};

// Packed RFFT/IRFFT remains float-specific. This generic foundation covers
// C2C storage; reduced-precision real transforms need separate host dispatch.
//
// For RFFT, we interleave batches of two real sequences into one complex one:
//
// z_k = x_k + j.y_k
// X_k = (Z_k + Z_(N-k)*) / 2
// Y_k = -j * ((Z_k - Z_(N-k)*) / 2)
//
// This roughly doubles the throughput over the regular FFT.
template <>
METAL_FUNC bool ReadWriter<float, float2>::out_of_bounds() const thread {
  int grid_index = elem.x * grid.y + elem.y;
  // We pack two sequences into one for RFFTs
  return grid_index * 2 >= batch_size;
}

template <>
METAL_FUNC void ReadWriter<float, float2>::load() const thread {
  size_t batch_idx = size_t(elem.x * grid.y) * n * 2 + elem.y * n * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n;

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    seq_buf[index].x = in[batch_idx + index];
    seq_buf[index].y = in[batch_idx + index + next_in];
  }
}

template <>
METAL_FUNC void ReadWriter<float, float2>::write() const thread {
  short n_over_2 = (n / 2) + 1;

  size_t batch_idx =
      size_t(elem.x * grid.y) * n_over_2 * 2 + elem.y * n_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n_over_2;

  float2 conj = {1, -1};
  float2 minus_j = {0, -1};

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread / 2 + 1; e++) {
    int index = metal::min(fft_idx + e * m, n_over_2 - 1);
    // x_0 = z_0.real
    // y_0 = z_0.imag
    if (index == 0) {
      out[batch_idx + index] = {seq_buf[index].x, 0};
      out[batch_idx + index + next_out] = {seq_buf[index].y, 0};
    } else {
      float2 x_k = seq_buf[index];
      float2 x_n_minus_k = seq_buf[n - index] * conj;
      out[batch_idx + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx + index + next_out] =
          complex_mul<float>(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
}

template <>
METAL_FUNC void ReadWriter<float, float2>::load_padded(
    int length,
    const device float2* w_k) const thread {
  size_t batch_idx = size_t(elem.x * grid.y) * length * 2 + elem.y * length * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : length;

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    if (index < length) {
      float2 elem =
          float2(in[batch_idx + index], in[batch_idx + index + next_in]);
      seq_buf[index] = complex_mul<float>(elem, w_k[index]);
    } else {
      seq_buf[index] = 0;
    }
  }
}

template <>
METAL_FUNC void ReadWriter<float, float2>::write_padded(
    int length,
    const device float2* w_k) const thread {
  int length_over_2 = (length / 2) + 1;
  size_t batch_idx =
      size_t(elem.x * grid.y) * length_over_2 * 2 + elem.y * length_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n + length - 1;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out = batch_size % 2 == 1 && grid_index * 2 == batch_size - 1
      ? 0
      : length_over_2;

  float2 conj = {1, -1};
  float2 inv_factor = {1.0f / n, -1.0f / n};
  float2 minus_j = {0, -1};

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread / 2 + 1; e++) {
    int index = metal::min(fft_idx + e * m, length_over_2 - 1);
    // x_0 = z_0.real
    // y_0 = z_0.imag
    if (index == 0) {
      float2 elem = complex_mul<float>(w_k[index], seq_buf[index] * inv_factor);
      out[batch_idx + index] = float2(elem.x, 0);
      out[batch_idx + index + next_out] = float2(elem.y, 0);
    } else {
      float2 x_k = complex_mul<float>(w_k[index], seq_buf[index] * inv_factor);
      float2 x_n_minus_k = complex_mul<float>(
          w_k[length - index], seq_buf[length - index] * inv_factor);
      x_n_minus_k *= conj;
      // w_k should happen before this extraction
      out[batch_idx + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx + index + next_out] =
          complex_mul<float>(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
}

// For IRFFT, we do the opposite
//
// Z_k = X_k + j.Y_k
// x_k = Re(Z_k)
// Y_k = Imag(Z_k)
template <>
METAL_FUNC bool ReadWriter<float2, float>::out_of_bounds() const thread {
  int grid_index = elem.x * grid.y + elem.y;
  // We pack two sequences into one for IRFFTs
  return grid_index * 2 >= batch_size;
}

template <>
METAL_FUNC void ReadWriter<float2, float>::load() const thread {
  short n_over_2 = (n / 2) + 1;
  size_t batch_idx =
      size_t(elem.x * grid.y) * n_over_2 * 2 + elem.y * n_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n_over_2;

  short m = grid.z;
  short fft_idx = elem.z;

  float2 conj = {1, -1};
  float2 plus_j = {0, 1};

  for (int t = 0; t < elems_per_thread / 2 + 1; t++) {
    int index = metal::min(fft_idx + t * m, n_over_2 - 1);
    float2 x = in[batch_idx + index];
    float2 y = in[batch_idx + index + next_in];
    // NumPy forces first input to be real
    bool first_val = index == 0;
    // NumPy forces last input on even irffts to be real
    bool last_val = n % 2 == 0 && index == n_over_2 - 1;
    if (first_val || last_val) {
      x = float2(x.x, 0);
      y = float2(y.x, 0);
    }
    seq_buf[index] = x + complex_mul<float>(y, plus_j);
    seq_buf[index].y = -seq_buf[index].y;
    if (index > 0 && !last_val) {
      seq_buf[n - index] = (x * conj) + complex_mul<float>(y * conj, plus_j);
      seq_buf[n - index].y = -seq_buf[n - index].y;
    }
  }
}

template <>
METAL_FUNC void ReadWriter<float2, float>::write() const thread {
  int batch_idx = elem.x * grid.y * n * 2 + elem.y * n * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : n;

  short m = grid.z;
  short fft_idx = elem.z;

  for (int e = 0; e < elems_per_thread; e++) {
    int index = metal::min(fft_idx + e * m, n - 1);
    out[batch_idx + index] = seq_buf[index].x / n;
    out[batch_idx + index + next_out] = seq_buf[index].y / -n;
  }
}

template <>
METAL_FUNC void ReadWriter<float2, float>::load_padded(
    int length,
    const device float2* w_k) const thread {
  int n_over_2 = (n / 2) + 1;
  int length_over_2 = (length / 2) + 1;

  size_t batch_idx =
      size_t(elem.x * grid.y) * length_over_2 * 2 + elem.y * length_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;

  // No out of bounds accesses on odd batch sizes
  int grid_index = elem.x * grid.y + elem.y;
  short next_in = batch_size % 2 == 1 && grid_index * 2 == batch_size - 1
      ? 0
      : length_over_2;

  short m = grid.z;
  short fft_idx = elem.z;

  float2 conj = {1, -1};
  float2 plus_j = {0, 1};

  for (int t = 0; t < elems_per_thread / 2 + 1; t++) {
    int index = metal::min(fft_idx + t * m, n_over_2 - 1);
    float2 x = in[batch_idx + index];
    float2 y = in[batch_idx + index + next_in];
    if (index < length_over_2) {
      bool last_val = length % 2 == 0 && index == length_over_2 - 1;
      if (last_val) {
        x = float2(x.x, 0);
        y = float2(y.x, 0);
      }
      float2 elem1 = x + complex_mul<float>(y, plus_j);
      seq_buf[index] = complex_mul<float>(elem1 * conj, w_k[index]);
      if (index > 0 && !last_val) {
        float2 elem2 = (x * conj) + complex_mul<float>(y * conj, plus_j);
        seq_buf[length - index] =
            complex_mul<float>(elem2 * conj, w_k[length - index]);
      }
    } else {
      short pad_index = metal::min(length + (index - length_over_2) * 2, n - 2);
      seq_buf[pad_index] = 0;
      seq_buf[pad_index + 1] = 0;
    }
  }
}

template <>
METAL_FUNC void ReadWriter<float2, float>::write_padded(
    int length,
    const device float2* w_k) const thread {
  size_t batch_idx = size_t(elem.x * grid.y) * length * 2 + elem.y * length * 2;
  threadgroup float2* seq_buf = buf + elem.y * n + length - 1;

  int grid_index = elem.x * grid.y + elem.y;
  short next_out =
      batch_size % 2 == 1 && grid_index * 2 == batch_size - 1 ? 0 : length;

  short m = grid.z;
  short fft_idx = elem.z;

  float2 inv_factor = {1.0f / n, -1.0f / n};
  for (int e = 0; e < elems_per_thread; e++) {
    int index = fft_idx + e * m;
    if (index < length) {
      float2 output =
          complex_mul<float>(seq_buf[index] * inv_factor, w_k[index]);
      out[batch_idx + index] = output.x / length;
      out[batch_idx + index + next_out] = output.y / -length;
    }
  }
}

// Four Step RFFT
template <>
METAL_FUNC void
ReadWriter<float2, float2, /*step=*/1, /*real=*/true>::load_strided(
    int stride,
    int overall_n) thread {
  // Silence compiler warnings
  (void)stride;
  (void)overall_n;
  // Don't invert between steps
  bool default_inv = inv;
  inv = false;
  load();
  inv = default_inv;
}

template <>
METAL_FUNC void
ReadWriter<float2, float2, /*step=*/1, /*real=*/true>::write_strided(
    int stride,
    int overall_n) thread {
  int overall_n_over_2 = overall_n / 2 + 1;
  int coalesce_width = grid.y;
  int tg_idx = elem.y * grid.z + elem.z;
  int outer_batch_size = stride / coalesce_width;

  int strided_batch_idx = (elem.x % outer_batch_size) * coalesce_width +
      overall_n_over_2 * (elem.x / outer_batch_size);
  strided_device_idx = strided_batch_idx +
      tg_idx / coalesce_width * elems_per_thread / 2 * stride +
      tg_idx % coalesce_width;
  strided_shared_idx = (tg_idx % coalesce_width) * n +
      tg_idx / coalesce_width * elems_per_thread / 2;
  for (int e = 0; e < elems_per_thread / 2; e++) {
    float2 output = buf[strided_shared_idx + e];
    out[strided_device_idx + e * stride] = output;
  }

  // Add on n/2 + 1 element
  if (tg_idx == 0 && elem.x % outer_batch_size == 0) {
    out[strided_batch_idx + overall_n / 2] = buf[n / 2];
  }
}

// Four Step IRFFT
template <>
METAL_FUNC void
ReadWriter<float2, float2, /*step=*/0, /*real=*/true>::load_strided(
    int stride,
    int overall_n) thread {
  int overall_n_over_2 = overall_n / 2 + 1;
  auto conj = float2(1, -1);

  compute_strided_indices(stride, overall_n);
  // Translate indices in terms of N - k
  for (int e = 0; e < elems_per_thread; e++) {
    int device_idx = strided_device_idx + e * stride;
    int overall_batch = device_idx / overall_n;
    int overall_index = device_idx % overall_n;
    if (overall_index < overall_n_over_2) {
      device_idx -= overall_batch * (overall_n - overall_n_over_2);
      buf[strided_shared_idx + e] = in[device_idx] * conj;
    } else {
      int conj_idx = overall_n - overall_index;
      device_idx = overall_batch * overall_n_over_2 + conj_idx;
      buf[strided_shared_idx + e] = in[device_idx];
    }
  }
}

template <>
METAL_FUNC void
ReadWriter<float2, float, /*step=*/1, /*real=*/true>::load_strided(
    int stride,
    int overall_n) thread {
  // Silence compiler warnings
  (void)stride;
  (void)overall_n;
  bool default_inv = inv;
  inv = false;
  load();
  inv = default_inv;
}

template <>
METAL_FUNC void
ReadWriter<float2, float, /*step=*/1, /*real=*/true>::write_strided(
    int stride,
    int overall_n) thread {
  compute_strided_indices(stride, overall_n);

  for (int e = 0; e < elems_per_thread; e++) {
    out[strided_device_idx + e * stride] =
        pre_out(buf[strided_shared_idx + e], overall_n).x;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/steel/defines.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/steel/defines.h"
// Copyright © 2024 Apple Inc.


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/fft.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/fft.h"
// Copyright © 2024 Apple Inc.

// Metal FFT using Stockham's algorithm
//
// References:
// - VkFFT (https://github.com/DTolm/VkFFT)
// - Eric Bainville's excellent page (http://www.bealto.com/gpu-fft.html)

#include <metal_common>


using namespace metal;

#define MAX_RADIX 13
// Reached when elems_per_thread_ = 6, max_radix = 13
// and some threads have to do 3 radix 6s requiring 18 complex values.
#define MAX_OUTPUT_SIZE 18

// Specialize for a particular value of N at runtime
STEEL_CONST bool inv_ [[function_constant(0)]];
STEEL_CONST bool is_power_of_2_ [[function_constant(1)]];
STEEL_CONST int elems_per_thread_ [[function_constant(2)]];
// rader_m = n / rader_n
STEEL_CONST int rader_m_ [[function_constant(3)]];
// Stockham steps
STEEL_CONST int radix_13_steps_ [[function_constant(4)]];
STEEL_CONST int radix_11_steps_ [[function_constant(5)]];
STEEL_CONST int radix_8_steps_ [[function_constant(6)]];
STEEL_CONST int radix_7_steps_ [[function_constant(7)]];
STEEL_CONST int radix_6_steps_ [[function_constant(8)]];
STEEL_CONST int radix_5_steps_ [[function_constant(9)]];
STEEL_CONST int radix_4_steps_ [[function_constant(10)]];
STEEL_CONST int radix_3_steps_ [[function_constant(11)]];
STEEL_CONST int radix_2_steps_ [[function_constant(12)]];
// Rader steps
STEEL_CONST int rader_13_steps_ [[function_constant(13)]];
STEEL_CONST int rader_11_steps_ [[function_constant(14)]];
STEEL_CONST int rader_8_steps_ [[function_constant(15)]];
STEEL_CONST int rader_7_steps_ [[function_constant(16)]];
STEEL_CONST int rader_6_steps_ [[function_constant(17)]];
STEEL_CONST int rader_5_steps_ [[function_constant(18)]];
STEEL_CONST int rader_4_steps_ [[function_constant(19)]];
STEEL_CONST int rader_3_steps_ [[function_constant(20)]];
STEEL_CONST int rader_2_steps_ [[function_constant(21)]];
STEEL_CONST bool use_bluestein_twiddle_table_ [[function_constant(22)]];

// Generate the compact radix-8 Stockham twiddle table used by the fused
// 4096-point Bluestein convolution. The first p=1 stage needs no twiddles;
// the remaining stages contribute 8, 64, and 512 entries.
[[kernel]] void generate_bluestein_twiddles(
    device float2* twiddles [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  uint p;
  uint k;
  if (index < 8) {
    p = 8;
    k = index;
  } else if (index < 72) {
    p = 64;
    k = index - 8;
  } else {
    p = 512;
    k = index - 72;
  }
  twiddles[index] = get_twiddle<float>(k, 8 * p);
}

// See "radix.h" for radix codelets.
template <typename T>
using RadixFunc = void (*)(thread vec<T, 2>*, thread vec<T, 2>*);

// Perform a single radix n butterfly with appropriate twiddles
template <
    typename T,
    bool use_twiddle_table,
    int radix,
    RadixFunc<T> radix_func>
METAL_FUNC void radix_butterfly(
    int i,
    int p,
    int twiddle_offset,
    thread vec<T, 2>* x,
    thread short* indices,
    thread vec<T, 2>* y,
    const device vec<T, 2>* twiddles) {
  // i: the index in the overall DFT that we're processing.
  // p: the size of the DFTs we're merging at this step.
  // m: how many threads are working on this DFT.
  int k, j;

  // Use faster bitwise operations when working with powers of two
  constexpr bool radix_p_2 = (radix & (radix - 1)) == 0;
  if (radix_p_2 && is_power_of_2_) {
    constexpr short power = __builtin_ctz(radix);
    k = i & (p - 1);
    j = ((i - k) << power) + k;
  } else {
    k = i % p;
    j = (i / p) * radix * p + k;
  }

  // Apply twiddles
  if (p > 1) {
    vec<T, 2> twiddle_1;
    if constexpr (use_twiddle_table) {
      twiddle_1 = twiddles[twiddle_offset + k];
    } else {
      twiddle_1 = get_twiddle<T>(k, radix * p);
    }
    vec<T, 2> twiddle = twiddle_1;
    x[1] = complex_mul<T>(x[1], twiddle);

    STEEL_PRAGMA_UNROLL
    for (int t = 2; t < radix; t++) {
      twiddle = complex_mul<T>(twiddle, twiddle_1);
      x[t] = complex_mul<T>(x[t], twiddle);
    }
  }

  radix_func(x, y);

  STEEL_PRAGMA_UNROLL
  for (int t = 0; t < radix; t++) {
    indices[t] = j + t * p;
  }
}

// Perform all the radix steps required for a
// particular radix size n.
template <
    typename T,
    bool use_twiddle_table,
    int radix,
    RadixFunc<T> radix_func>
METAL_FUNC void radix_n_steps(
    int i,
    thread int* p,
    thread int* twiddle_offset,
    int m,
    int n,
    int num_steps,
    thread vec<T, 2>* inputs,
    thread short* indices,
    thread vec<T, 2>* values,
    threadgroup vec<T, 2>* buf,
    const device vec<T, 2>* twiddles) {
  int m_r = n / radix;
  // When combining different sized radices, we have to do
  // multiple butterflies in a single thread.
  // E.g. n = 28 = 4 * 7
  // 4 threads, 7 elems_per_thread
  // All threads do 1 radix7 butterfly.
  // 3 threads do 2 radix4 butterflies.
  // 1 thread does 1 radix4 butterfly.
  int max_radices_per_thread = (elems_per_thread_ + radix - 1) / radix;

  int index = 0;
  int r_index = 0;
  for (int s = 0; s < num_steps; s++) {
    for (int t = 0; t < max_radices_per_thread; t++) {
      index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          inputs[r] = buf[index + r * m_r];
        }
        radix_butterfly<T, use_twiddle_table, radix, radix_func>(
            index,
            *p,
            *twiddle_offset,
            inputs,
            indices + t * radix,
            values + t * radix,
            twiddles);
      }
    }

    // Wait until all threads have read their inputs into thread local mem
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < max_radices_per_thread; t++) {
      index = i + t * m;
      if (index < m_r) {
        for (int r = 0; r < radix; r++) {
          r_index = t * radix + r;
          buf[indices[r_index]] = values[r_index];
        }
      }
    }

    // Wait until all threads have written back to threadgroup mem
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (use_twiddle_table) {
      if (*p > 1) {
        *twiddle_offset += *p;
      }
    }
    *p *= radix;
  }
}

#define RADIX_STEP(radix, radix_func, num_steps)             \
  radix_n_steps<T, use_twiddle_table, radix, radix_func<T>>( \
      fft_idx,                                               \
      p,                                                     \
      &twiddle_offset,                                       \
      m,                                                     \
      n,                                                     \
      num_steps,                                             \
      inputs,                                                \
      indices,                                               \
      values,                                                \
      buf,                                                   \
      twiddles);

template <typename T, bool rader = false, bool use_twiddle_table = false>
METAL_FUNC void perform_fft(
    int fft_idx,
    thread int* p,
    int m,
    int n,
    threadgroup vec<T, 2>* buf,
    const device vec<T, 2>* twiddles = nullptr) {
  vec<T, 2> inputs[MAX_RADIX];
  short indices[MAX_OUTPUT_SIZE];
  vec<T, 2> values[MAX_OUTPUT_SIZE];
  int twiddle_offset = 0;

  RADIX_STEP(2, radix2, rader ? rader_2_steps_ : radix_2_steps_);
  RADIX_STEP(3, radix3, rader ? rader_3_steps_ : radix_3_steps_);
  RADIX_STEP(4, radix4, rader ? rader_4_steps_ : radix_4_steps_);
  RADIX_STEP(5, radix5, rader ? rader_5_steps_ : radix_5_steps_);
  RADIX_STEP(6, radix6, rader ? rader_6_steps_ : radix_6_steps_);
  RADIX_STEP(7, radix7, rader ? rader_7_steps_ : radix_7_steps_);
  RADIX_STEP(8, radix8, rader ? rader_8_steps_ : radix_8_steps_);
  RADIX_STEP(11, radix11, rader ? rader_11_steps_ : radix_11_steps_);
  RADIX_STEP(13, radix13, rader ? rader_13_steps_ : radix_13_steps_);
}

// Each FFT is computed entirely in shared GPU memory.
//
// N is decomposed into radix-n DFTs:
// e.g. 128 = 2 * 4 * 4 * 4
template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  threadgroup vec<scalar_T, 2> shared_in[tg_mem_size];

  thread ReadWriter<in_T, out_T> read_writer = ReadWriter<in_T, out_T>(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      inv_);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load();

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  int fft_idx = elem.z; // Thread index in DFT
  int m = grid.z; // Threads per DFT
  int tg_idx = elem.y * n; // Index of this DFT in threadgroup
  threadgroup vec<scalar_T, 2>* buf = &shared_in[tg_idx];

  perform_fft<scalar_T>(fft_idx, &p, m, n, buf);

  read_writer.write();
}

template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void rader_fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    const device vec<typename FFTIOTypeTraits<in_T, out_T>::scalar_T, 2>*
        raders_b_q [[buffer(2)]],
    const device short* raders_g_q [[buffer(3)]],
    const device short* raders_g_minus_q [[buffer(4)]],
    constant const int& n,
    constant const int& batch_size,
    constant const int& rader_n,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  // Use Rader's algorithm to compute fast FFTs
  // when a prime factor `p` of `n` is greater than 13 but
  // has `p - 1` Stockham decomposable into prime factors <= 13.
  //
  // E.g. n = 102
  //        = 2 * 3 * 17
  // .      = 2 * 3 * RADER(16)
  // .      = 2 * 3 * RADER(4 * 4)
  //
  // In numpy:
  //   x_perm = x[g_q]
  //   y = np.fft.fft(x_perm) * b_q
  //   z = np.fft.ifft(y) + x[0]
  //   out = z[g_minus_q]
  //   out[0]  = x[1:].sum()
  //
  // Where the g_q and g_minus_q are permutations formed
  // by the group under multiplicative modulo N using the
  // primitive root of N and b_q is a constant.
  // See https://en.wikipedia.org/wiki/Rader%27s_FFT_algorithm
  //
  // Rader's uses fewer operations than Bluestein's and so
  // is more accurate. It's also faster in most cases.
  threadgroup vec<scalar_T, 2> shared_in[tg_mem_size];

  thread ReadWriter<in_T, out_T> read_writer = ReadWriter<in_T, out_T>(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      inv_);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load();

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The number of the threads we're using for each DFT
  int m = grid.z;

  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  threadgroup vec<scalar_T, 2>* buf = &shared_in[tg_idx];

  // rader_m = n / rader_n;
  int rader_m = rader_m_;

  // We have to load two x_0s for each thread since sometimes
  // elems_per_thread_ crosses a boundary.
  // E.g. with n = 34, rader_n = 17, elems_per_thread_ = 4
  // 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 8 8
  // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  short x_0_index =
      metal::min(fft_idx * elems_per_thread_ / (rader_n - 1), rader_m - 1);
  vec<scalar_T, 2> x_0[2] = {buf[x_0_index], buf[x_0_index + 1]};

  // Do the Rader permutation in shared memory
  vec<scalar_T, 2> temp[MAX_RADIX];
  int max_index = n - rader_m - 1;
  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short g_q = raders_g_q[index / rader_m];
    temp[e] = buf[rader_m + (g_q - 1) * rader_m + index % rader_m];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    buf[index + rader_m] = temp[e];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Rader FFT on x[rader_m:]
  int p = 1;
  perform_fft<scalar_T, /*rader=*/true>(
      fft_idx, &p, m, n - rader_m, buf + rader_m);

  // x_1 + ... + x_n is computed for us in the first FFT step so
  // we save it in the first rader_m indices of the array for later.
  int x_sum_index = metal::min(fft_idx, rader_m - 1);
  buf[x_sum_index] = buf[rader_m + x_sum_index * (rader_n - 1)];

  // The convolution is conjugated and scaled by 1/(rader_n - 1) exactly once
  // each. Float lanes apply the scale after the inverse FFT, preserving the
  // existing float code generation; reduced lanes apply it before, so the
  // intermediates cannot overflow the narrow lane range.
  scalar_T rader_inv_r = static_cast<scalar_T>(1.0f / (rader_n - 1));
  vec<scalar_T, 2> convolution_factor = {1.0f, -1.0f};
  vec<scalar_T, 2> convolution_inv_factor = {rader_inv_r, -rader_inv_r};
  if constexpr (!metal::is_same_v<scalar_T, float>) {
    convolution_factor = convolution_inv_factor;
    convolution_inv_factor = {1.0f, -1.0f};
  }

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short interleaved_index =
        index / rader_m + (index % rader_m) * (rader_n - 1);
    temp[e] = complex_mul<scalar_T>(
        buf[rader_m + interleaved_index],
        raders_b_q[interleaved_index % (rader_n - 1)]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    buf[rader_m + index] = temp[e] * convolution_factor;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Rader IFFT on x[rader_m:]
  p = 1;
  perform_fft<scalar_T, /*rader=*/true>(
      fft_idx, &p, m, n - rader_m, buf + rader_m);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, n - rader_m - 1);
    short diff_index = index / (rader_n - 1) - x_0_index;
    temp[e] = buf[rader_m + index] * convolution_inv_factor + x_0[diff_index];
  }

  // Use the sum of elements that was computed in the first FFT
  vec<scalar_T, 2> x_sum = buf[x_0_index] + x_0[0];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short g_q_index = index % (rader_n - 1);
    short g_q = raders_g_minus_q[g_q_index];
    short out_index = index - g_q_index + g_q + (index / (rader_n - 1));
    buf[out_index] = temp[e];
  }

  buf[x_0_index * rader_n] = x_sum;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  p = rader_n;
  perform_fft<scalar_T>(fft_idx, &p, m, n, buf);

  read_writer.write();
}

template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void bluestein_fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    const device vec<typename FFTIOTypeTraits<in_T, out_T>::scalar_T, 2>* w_q
    [[buffer(2)]],
    const device vec<typename FFTIOTypeTraits<in_T, out_T>::scalar_T, 2>* w_k
    [[buffer(3)]],
    constant const int& length,
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  // Computes arbitrary length FFTs with Bluestein's algorithm
  //
  // In numpy:
  //   bluestein_n = next_power_of_2(2*n - 1)
  //   out = w_k * np.fft.ifft(np.fft.fft(w_k * in, bluestein_n) * w_q)
  //
  // Where w_k and w_q are precomputed on CPU in high precision as:
  //   w_k = np.exp(-1j * np.pi / n * (np.arange(-n + 1, n) ** 2))
  //   w_q = np.fft.fft(1/w_k[-n:])
  threadgroup vec<scalar_T, 2> shared_in[tg_mem_size];

  thread ReadWriter<in_T, out_T> read_writer = ReadWriter<in_T, out_T>(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      inv_);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load_padded(length, w_k);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  int fft_idx = elem.z; // Thread index in DFT
  int m = grid.z; // Threads per DFT
  int tg_idx = elem.y * n; // Index of this DFT in threadgroup
  threadgroup vec<scalar_T, 2>* buf = &shared_in[tg_idx];
  const device vec<scalar_T, 2>* twiddles = w_q + n;

  // fft
  if constexpr (
      tg_mem_size == 4096 && metal::is_same_v<in_T, float2> &&
      metal::is_same_v<out_T, float2>) {
    if (use_bluestein_twiddle_table_) {
      perform_fft<scalar_T, false, true>(fft_idx, &p, m, n, buf, twiddles);
    } else {
      perform_fft<scalar_T>(fft_idx, &p, m, n, buf);
    }
  } else {
    perform_fft<scalar_T>(fft_idx, &p, m, n, buf);
  }

  vec<scalar_T, 2> convolution_factor = {1.0f, -1.0f};
  if constexpr (!metal::is_same_v<scalar_T, float>) {
    // Reduced lanes normalize the convolution before the inverse FFT so the
    // intermediates cannot overflow; the write path then only conjugates.
    scalar_T inv_n = static_cast<scalar_T>(1.0f / n);
    convolution_factor = {inv_n, -inv_n};
  }
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    buf[index] =
        complex_mul<scalar_T>(buf[index], w_q[index]) * convolution_factor;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ifft
  p = 1;
  if constexpr (
      tg_mem_size == 4096 && metal::is_same_v<in_T, float2> &&
      metal::is_same_v<out_T, float2>) {
    if (use_bluestein_twiddle_table_) {
      perform_fft<scalar_T, false, true>(fft_idx, &p, m, n, buf, twiddles);
    } else {
      perform_fft<scalar_T>(fft_idx, &p, m, n, buf);
    }
  } else {
    perform_fft<scalar_T>(fft_idx, &p, m, n, buf);
  }

  read_writer.write_padded(length, w_k);
}

template <
    int tg_mem_size,
    typename in_T,
    typename out_T,
    int step,
    bool real = false>
[[kernel]] void four_step_fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    constant const int& n1,
    constant const int& n2,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  using scalar_T = typename FFTIOTypeTraits<in_T, out_T>::scalar_T;
  // Fast four step FFT implementation for powers of 2.
  int overall_n = n1 * n2;
  int n = step == 0 ? n1 : n2;
  int stride = step == 0 ? n2 : n1;

  // The number of the threads we're using for each DFT
  int m = grid.z;
  int fft_idx = elem.z;

  threadgroup vec<scalar_T, 2> shared_in[tg_mem_size];
  threadgroup vec<scalar_T, 2>* buf = &shared_in[elem.y * n];

  using read_writer_t = ReadWriter<in_T, out_T, step, real>;
  read_writer_t read_writer = read_writer_t(
      in,
      &shared_in[0],
      out,
      n,
      batch_size,
      elems_per_thread_,
      elem,
      grid,
      inv_);

  if (read_writer.out_of_bounds()) {
    return;
  };
  read_writer.load_strided(stride, overall_n);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  int p = 1;
  perform_fft<scalar_T>(fft_idx, &p, m, n, buf);

  read_writer.write_strided(stride, overall_n);
}

///////////////////////////////////////////////////////////////////////////////
)preamble";
}

} // namespace mlx::core::metal
