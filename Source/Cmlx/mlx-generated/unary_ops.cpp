namespace mlx::core::metal {

const char* unary_ops() {
  return R"preamble(
using ieee_float_shape_type = union {
  float value;
  uint32_t word;
};
inline void get_float_word(thread uint32_t& i, float d) {
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}
inline void get_float_word(thread int32_t& i, float d) {
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}
inline void set_float_word(thread float& d, uint32_t i) {
  ieee_float_shape_type sf_u;
  sf_u.word = (i);
  (d) = sf_u.value;
}
inline float frexp_expf(float x, thread int* expt) {
  const uint32_t k = 235;
  const float kln2 = 162.88958740F;
  float exp_x;
  uint32_t hx;
  exp_x = metal::exp(x - kln2);
  get_float_word(hx, exp_x);
  *expt = (hx >> 23) - (0x7f + 127) + k;
  set_float_word(exp_x, (hx & 0x7fffff) | ((0x7f + 127) << 23));
  return exp_x;
}
inline complex64_t ldexp_cexpf(complex64_t z, int expt) {
  float x, y, exp_x, scale1, scale2;
  int ex_expt, half_expt;
  x = z.real;
  y = z.imag;
  exp_x = frexp_expf(x, &ex_expt);
  expt += ex_expt;
  half_expt = expt / 2;
  set_float_word(scale1, (0x7f + half_expt) << 23);
  half_expt = expt - half_expt;
  set_float_word(scale2, (0x7f + half_expt) << 23);
  return complex64_t{
      metal::cos(y) * exp_x * scale1 * scale2,
      metal::sin(y) * exp_x * scale1 * scale2};
}
inline complex64_t cexpf(const thread complex64_t& z) {
  float x, y, exp_x;
  uint32_t hx, hy;
  const uint32_t exp_ovfl = 0x42b17218, cexp_ovfl = 0x43400074;
  x = z.real;
  y = z.imag;
  get_float_word(hy, y);
  hy &= 0x7fffffff;
  if (hy == 0) {
    return complex64_t{metal::exp(x), y};
  }
  get_float_word(hx, x);
  if ((hx & 0x7fffffff) == 0) {
    return complex64_t{metal::cos(y), metal::sin(y)};
  }
  if (hy >= 0x7f800000) {
    if ((hx & 0x7fffffff) != 0x7f800000) {
      return complex64_t{y - y, y - y};
    } else if (hx & 0x80000000) {
      return complex64_t{0.0, 0.0};
    } else {
      return complex64_t{x, y - y};
    }
  }
  if (hx >= exp_ovfl && hx <= cexp_ovfl) {
    return ldexp_cexpf(z, 0);
  } else {
    exp_x = metal::exp(x);
    return complex64_t{exp_x * metal::cos(y), exp_x * metal::sin(y)};
  }
}
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
float expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;
  j = fma(1.442695f, a, 12582912.f);
  j = j - 12582912.0f;
  i = (int)j;
  f = fma(j, -6.93145752e-1f, a);
  s = f * f;
  if (a == 0.0f)
    s = a;
  r = 1.97350979e-4f;
  r = fma(r, f, 1.39309070e-3f);
  r = fma(r, f, 8.33343994e-3f);
  r = fma(r, f, 4.16668020e-2f);
  r = fma(r, f, 1.66666716e-1f);
  r = fma(r, f, 4.99999970e-1f);
  u = (j == 1) ? (f + 0.5f) : f;
  v = fma(r, s, u);
  s = 0.5f * b;
  t = ldexp(s, i);
  y = t - s;
  x = (t - y) - s;
  r = fma(v, t, x) + y;
  r = r + r;
  if (j == 0)
    r = v;
  if (j == 1)
    r = v + v;
  return r;
}
float expm1f(float a) {
  float r;
  r = expm1f_scaled_unchecked(a, 1.0f);
  if (abs(a - 1.0f) > 88.0f) {
    r = pow(2, a);
    r = fma(r, r, -1.0f);
  }
  return r;
}
struct fp8_e4m3 {
  template <typename T>
  fp8_e4m3(T f) {
    uint32_t fp8_max = 543 << 21;
    uint32_t denorm_mask = 141 << 23;
    uint32_t f_bits = as_type<uint32_t>(static_cast<float>(f));
    uint32_t sign = f_bits & 0x80000000;
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      bits = 0x7E;
    } else {
      if (f_bits < (121 << 23)) {
        f_bits = as_type<uint32_t>(
            as_type<float>(f_bits) + as_type<float>(denorm_mask));
        bits = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        bits = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    bits |= static_cast<uint8_t>(sign >> 24);
  }
  operator float() {
    uint32_t w = static_cast<uint32_t>(bits) << 24;
    uint32_t sign = w & 0x80000000;
    uint32_t nonsign = w & 0x7FFFFFFF;
    uint32_t renorm_shift = metal::clz(nonsign);
    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
    int32_t inf_nan_mask =
        (static_cast<int32_t>(nonsign + 0x01000000) >> 8) & 0x7F800000;
    int32_t zero_mask = static_cast<int32_t>(nonsign - 1) >> 31;
    uint32_t result = sign |
        ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
          inf_nan_mask) &
         ~zero_mask);
    return as_type<float>(result);
  }
  uint8_t bits;
};
struct fp8_e8m0 {
  fp8_e8m0(float x) {
    if (!metal::isfinite(x)) {
      bits = 0xFF;
      return;
    }
    if (x < 0.0f) {
      bits = 0x00;
      return;
    }
    float le = metal::log2(x);
    int n = int(metal::round(le));
    n = n < -127 ? -127 : n;
    n = n > 127 ? 127 : n;
    bits = static_cast<uint8_t>(n + 127);
  }
  operator bfloat16_t() {
    uint16_t out = (bits == 0 ? 0x40 : (static_cast<uint16_t>(bits) << 7));
    return as_type<bfloat16_t>(out);
  }
  operator float() {
    return static_cast<float>(this->operator bfloat16_t());
  }
  uint8_t bits;
};

namespace {
constant float inf = metal::numeric_limits<float>::infinity();
}
struct Abs {
  template <typename T>
  T operator()(T x) {
    return metal::abs(x);
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
  complex64_t operator()(complex64_t x) {
    return {metal::precise::sqrt(x.real * x.real + x.imag * x.imag), 0};
  };
};
struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::acos(x);
  };
  complex64_t operator()(complex64_t x);
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
  complex64_t operator()(complex64_t x);
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
  complex64_t operator()(complex64_t x);
};
struct ArcTanh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::atanh(x);
  };
};
struct BitwiseInvert {
  template <typename T>
  T operator()(T x) {
    return ~x;
  };
};
struct Ceil {
  template <typename T>
  T operator()(T x) {
    return metal::ceil(x);
  };
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};
struct Cos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cos(x);
  };
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
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::cosh(x.real) * metal::precise::cos(x.imag),
        metal::precise::sinh(x.real) * metal::precise::sin(x.imag)};
  };
};
struct Conjugate {
  complex64_t operator()(complex64_t x) {
    return complex64_t{x.real, -x.imag};
  }
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
  complex64_t operator()(complex64_t x) {
    return cexpf(x);
  }
};
struct Expm1 {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(expm1f(static_cast<float>(x)));
  };
};
struct Floor {
  template <typename T>
  T operator()(T x) {
    return metal::floor(x);
  };
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};
struct Imag {
  float operator()(complex64_t x) {
    return x.imag;
  };
};
struct Log {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log(x);
  };
  complex64_t operator()(complex64_t x) {
    auto r = metal::precise::log(Abs{}(x).real);
    auto i = metal::precise::atan2(x.imag, x.real);
    return {r, i};
  };
};
struct Log2 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log2(x);
  };
  complex64_t operator()(complex64_t x) {
    auto y = Log{}(x);
    return {y.real / M_LN2_F, y.imag / M_LN2_F};
  };
};
struct Log10 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log10(x);
  };
  complex64_t operator()(complex64_t x) {
    auto y = Log{}(x);
    return {y.real / M_LN10_F, y.imag / M_LN10_F};
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
struct Real {
  float operator()(complex64_t x) {
    return x.real;
  };
};
struct Round {
  template <typename T>
  T operator()(T x) {
    return metal::rint(x);
  };
  complex64_t operator()(complex64_t x) {
    return {metal::rint(x.real), metal::rint(x.imag)};
  };
};
struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  };
  uint32_t operator()(uint32_t x) {
    return x != 0;
  };
  complex64_t operator()(complex64_t x) {
    if (x == complex64_t(0)) {
      return x;
    }
    return x /
        (complex64_t)metal::precise::sqrt(x.real * x.real + x.imag * x.imag);
  };
};
struct Sin {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sin(x);
  };
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
  complex64_t operator()(complex64_t x) {
    if (x.real == 0.0 && x.imag == 0.0) {
      return {0.0, 0.0};
    }
    auto r = Abs{}(x).real;
    auto a = metal::precise::sqrt((r + x.real) / 2.0);
    auto b_abs = metal::precise::sqrt((r - x.real) / 2.0);
    auto b = metal::copysign(b_abs, x.imag);
    return {a, b};
  }
};
struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::rsqrt(x);
  };
  complex64_t operator()(complex64_t x) {
    return 1.0 / Sqrt{}(x);
  }
};
struct Tan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tan(x);
  };
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
  complex64_t operator()(complex64_t x) {
    float tanh_a = metal::precise::tanh(x.real);
    float tan_b = metal::precise::tan(x.imag);
    float t1 = tanh_a * tan_b;
    float denom = 1. + t1 * t1;
    return {(tanh_a + tan_b * t1) / denom, (tan_b - tanh_a * t1) / denom};
  };
};
complex64_t ArcCos::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto y = Log{}(x + i * Sqrt{}(1.0 - x * x));
  return {y.imag, -y.real};
};
complex64_t ArcSin::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto y = Log{}(i * x + Sqrt{}(1.0 - x * x));
  return {y.imag, -y.real};
};
complex64_t ArcTan::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto ix = i * x;
  return (1.0 / complex64_t{0.0, 2.0}) * Log{}((1.0 + ix) / (1.0 - ix));
};
struct ToFP8 {
  template <typename T>
  uint8_t operator()(T f) {
    return fp8_e4m3(f).bits;
  }
};
struct FromFP8 {
  float operator()(uint8_t x) {
    return float(*(thread fp8_e4m3*)(&x));
  }
};
)preamble";
}

} // namespace mlx::core::metal
