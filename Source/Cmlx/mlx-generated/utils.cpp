namespace mlx::core::metal {

const char* utils() {
  return R"preamble(
using namespace metal;
typedef bfloat bfloat16_t;
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
METAL_FUNC bfloat16_t simd_broadcast(bfloat16_t data, ushort broadcast_lane_id) { return as_type<bfloat16_t>(__metal_simd_broadcast(as_type<uint16_t>(data), broadcast_lane_id)); } METAL_FUNC bfloat16_t simd_shuffle(bfloat16_t data, ushort simd_lane_id) { return as_type<bfloat16_t>(__metal_simd_shuffle(as_type<uint16_t>(data), simd_lane_id)); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_down( bfloat16_t data, bfloat16_t filling_data, ushort delta, ushort modulo) { return as_type<bfloat16_t>(__metal_simd_shuffle_and_fill_down( as_type<uint16_t>(data), as_type<uint16_t>(filling_data), delta, modulo)); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_down( bfloat16_t data, bfloat16_t filling_data, ushort delta) { return as_type<bfloat16_t>(__metal_simd_shuffle_and_fill_down( as_type<uint16_t>(data), as_type<uint16_t>(filling_data), delta, __metal_get_simdgroup_size(ushort()))); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_up( bfloat16_t data, bfloat16_t filling_data, ushort delta, ushort modulo) { return as_type<bfloat16_t>(__metal_simd_shuffle_and_fill_up( as_type<uint16_t>(data), as_type<uint16_t>(filling_data), delta, modulo)); } METAL_FUNC bfloat16_t simd_shuffle_and_fill_up( bfloat16_t data, bfloat16_t filling_data, ushort delta) { return as_type<bfloat16_t>(__metal_simd_shuffle_and_fill_up( as_type<uint16_t>(data), as_type<uint16_t>(filling_data), delta, __metal_get_simdgroup_size(ushort()))); } METAL_FUNC bfloat16_t simd_shuffle_down(bfloat16_t data, ushort delta) { return as_type<bfloat16_t>(__metal_simd_shuffle_down(as_type<uint16_t>(data), delta)); } METAL_FUNC bfloat16_t simd_shuffle_rotate_down(bfloat16_t data, ushort delta) { return as_type<bfloat16_t>(__metal_simd_shuffle_rotate_down(as_type<uint16_t>(data), delta)); } METAL_FUNC bfloat16_t simd_shuffle_rotate_up(bfloat16_t data, ushort delta) { return as_type<bfloat16_t>(__metal_simd_shuffle_rotate_up(as_type<uint16_t>(data), delta)); } METAL_FUNC bfloat16_t simd_shuffle_up(bfloat16_t data, ushort delta) { return as_type<bfloat16_t>(__metal_simd_shuffle_up(as_type<uint16_t>(data), delta)); } METAL_FUNC bfloat16_t simd_shuffle_xor(bfloat16_t data, ushort mask) { return as_type<bfloat16_t>(__metal_simd_shuffle_xor(as_type<uint16_t>(data), mask)); };
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
)preamble";
}

} // namespace mlx::core::metal
