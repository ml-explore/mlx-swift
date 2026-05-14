// TurboQuant KV cache compression operations
// Based on TurboQuant paper (Zandieh et al., arXiv 2504.19874)

#include <cstring>
#include <vector>
#include <stdexcept>
#include <string>

#include "mlx/mlx.h"
#include "turbo_quant.h"

namespace {
static constexpr int TURBO_K_RECORD = 68;
static constexpr int TURBO_V_RECORD = 50;
} // anonymous namespace

namespace mlx::core::fast {

static std::pair<mlx::core::array, const float*>
turbo_to_f32(const mlx::core::array& x, mlx::core::StreamOrDevice s) {
  auto x_f32 = mlx::core::astype(x, mlx::core::float32, s);
  mlx::core::eval(x_f32);
  return {x_f32, x_f32.data<float>()};
}

array turbo_encode_k(const array& keys, StreamOrDevice s_) {
  auto s = to_stream(s_);
  const int head_dim = static_cast<int>(keys.shape(-1));
  if (head_dim != 128 && head_dim != 256) {
    throw std::invalid_argument(
        "[turbo_encode_k] last dim must be 128 or 256, got " +
        std::to_string(head_dim));
  }
  const int n_subgroups = head_dim / TURBO_D;
  const int record_bytes = TURBO_K_RECORD * n_subgroups;
  auto [keys_f32, src] = turbo_to_f32(keys, s);
  const int N = static_cast<int>(keys_f32.size() / head_dim);
  std::vector<uint8_t> buf(static_cast<size_t>(N) * record_bytes, 0u);
  for (int i = 0; i < N; ++i) {
    uint8_t* dst = buf.data() + i * record_bytes;
    for (int g = 0; g < n_subgroups; ++g) {
      TurboQuantK rec = turbo_quantize_k(
          src + i * head_dim + g * TURBO_D, TURBO_D);
      uint8_t* sub_dst = dst + g * TURBO_K_RECORD;
      std::memcpy(sub_dst,      rec.indices,     48);
      std::memcpy(sub_dst + 48, rec.qjl_signs,   16);
      std::memcpy(sub_dst + 64, &rec.norm_fp16,   2);
      std::memcpy(sub_dst + 66, &rec.rnorm_fp16,  2);
    }
  }
  Shape out_shape = keys.shape();
  out_shape.back() = record_bytes;
  return array(buf.data(), out_shape, uint8);
}

array turbo_encode_v(const array& values, StreamOrDevice s_) {
  auto s = to_stream(s_);
  const int head_dim = static_cast<int>(values.shape(-1));
  if (head_dim != 128 && head_dim != 256) {
    throw std::invalid_argument(
        "[turbo_encode_v] last dim must be 128 or 256, got " +
        std::to_string(head_dim));
  }
  const int n_subgroups = head_dim / TURBO_D;
  const int record_bytes = TURBO_V_RECORD * n_subgroups;
  auto [vals_f32, src] = turbo_to_f32(values, s);
  const int N = static_cast<int>(vals_f32.size() / head_dim);
  std::vector<uint8_t> buf(static_cast<size_t>(N) * record_bytes, 0u);
  for (int i = 0; i < N; ++i) {
    uint8_t* dst = buf.data() + i * record_bytes;
    for (int g = 0; g < n_subgroups; ++g) {
      TurboQuantV rec = turbo_quantize_v(
          src + i * head_dim + g * TURBO_D, TURBO_D);
      uint8_t* sub_dst = dst + g * TURBO_V_RECORD;
      std::memcpy(sub_dst,      rec.indices,    48);
      std::memcpy(sub_dst + 48, &rec.norm_fp16,  2);
    }
  }
  Shape out_shape = values.shape();
  out_shape.back() = record_bytes;
  return array(buf.data(), out_shape, uint8);
}

array turbo_decode_k(const array& packed, StreamOrDevice s_) {
  auto s = to_stream(s_);
  const int record_bytes = static_cast<int>(packed.shape(-1));
  if (record_bytes != TURBO_K_RECORD && record_bytes != TURBO_K_RECORD * 2) {
    throw std::invalid_argument(
        "[turbo_decode_k] last dim must be 68 or 136, got " +
        std::to_string(record_bytes));
  }
  const int n_subgroups = record_bytes / TURBO_K_RECORD;
  const int head_dim = n_subgroups * TURBO_D;
  auto packed_u8 = astype(packed, uint8, s);
  eval(packed_u8);
  const uint8_t* src = packed_u8.data<uint8_t>();
  const int N = static_cast<int>(packed_u8.size() / record_bytes);
  std::vector<float> buf(static_cast<size_t>(N) * head_dim);
  for (int i = 0; i < N; ++i) {
    for (int g = 0; g < n_subgroups; ++g) {
      const uint8_t* sub_src = src + i * record_bytes + g * TURBO_K_RECORD;
      TurboQuantK rec;
      std::memset(&rec, 0, sizeof(rec));
      std::memcpy(rec.indices,     sub_src,      48);
      std::memcpy(rec.qjl_signs,   sub_src + 48, 16);
      std::memcpy(&rec.norm_fp16,  sub_src + 64,  2);
      std::memcpy(&rec.rnorm_fp16, sub_src + 66,  2);
      turbo_dequantize_k(
          rec,
          buf.data() + i * head_dim + g * TURBO_D,
          TURBO_D);
    }
  }
  Shape out_shape = packed.shape();
  out_shape.back() = head_dim;
  return array(buf.data(), out_shape, float32);
}

array turbo_decode_v(const array& packed, StreamOrDevice s_) {
  auto s = to_stream(s_);
  const int record_bytes = static_cast<int>(packed.shape(-1));
  if (record_bytes != TURBO_V_RECORD && record_bytes != TURBO_V_RECORD * 2) {
    throw std::invalid_argument(
        "[turbo_decode_v] last dim must be 50 or 100, got " +
        std::to_string(record_bytes));
  }
  const int n_subgroups = record_bytes / TURBO_V_RECORD;
  const int head_dim = n_subgroups * TURBO_D;
  auto packed_u8 = astype(packed, uint8, s);
  eval(packed_u8);
  const uint8_t* src = packed_u8.data<uint8_t>();
  const int N = static_cast<int>(packed_u8.size() / record_bytes);
  std::vector<float> buf(static_cast<size_t>(N) * head_dim);
  for (int i = 0; i < N; ++i) {
    for (int g = 0; g < n_subgroups; ++g) {
      const uint8_t* sub_src = src + i * record_bytes + g * TURBO_V_RECORD;
      TurboQuantV rec;
      std::memset(&rec, 0, sizeof(rec));
      std::memcpy(rec.indices,    sub_src,      48);
      std::memcpy(&rec.norm_fp16, sub_src + 48,  2);
      turbo_dequantize_v(
          rec,
          buf.data() + i * head_dim + g * TURBO_D,
          TURBO_D);
    }
  }
  Shape out_shape = packed.shape();
  out_shape.back() = head_dim;
  return array(buf.data(), out_shape, float32);
}

} // namespace mlx::core::fast
