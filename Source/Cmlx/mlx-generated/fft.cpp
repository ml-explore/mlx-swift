namespace mlx::core::metal {

const char* fft() {
  return R"preamble(
METAL_FUNC float2 complex_mul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
METAL_FUNC float2 complex_mul_conj(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, -a.x * b.y - a.y * b.x);
}
METAL_FUNC float2 get_twiddle(int k, int p) {
  float theta = -2.0f * k * M_PI_F / p;
  float2 twiddle = {metal::fast::cos(theta), metal::fast::sin(theta)};
  return twiddle;
}
METAL_FUNC void radix2(thread float2* x, thread float2* y) {
  y[0] = x[0] + x[1];
  y[1] = x[0] - x[1];
}
METAL_FUNC void radix3(thread float2* x, thread float2* y) {
  float pi_2_3 = -0.8660254037844387;
  float2 a_1 = x[1] + x[2];
  float2 a_2 = x[1] - x[2];
  y[0] = x[0] + a_1;
  float2 b_1 = x[0] - 0.5 * a_1;
  float2 b_2 = pi_2_3 * a_2;
  float2 b_2_j = {-b_2.y, b_2.x};
  y[1] = b_1 + b_2_j;
  y[2] = b_1 - b_2_j;
}
METAL_FUNC void radix4(thread float2* x, thread float2* y) {
  float2 z_0 = x[0] + x[2];
  float2 z_1 = x[0] - x[2];
  float2 z_2 = x[1] + x[3];
  float2 z_3 = x[1] - x[3];
  float2 z_3_i = {z_3.y, -z_3.x};
  y[0] = z_0 + z_2;
  y[1] = z_1 + z_3_i;
  y[2] = z_0 - z_2;
  y[3] = z_1 - z_3_i;
}
METAL_FUNC void radix5(thread float2* x, thread float2* y) {
  float2 root_5_4 = 0.5590169943749475;
  float2 sin_2pi_5 = 0.9510565162951535;
  float2 sin_1pi_5 = 0.5877852522924731;
  float2 a_1 = x[1] + x[4];
  float2 a_2 = x[2] + x[3];
  float2 a_3 = x[1] - x[4];
  float2 a_4 = x[2] - x[3];
  float2 a_5 = a_1 + a_2;
  float2 a_6 = root_5_4 * (a_1 - a_2);
  float2 a_7 = x[0] - a_5 / 4;
  float2 a_8 = a_7 + a_6;
  float2 a_9 = a_7 - a_6;
  float2 a_10 = sin_2pi_5 * a_3 + sin_1pi_5 * a_4;
  float2 a_11 = sin_1pi_5 * a_3 - sin_2pi_5 * a_4;
  float2 a_10_j = {a_10.y, -a_10.x};
  float2 a_11_j = {a_11.y, -a_11.x};
  y[0] = x[0] + a_5;
  y[1] = a_8 + a_10_j;
  y[2] = a_9 + a_11_j;
  y[3] = a_9 - a_11_j;
  y[4] = a_8 - a_10_j;
}
METAL_FUNC void radix6(thread float2* x, thread float2* y) {
  float sin_pi_3 = 0.8660254037844387;
  float2 a_1 = x[2] + x[4];
  float2 a_2 = x[0] - a_1 / 2;
  float2 a_3 = sin_pi_3 * (x[2] - x[4]);
  float2 a_4 = x[5] + x[1];
  float2 a_5 = x[3] - a_4 / 2;
  float2 a_6 = sin_pi_3 * (x[5] - x[1]);
  float2 a_7 = x[0] + a_1;
  float2 a_3_i = {a_3.y, -a_3.x};
  float2 a_6_i = {a_6.y, -a_6.x};
  float2 a_8 = a_2 + a_3_i;
  float2 a_9 = a_2 - a_3_i;
  float2 a_10 = x[3] + a_4;
  float2 a_11 = a_5 + a_6_i;
  float2 a_12 = a_5 - a_6_i;
  y[0] = a_7 + a_10;
  y[1] = a_8 - a_11;
  y[2] = a_9 + a_12;
  y[3] = a_7 - a_10;
  y[4] = a_8 + a_11;
  y[5] = a_9 - a_12;
}
METAL_FUNC void radix7(thread float2* x, thread float2* y) {
  float2 inv = {1 / 6.0, -1 / 6.0};
  float2 in1[6] = {x[1], x[3], x[2], x[6], x[4], x[5]};
  radix6(in1, y + 1);
  y[0] = y[1] + x[0];
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(2.44013336, -1.02261879));
  y[3] = complex_mul_conj(y[3], float2(2.37046941, -1.17510629));
  y[4] = complex_mul_conj(y[4], float2(0, -2.64575131));
  y[5] = complex_mul_conj(y[5], float2(2.37046941, 1.17510629));
  y[6] = complex_mul_conj(y[6], float2(-2.44013336, -1.02261879));
  radix6(y + 1, x + 1);
  y[1] = x[1] * inv + x[0];
  y[5] = x[2] * inv + x[0];
  y[4] = x[3] * inv + x[0];
  y[6] = x[4] * inv + x[0];
  y[2] = x[5] * inv + x[0];
  y[3] = x[6] * inv + x[0];
}
METAL_FUNC void radix8(thread float2* x, thread float2* y) {
  float cos_pi_4 = 0.7071067811865476;
  float2 w_0 = {cos_pi_4, -cos_pi_4};
  float2 w_1 = {-cos_pi_4, -cos_pi_4};
  float2 temp[8] = {x[0], x[2], x[4], x[6], x[1], x[3], x[5], x[7]};
  radix4(temp, x);
  radix4(temp + 4, x + 4);
  y[0] = x[0] + x[4];
  y[4] = x[0] - x[4];
  float2 x_5 = complex_mul(x[5], w_0);
  y[1] = x[1] + x_5;
  y[5] = x[1] - x_5;
  float2 x_6 = {x[6].y, -x[6].x};
  y[2] = x[2] + x_6;
  y[6] = x[2] - x_6;
  float2 x_7 = complex_mul(x[7], w_1);
  y[3] = x[3] + x_7;
  y[7] = x[3] - x_7;
}
template <bool raders_perm>
METAL_FUNC void radix10(thread float2* x, thread float2* y) {
  float2 w[4];
  w[0] = {0.8090169943749475, -0.5877852522924731};
  w[1] = {0.30901699437494745, -0.9510565162951535};
  w[2] = {-w[1].x, w[1].y};
  w[3] = {-w[0].x, w[0].y};
  if (raders_perm) {
    float2 temp[10] = {
        x[0], x[3], x[4], x[8], x[2], x[1], x[7], x[9], x[6], x[5]};
    radix5(temp, x);
    radix5(temp + 5, x + 5);
  } else {
    float2 temp[10] = {
        x[0], x[2], x[4], x[6], x[8], x[1], x[3], x[5], x[7], x[9]};
    radix5(temp, x);
    radix5(temp + 5, x + 5);
  }
  y[0] = x[0] + x[5];
  y[5] = x[0] - x[5];
  for (int t = 1; t < 5; t++) {
    float2 a = complex_mul(x[t + 5], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 5] = x[t] - a;
  }
}
METAL_FUNC void radix11(thread float2* x, thread float2* y) {
  float2 inv = {1 / 10.0, -1 / 10.0};
  radix10<true>(x + 1, y + 1);
  y[0] = y[1] + x[0];
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(0.955301878, -3.17606649));
  y[3] = complex_mul_conj(y[3], float2(2.63610556, 2.01269656));
  y[4] = complex_mul_conj(y[4], float2(2.54127802, 2.13117479));
  y[5] = complex_mul_conj(y[5], float2(2.07016210, 2.59122150));
  y[6] = complex_mul_conj(y[6], float2(0, -3.31662479));
  y[7] = complex_mul_conj(y[7], float2(2.07016210, -2.59122150));
  y[8] = complex_mul_conj(y[8], float2(-2.54127802, 2.13117479));
  y[9] = complex_mul_conj(y[9], float2(2.63610556, -2.01269656));
  y[10] = complex_mul_conj(y[10], float2(-0.955301878, -3.17606649));
  radix10<false>(y + 1, x + 1);
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
template <bool raders_perm>
METAL_FUNC void radix12(thread float2* x, thread float2* y) {
  float2 w[6];
  float sin_pi_3 = 0.8660254037844387;
  w[0] = {sin_pi_3, -0.5};
  w[1] = {0.5, -sin_pi_3};
  w[2] = {0, -1};
  w[3] = {-0.5, -sin_pi_3};
  w[4] = {-sin_pi_3, -0.5};
  if (raders_perm) {
    float2 temp[12] = {
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
    radix6(temp, x);
    radix6(temp + 6, x + 6);
  } else {
    float2 temp[12] = {
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
    radix6(temp, x);
    radix6(temp + 6, x + 6);
  }
  y[0] = x[0] + x[6];
  y[6] = x[0] - x[6];
  for (int t = 1; t < 6; t++) {
    float2 a = complex_mul(x[t + 6], w[t - 1]);
    y[t] = x[t] + a;
    y[t + 6] = x[t] - a;
  }
}
METAL_FUNC void radix13(thread float2* x, thread float2* y) {
  float2 inv = {1 / 12.0, -1 / 12.0};
  radix12<true>(x + 1, y + 1);
  y[0] = y[1] + x[0];
  y[1] = complex_mul_conj(y[1], float2(-1, 0));
  y[2] = complex_mul_conj(y[2], float2(3.07497206, -1.88269669));
  y[3] = complex_mul_conj(y[3], float2(3.09912468, 1.84266823));
  y[4] = complex_mul_conj(y[4], float2(3.45084438, -1.04483161));
  y[5] = complex_mul_conj(y[5], float2(0.91083583, 3.48860690));
  y[6] = complex_mul_conj(y[6], float2(-3.60286363, 0.139189267));
  y[7] = complex_mul_conj(y[7], float2(3.60555128, 0));
  y[8] = complex_mul_conj(y[8], float2(3.60286363, 0.139189267));
  y[9] = complex_mul_conj(y[9], float2(0.91083583, -3.48860690));
  y[10] = complex_mul_conj(y[10], float2(-3.45084438, -1.04483161));
  y[11] = complex_mul_conj(y[11], float2(3.09912468, -1.84266823));
  y[12] = complex_mul_conj(y[12], float2(-3.07497206, -1.88269669));
  radix12<false>(y + 1, x + 1);
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
using namespace metal;
template <
    typename in_T,
    typename out_T,
    int step = 0,
    bool four_step_real = false>
struct ReadWriter {
  const device in_T* in;
  threadgroup float2* buf;
  device out_T* out;
  int n;
  int batch_size;
  int elems_per_thread;
  uint3 elem;
  uint3 grid;
  int threads_per_tg;
  bool inv;
  int strided_device_idx = 0;
  int strided_shared_idx = 0;
  METAL_FUNC ReadWriter(
      const device in_T* in_,
      threadgroup float2* buf_,
      device out_T* out_,
      const short n_,
      const int batch_size_,
      const short elems_per_thread_,
      const uint3 elem_,
      const uint3 grid_,
      const bool inv_)
      : in(in_),
        buf(buf_),
        out(out_),
        n(n_),
        batch_size(batch_size_),
        elems_per_thread(elems_per_thread_),
        elem(elem_),
        grid(grid_),
        inv(inv_) {
    threads_per_tg = elem.x == grid.x - 1
        ? (batch_size - (grid.x - 1) * grid.y) * grid.z
        : grid.y * grid.z;
  }
  METAL_FUNC float2 post_in(float2 elem) const {
    return inv ? float2(elem.x, -elem.y) : elem;
  }
  METAL_FUNC float2 post_in(float elem) const {
    return float2(elem, 0);
  }
  METAL_FUNC float2 pre_out(float2 elem) const {
    return inv ? float2(elem.x / n, -elem.y / n) : elem;
  }
  METAL_FUNC float2 pre_out(float2 elem, int length) const {
    return inv ? float2(elem.x / length, -elem.y / length) : elem;
  }
  METAL_FUNC bool out_of_bounds() const {
    int grid_index = elem.x * grid.y + elem.y;
    return grid_index >= batch_size;
  }
  METAL_FUNC void load() const {
    int batch_idx = elem.x * grid.y * n;
    short tg_idx = elem.y * grid.z + elem.z;
    short max_index = grid.y * n - 2;
    constexpr int read_width = 2;
    for (short e = 0; e < (elems_per_thread / read_width); e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      index = metal::min(index, max_index);
      buf[index] = post_in(in[batch_idx + index]);
      buf[index + 1] = post_in(in[batch_idx + index + 1]);
    }
    max_index += 1;
    if (elems_per_thread % 2 != 0) {
      short index = tg_idx +
          read_width * threads_per_tg * (elems_per_thread / read_width);
      index = metal::min(index, max_index);
      buf[index] = post_in(in[batch_idx + index]);
    }
  }
  METAL_FUNC void write() const {
    int batch_idx = elem.x * grid.y * n;
    short tg_idx = elem.y * grid.z + elem.z;
    short max_index = grid.y * n - 2;
    constexpr int read_width = 2;
    for (short e = 0; e < (elems_per_thread / read_width); e++) {
      short index = read_width * tg_idx + read_width * threads_per_tg * e;
      index = metal::min(index, max_index);
      out[batch_idx + index] = pre_out(buf[index]);
      out[batch_idx + index + 1] = pre_out(buf[index + 1]);
    }
    max_index += 1;
    if (elems_per_thread % 2 != 0) {
      short index = tg_idx +
          read_width * threads_per_tg * (elems_per_thread / read_width);
      index = metal::min(index, max_index);
      out[batch_idx + index] = pre_out(buf[index]);
    }
  }
  METAL_FUNC void load_padded(int length, const device float2* w_k) const {
    int batch_idx = elem.x * grid.y * length + elem.y * length;
    int fft_idx = elem.z;
    int m = grid.z;
    threadgroup float2* seq_buf = buf + elem.y * n;
    for (int e = 0; e < elems_per_thread; e++) {
      int index = metal::min(fft_idx + e * m, n - 1);
      if (index < length) {
        float2 elem = post_in(in[batch_idx + index]);
        seq_buf[index] = complex_mul(elem, w_k[index]);
      } else {
        seq_buf[index] = 0.0;
      }
    }
  }
  METAL_FUNC void write_padded(int length, const device float2* w_k) const {
    int batch_idx = elem.x * grid.y * length + elem.y * length;
    int fft_idx = elem.z;
    int m = grid.z;
    float2 inv_factor = {1.0f / n, -1.0f / n};
    threadgroup float2* seq_buf = buf + elem.y * n;
    for (int e = 0; e < elems_per_thread; e++) {
      int index = metal::min(fft_idx + e * m, n - 1);
      if (index < length) {
        float2 elem = seq_buf[index + length - 1] * inv_factor;
        out[batch_idx + index] = pre_out(complex_mul(elem, w_k[index]), length);
      }
    }
  }
  METAL_FUNC void compute_strided_indices(int stride, int overall_n) {
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
  METAL_FUNC void load_strided(int stride, int overall_n) {
    compute_strided_indices(stride, overall_n);
    for (int e = 0; e < elems_per_thread; e++) {
      buf[strided_shared_idx + e] =
          post_in(in[strided_device_idx + e * stride]);
    }
  }
  METAL_FUNC void write_strided(int stride, int overall_n) {
    for (int e = 0; e < elems_per_thread; e++) {
      float2 output = buf[strided_shared_idx + e];
      int combined_idx = (strided_device_idx + e * stride) % overall_n;
      int ij = (combined_idx / stride) * (combined_idx % stride);
      float2 twiddle = get_twiddle(ij, overall_n);
      out[strided_device_idx + e * stride] = complex_mul(output, twiddle);
    }
  }
};
template <>
METAL_FUNC void ReadWriter<float2, float2, 1>::load_strided(
    int stride,
    int overall_n) {
  (void)stride;
  (void)overall_n;
  bool default_inv = inv;
  inv = false;
  load();
  inv = default_inv;
}
template <>
METAL_FUNC void ReadWriter<float2, float2, 1>::write_strided(
    int stride,
    int overall_n) {
  compute_strided_indices(stride, overall_n);
  for (int e = 0; e < elems_per_thread; e++) {
    float2 output = buf[strided_shared_idx + e];
    out[strided_device_idx + e * stride] = pre_out(output, overall_n);
  }
}
template <>
METAL_FUNC bool ReadWriter<float, float2>::out_of_bounds() const {
  int grid_index = elem.x * grid.y + elem.y;
  return grid_index * 2 >= batch_size;
}
template <>
METAL_FUNC void ReadWriter<float, float2>::load() const {
  int batch_idx = elem.x * grid.y * n * 2 + elem.y * n * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;
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
METAL_FUNC void ReadWriter<float, float2>::write() const {
  short n_over_2 = (n / 2) + 1;
  int batch_idx = elem.x * grid.y * n_over_2 * 2 + elem.y * n_over_2 * 2;
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
    if (index == 0) {
      out[batch_idx + index] = {seq_buf[index].x, 0};
      out[batch_idx + index + next_out] = {seq_buf[index].y, 0};
    } else {
      float2 x_k = seq_buf[index];
      float2 x_n_minus_k = seq_buf[n - index] * conj;
      out[batch_idx + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx + index + next_out] =
          complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
}
template <>
METAL_FUNC void ReadWriter<float, float2>::load_padded(
    int length,
    const device float2* w_k) const {
  int batch_idx = elem.x * grid.y * length * 2 + elem.y * length * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;
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
      seq_buf[index] = complex_mul(elem, w_k[index]);
    } else {
      seq_buf[index] = 0;
    }
  }
}
template <>
METAL_FUNC void ReadWriter<float, float2>::write_padded(
    int length,
    const device float2* w_k) const {
  int length_over_2 = (length / 2) + 1;
  int batch_idx =
      elem.x * grid.y * length_over_2 * 2 + elem.y * length_over_2 * 2;
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
    if (index == 0) {
      float2 elem = complex_mul(w_k[index], seq_buf[index] * inv_factor);
      out[batch_idx + index] = float2(elem.x, 0);
      out[batch_idx + index + next_out] = float2(elem.y, 0);
    } else {
      float2 x_k = complex_mul(w_k[index], seq_buf[index] * inv_factor);
      float2 x_n_minus_k = complex_mul(
          w_k[length - index], seq_buf[length - index] * inv_factor);
      x_n_minus_k *= conj;
      out[batch_idx + index] = (x_k + x_n_minus_k) / 2;
      out[batch_idx + index + next_out] =
          complex_mul(((x_k - x_n_minus_k) / 2), minus_j);
    }
  }
}
template <>
METAL_FUNC bool ReadWriter<float2, float>::out_of_bounds() const {
  int grid_index = elem.x * grid.y + elem.y;
  return grid_index * 2 >= batch_size;
}
template <>
METAL_FUNC void ReadWriter<float2, float>::load() const {
  short n_over_2 = (n / 2) + 1;
  int batch_idx = elem.x * grid.y * n_over_2 * 2 + elem.y * n_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;
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
    bool first_val = index == 0;
    bool last_val = n % 2 == 0 && index == n_over_2 - 1;
    if (first_val || last_val) {
      x = float2(x.x, 0);
      y = float2(y.x, 0);
    }
    seq_buf[index] = x + complex_mul(y, plus_j);
    seq_buf[index].y = -seq_buf[index].y;
    if (index > 0 && !last_val) {
      seq_buf[n - index] = (x * conj) + complex_mul(y * conj, plus_j);
      seq_buf[n - index].y = -seq_buf[n - index].y;
    }
  }
}
template <>
METAL_FUNC void ReadWriter<float2, float>::write() const {
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
    const device float2* w_k) const {
  int n_over_2 = (n / 2) + 1;
  int length_over_2 = (length / 2) + 1;
  int batch_idx =
      elem.x * grid.y * length_over_2 * 2 + elem.y * length_over_2 * 2;
  threadgroup float2* seq_buf = buf + elem.y * n;
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
      float2 elem1 = x + complex_mul(y, plus_j);
      seq_buf[index] = complex_mul(elem1 * conj, w_k[index]);
      if (index > 0 && !last_val) {
        float2 elem2 = (x * conj) + complex_mul(y * conj, plus_j);
        seq_buf[length - index] =
            complex_mul(elem2 * conj, w_k[length - index]);
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
    const device float2* w_k) const {
  int batch_idx = elem.x * grid.y * length * 2 + elem.y * length * 2;
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
      float2 output = complex_mul(seq_buf[index] * inv_factor, w_k[index]);
      out[batch_idx + index] = output.x / length;
      out[batch_idx + index + next_out] = output.y / -length;
    }
  }
}
template <>
METAL_FUNC void
ReadWriter<float2, float2, 1, true>::load_strided(
    int stride,
    int overall_n) {
  (void)stride;
  (void)overall_n;
  bool default_inv = inv;
  inv = false;
  load();
  inv = default_inv;
}
template <>
METAL_FUNC void
ReadWriter<float2, float2, 1, true>::write_strided(
    int stride,
    int overall_n) {
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
  if (tg_idx == 0 && elem.x % outer_batch_size == 0) {
    out[strided_batch_idx + overall_n / 2] = buf[n / 2];
  }
}
template <>
METAL_FUNC void
ReadWriter<float2, float2, 0, true>::load_strided(
    int stride,
    int overall_n) {
  int overall_n_over_2 = overall_n / 2 + 1;
  auto conj = float2(1, -1);
  compute_strided_indices(stride, overall_n);
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
ReadWriter<float2, float, 1, true>::load_strided(
    int stride,
    int overall_n) {
  (void)stride;
  (void)overall_n;
  bool default_inv = inv;
  inv = false;
  load();
  inv = default_inv;
}
template <>
METAL_FUNC void
ReadWriter<float2, float, 1, true>::write_strided(
    int stride,
    int overall_n) {
  compute_strided_indices(stride, overall_n);
  for (int e = 0; e < elems_per_thread; e++) {
    out[strided_device_idx + e * stride] =
        pre_out(buf[strided_shared_idx + e], overall_n).x;
  }
}

using namespace metal;
static constant constexpr const bool inv_ [[function_constant(0)]];
static constant constexpr const bool is_power_of_2_ [[function_constant(1)]];
static constant constexpr const int elems_per_thread_ [[function_constant(2)]];
static constant constexpr const int rader_m_ [[function_constant(3)]];
static constant constexpr const int radix_13_steps_ [[function_constant(4)]];
static constant constexpr const int radix_11_steps_ [[function_constant(5)]];
static constant constexpr const int radix_8_steps_ [[function_constant(6)]];
static constant constexpr const int radix_7_steps_ [[function_constant(7)]];
static constant constexpr const int radix_6_steps_ [[function_constant(8)]];
static constant constexpr const int radix_5_steps_ [[function_constant(9)]];
static constant constexpr const int radix_4_steps_ [[function_constant(10)]];
static constant constexpr const int radix_3_steps_ [[function_constant(11)]];
static constant constexpr const int radix_2_steps_ [[function_constant(12)]];
static constant constexpr const int rader_13_steps_ [[function_constant(13)]];
static constant constexpr const int rader_11_steps_ [[function_constant(14)]];
static constant constexpr const int rader_8_steps_ [[function_constant(15)]];
static constant constexpr const int rader_7_steps_ [[function_constant(16)]];
static constant constexpr const int rader_6_steps_ [[function_constant(17)]];
static constant constexpr const int rader_5_steps_ [[function_constant(18)]];
static constant constexpr const int rader_4_steps_ [[function_constant(19)]];
static constant constexpr const int rader_3_steps_ [[function_constant(20)]];
static constant constexpr const int rader_2_steps_ [[function_constant(21)]];
typedef void (*RadixFunc)(thread float2*, thread float2*);
template <int radix, RadixFunc radix_func>
METAL_FUNC void radix_butterfly(
    int i,
    int p,
    thread float2* x,
    thread short* indices,
    thread float2* y) {
  int k, j;
  constexpr bool radix_p_2 = (radix & (radix - 1)) == 0;
  if (radix_p_2 && is_power_of_2_) {
    constexpr short power = __builtin_ctz(radix);
    k = i & (p - 1);
    j = ((i - k) << power) + k;
  } else {
    k = i % p;
    j = (i / p) * radix * p + k;
  }
  if (p > 1) {
    float2 twiddle_1 = get_twiddle(k, radix * p);
    float2 twiddle = twiddle_1;
    x[1] = complex_mul(x[1], twiddle);
#pragma clang loop unroll(full)
    for (int t = 2; t < radix; t++) {
      twiddle = complex_mul(twiddle, twiddle_1);
      x[t] = complex_mul(x[t], twiddle);
    }
  }
  radix_func(x, y);
#pragma clang loop unroll(full)
  for (int t = 0; t < radix; t++) {
    indices[t] = j + t * p;
  }
}
template <int radix, RadixFunc radix_func>
METAL_FUNC void radix_n_steps(
    int i,
    thread int* p,
    int m,
    int n,
    int num_steps,
    thread float2* inputs,
    thread short* indices,
    thread float2* values,
    threadgroup float2* buf) {
  int m_r = n / radix;
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
        radix_butterfly<radix, radix_func>(
            index, *p, inputs, indices + t * radix, values + t * radix);
      }
    }
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *p *= radix;
  }
}
template <bool rader = false>
METAL_FUNC void
perform_fft(int fft_idx, thread int* p, int m, int n, threadgroup float2* buf) {
  float2 inputs[13];
  short indices[18];
  float2 values[18];
  radix_n_steps<2, radix2>( fft_idx, p, m, n, rader ? rader_2_steps_ : radix_2_steps_, inputs, indices, values, buf);;
  radix_n_steps<3, radix3>( fft_idx, p, m, n, rader ? rader_3_steps_ : radix_3_steps_, inputs, indices, values, buf);;
  radix_n_steps<4, radix4>( fft_idx, p, m, n, rader ? rader_4_steps_ : radix_4_steps_, inputs, indices, values, buf);;
  radix_n_steps<5, radix5>( fft_idx, p, m, n, rader ? rader_5_steps_ : radix_5_steps_, inputs, indices, values, buf);;
  radix_n_steps<6, radix6>( fft_idx, p, m, n, rader ? rader_6_steps_ : radix_6_steps_, inputs, indices, values, buf);;
  radix_n_steps<7, radix7>( fft_idx, p, m, n, rader ? rader_7_steps_ : radix_7_steps_, inputs, indices, values, buf);;
  radix_n_steps<8, radix8>( fft_idx, p, m, n, rader ? rader_8_steps_ : radix_8_steps_, inputs, indices, values, buf);;
  radix_n_steps<11, radix11>( fft_idx, p, m, n, rader ? rader_11_steps_ : radix_11_steps_, inputs, indices, values, buf);;
  radix_n_steps<13, radix13>( fft_idx, p, m, n, rader ? rader_13_steps_ : radix_13_steps_, inputs, indices, values, buf);;
}
template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  threadgroup float2 shared_in[tg_mem_size];
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
  int fft_idx = elem.z;
  int m = grid.z;
  int tg_idx = elem.y * n;
  threadgroup float2* buf = &shared_in[tg_idx];
  perform_fft(fft_idx, &p, m, n, buf);
  read_writer.write();
}
template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void rader_fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    const device float2* raders_b_q [[buffer(2)]],
    const device short* raders_g_q [[buffer(3)]],
    const device short* raders_g_minus_q [[buffer(4)]],
    constant const int& n,
    constant const int& batch_size,
    constant const int& rader_n,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  threadgroup float2 shared_in[tg_mem_size];
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
  int m = grid.z;
  int fft_idx = elem.z;
  int tg_idx = elem.y * n;
  threadgroup float2* buf = &shared_in[tg_idx];
  int rader_m = rader_m_;
  short x_0_index =
      metal::min(fft_idx * elems_per_thread_ / (rader_n - 1), rader_m - 1);
  float2 x_0[2] = {buf[x_0_index], buf[x_0_index + 1]};
  float2 temp[13];
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
  int p = 1;
  perform_fft< true>(fft_idx, &p, m, n - rader_m, buf + rader_m);
  int x_sum_index = metal::min(fft_idx, rader_m - 1);
  buf[x_sum_index] = buf[rader_m + x_sum_index * (rader_n - 1)];
  float2 inv = {1.0f, -1.0f};
  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    short interleaved_index =
        index / rader_m + (index % rader_m) * (rader_n - 1);
    temp[e] = complex_mul(
        buf[rader_m + interleaved_index],
        raders_b_q[interleaved_index % (rader_n - 1)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, max_index);
    buf[rader_m + index] = temp[e] * inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  p = 1;
  perform_fft< true>(fft_idx, &p, m, n - rader_m, buf + rader_m);
  float2 rader_inv_factor = {1.0f / (rader_n - 1), -1.0f / (rader_n - 1)};
  for (int e = 0; e < elems_per_thread_; e++) {
    short index = metal::min(fft_idx * elems_per_thread_ + e, n - rader_m - 1);
    short diff_index = index / (rader_n - 1) - x_0_index;
    temp[e] = buf[rader_m + index] * rader_inv_factor + x_0[diff_index];
  }
  float2 x_sum = buf[x_0_index] + x_0[0];
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
  perform_fft(fft_idx, &p, m, n, buf);
  read_writer.write();
}
template <int tg_mem_size, typename in_T, typename out_T>
[[kernel]] void bluestein_fft(
    const device in_T* in [[buffer(0)]],
    device out_T* out [[buffer(1)]],
    const device float2* w_q [[buffer(2)]],
    const device float2* w_k [[buffer(3)]],
    constant const int& length,
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  threadgroup float2 shared_in[tg_mem_size];
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
  int fft_idx = elem.z;
  int m = grid.z;
  int tg_idx = elem.y * n;
  threadgroup float2* buf = &shared_in[tg_idx];
  perform_fft(fft_idx, &p, m, n, buf);
  float2 inv = float2(1.0f, -1.0f);
  for (int t = 0; t < elems_per_thread_; t++) {
    int index = fft_idx + t * m;
    buf[index] = complex_mul(buf[index], w_q[index]) * inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  p = 1;
  perform_fft(fft_idx, &p, m, n, buf);
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
  int overall_n = n1 * n2;
  int n = step == 0 ? n1 : n2;
  int stride = step == 0 ? n2 : n1;
  int m = grid.z;
  int fft_idx = elem.z;
  threadgroup float2 shared_in[tg_mem_size];
  threadgroup float2* buf = &shared_in[elem.y * n];
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
  perform_fft(fft_idx, &p, m, n, buf);
  read_writer.write_strided(stride, overall_n);
}
)preamble";
}

} // namespace mlx::core::metal
