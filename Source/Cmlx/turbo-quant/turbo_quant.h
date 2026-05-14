// Copyright © 2026 SharpAI
// turbo_quant.h — TurboQuant KV Cache compression for MLX
//
// Ported from TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache)
//   Primary sources:
//     ggml/src/ggml-turbo-quant.c    — CPU quantize/dequantize logic
//     ggml/src/ggml-metal/turbo-wht.h — WHT sign arrays & rotation math
//   Python validation: TheTom/turboquant_plus
//   Paper: Zandieh et al., "TurboQuant", AISTATS/ICLR 2026
//
// Algorithm summary:
//   Stage 1 (PolarQuant, 2 bits for V; 2 bits within 3-bit for K):
//     1. Compute L2 norm of the head_dim vector
//     2. Normalize to unit sphere
//     3. Apply WHT rotation: D1 * FWHT * D2  (O(d log d))
//     4. Quantize each coordinate to nearest Lloyd-Max centroid
//     5. Correct stored norm: grp_norm / recon_norm
//   Stage 2 (QJL residual, 1 bit — K cache only, for inner-product bias removal):
//     1. Reconstruct MSE approximation, compute residual
//     2. Project residual via random Gaussian matrix S
//     3. Store sign bits of S @ residual

#pragma once

#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

// ---------------------------------------------------------------------------
// Constants — must match turbo-wht.h and ggml-turbo-quant.c exactly (seed=42)
// ---------------------------------------------------------------------------

static constexpr int  TURBO_D            = 128;   // head_dim (rotation group)
static constexpr float TURBO_QJL_CONST  = 1.2533141373155003f; // sqrt(pi/2)
static constexpr int  TURBO_SEED_ROTATION = 42;
static constexpr int  TURBO_SEED_QJL     = 1042;

// 3-bit Lloyd-Max centroids for N(0, 1/128) — from ggml-turbo-quant.c
static constexpr float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// 3-bit centroid decision boundaries (midpoints between adjacent centroids)
static constexpr float BOUNDARIES_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f,  0.000000f,
     0.043589f,  0.091775f,  0.154259f
};

// WHT sign arrays — seed=42, must match turbo-wht.h exactly
static constexpr float TURBO_S1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static constexpr float TURBO_S2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

// QJL sign arrays — seed=1042, must match turbo-wht.h exactly
static constexpr float TURBO_QJL_S1[128] = {
    1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,
    1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,
    -1,-1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,1,
    1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1
};

static constexpr float TURBO_QJL_S2[128] = {
    1,1,-1,1,1,-1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,1,
    -1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,1,-1,1,1,
    -1,-1,1,-1,1,1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,
    1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,1,1,-1
};

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform (in-place, normalized by 1/sqrt(n))
// ---------------------------------------------------------------------------

static inline void turbo_fwht(float* x, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt = (n == 128) ? 0.08838834764831845f : 0.125f;
    for (int i = 0; i < n; i++) x[i] *= inv_sqrt;
}

// Forward rotation: D1 @ FWHT @ D2
static inline void turbo_rotate_forward(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] *= TURBO_S1[i];
    turbo_fwht(x, n);
    for (int i = 0; i < n; i++) x[i] *= TURBO_S2[i];
}

// Inverse rotation: D2 @ FWHT @ D1  (FWHT is self-inverse up to normalization)
static inline void turbo_rotate_inverse(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] *= TURBO_S2[i];
    turbo_fwht(x, n);
    for (int i = 0; i < n; i++) x[i] *= TURBO_S1[i];
}

// QJL rotation (different seed)
static inline void turbo_qjl_rotate(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] *= TURBO_QJL_S1[i];
    turbo_fwht(x, n);
    for (int i = 0; i < n; i++) x[i] *= TURBO_QJL_S2[i];
}

// ---------------------------------------------------------------------------
// Nearest 3-bit centroid (O(log 8) binary search on boundaries)
// ---------------------------------------------------------------------------

static inline int nearest_centroid_3bit(float v) {
    if (v < BOUNDARIES_3BIT[3]) {  // v < 0.0
        if (v < BOUNDARIES_3BIT[1]) return (v < BOUNDARIES_3BIT[0]) ? 0 : 1;
        return (v < BOUNDARIES_3BIT[2]) ? 2 : 3;
    } else {
        if (v < BOUNDARIES_3BIT[5]) return (v < BOUNDARIES_3BIT[4]) ? 4 : 5;
        return (v < BOUNDARIES_3BIT[6]) ? 6 : 7;
    }
}

// ---------------------------------------------------------------------------
// TurboQuant storage — packed bit arrays for a single head_dim=128 vector
// ---------------------------------------------------------------------------

// TURBO3: 3-bit PolarQuant (V cache — MSE optimal)
//   Storage: 48 bytes indices (3 bits × 128 = 384 bits) + 2 bytes norm (fp16)
struct TurboQuantV {
    uint8_t  indices[48];   // 3 bits per coordinate, packed
    uint16_t norm_fp16;     // corrected L2 norm as fp16
};

// TURBO4: 3-bit PolarQuant + 1-bit QJL (K cache — inner product optimal)
//   Storage: 48 bytes indices + 16 bytes QJL signs + 2 bytes norm + 2 bytes rnorm
struct TurboQuantK {
    uint8_t  indices[48];   // 3-bit PolarQuant indices, packed
    uint8_t  qjl_signs[16]; // 1-bit QJL sign per coordinate (128 bits)
    uint16_t norm_fp16;     // original L2 norm as fp16
    uint16_t rnorm_fp16;    // residual norm as fp16
};

// ---------------------------------------------------------------------------
// fp16 <-> fp32 helpers (portable, no intrinsics needed)
// ---------------------------------------------------------------------------

static inline uint16_t fp32_to_fp16(float f) {
    // Fast but portable fp32->fp16 conversion
    union { float f; uint32_t u; } v = {f};
    uint32_t u = v.u;
    uint16_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFF;
    if (exp <= 0)  return sign;
    if (exp >= 31) return sign | 0x7C00;
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { union{uint32_t u;float f;} v={sign}; return v.f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3FF; exp++;
    } else if (exp == 31) {
        union{uint32_t u;float f;} v={sign|(0xFF<<23)|mant}; return v.f;
    }
    union{uint32_t u;float f;} v={sign|((exp+127-15)<<23)|(mant<<13)};
    return v.f;
}

// ---------------------------------------------------------------------------
// Pack / unpack 3-bit indices into byte arrays
// ---------------------------------------------------------------------------

static inline void pack_3bit(const uint8_t* idx, uint8_t* packed, int d) {
    // 3 bits per element → 3 bytes per 8 elements
    for (int i = 0; i < d; i++) {
        int bit_offset = i * 3;
        int byte_idx   = bit_offset / 8;
        int bit_pos    = bit_offset % 8;
        packed[byte_idx] |= (uint8_t)((idx[i] & 0x7) << bit_pos);
        if (bit_pos > 5) {
            packed[byte_idx + 1] |= (uint8_t)((idx[i] & 0x7) >> (8 - bit_pos));
        }
    }
}

static inline void unpack_3bit(const uint8_t* packed, uint8_t* idx, int d) {
    for (int i = 0; i < d; i++) {
        int     bit_offset = i * 3;
        int     byte_idx   = bit_offset / 8;
        int     bit_pos    = bit_offset % 8;
        uint16_t raw       = (uint16_t)packed[byte_idx];
        if (byte_idx + 1 < (d * 3 + 7) / 8)
            raw |= (uint16_t)packed[byte_idx + 1] << 8;
        idx[i] = (uint8_t)((raw >> bit_pos) & 0x7);
    }
}

// ---------------------------------------------------------------------------
// Quantize one head_dim vector → TurboQuantV  (3-bit PolarQuant, V cache)
// ---------------------------------------------------------------------------

static inline TurboQuantV turbo_quantize_v(const float* src, int d) {
    TurboQuantV out;
    std::memset(&out, 0, sizeof(out));

    // 1. Compute L2 norm
    float norm_sq = 0.f;
    float buf[TURBO_D];
    for (int i = 0; i < d; i++) { buf[i] = src[i]; norm_sq += buf[i] * buf[i]; }
    float grp_norm = std::sqrt(norm_sq);
    float inv_norm = (grp_norm > 1e-10f) ? 1.f / grp_norm : 0.f;

    // 2. Normalize
    for (int i = 0; i < d; i++) buf[i] *= inv_norm;

    // 3. WHT rotation
    turbo_rotate_forward(buf, d);

    // 4. Quantize, accumulate reconstructed norm²
    uint8_t indices[TURBO_D];
    float recon_sq = 0.f;
    for (int i = 0; i < d; i++) {
        indices[i] = (uint8_t)nearest_centroid_3bit(buf[i]);
        recon_sq += CENTROIDS_3BIT[indices[i]] * CENTROIDS_3BIT[indices[i]];
    }

    // 5. Corrected norm: grp_norm / recon_norm
    float recon_norm = std::sqrt(recon_sq);
    float corrected  = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    out.norm_fp16 = fp32_to_fp16(corrected);

    // 6. Pack 3-bit indices
    pack_3bit(indices, out.indices, d);
    return out;
}

// ---------------------------------------------------------------------------
// Dequantize TurboQuantV → float vector  (CPU debug path; GPU uses Metal)
// ---------------------------------------------------------------------------

static inline void turbo_dequantize_v(const TurboQuantV& v, float* dst, int d) {
    uint8_t indices[TURBO_D];
    unpack_3bit(v.indices, indices, d);

    float norm = fp16_to_fp32(v.norm_fp16);
    float buf[TURBO_D];
    for (int i = 0; i < d; i++) buf[i] = CENTROIDS_3BIT[indices[i]];

    turbo_rotate_inverse(buf, d);
    for (int i = 0; i < d; i++) dst[i] = buf[i] * norm;
}

// ---------------------------------------------------------------------------
// Quantize one head_dim vector → TurboQuantK  (3-bit PolarQuant + 1-bit QJL)
// ---------------------------------------------------------------------------

static inline TurboQuantK turbo_quantize_k(const float* src, int d) {
    TurboQuantK out;
    std::memset(&out, 0, sizeof(out));

    // 1. Norm + normalize
    float norm_sq = 0.f;
    float normalized[TURBO_D];
    for (int i = 0; i < d; i++) { norm_sq += src[i] * src[i]; }
    float norm    = std::sqrt(norm_sq);
    float inv     = (norm > 1e-10f) ? 1.f / norm : 0.f;
    for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;

    // 2. WHT rotation
    float rotated[TURBO_D];
    std::memcpy(rotated, normalized, d * sizeof(float));
    turbo_rotate_forward(rotated, d);

    // 3. 3-bit quantization
    uint8_t indices[TURBO_D];
    for (int i = 0; i < d; i++) indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);

    // 4. Reconstruct MSE approximation → residual
    float mse_recon[TURBO_D];
    for (int i = 0; i < d; i++) mse_recon[i] = CENTROIDS_3BIT[indices[i]];
    turbo_rotate_inverse(mse_recon, d);     // back to original space

    float residual[TURBO_D];
    float rnorm_sq = 0.f;
    for (int i = 0; i < d; i++) {
        residual[i] = normalized[i] - mse_recon[i];
        rnorm_sq   += residual[i] * residual[i];
    }
    float rnorm = std::sqrt(rnorm_sq);

    // 5. QJL: WHT-based projection of residual, store sign bits
    float projected[TURBO_D];
    std::memcpy(projected, residual, d * sizeof(float));
    turbo_qjl_rotate(projected, d);

    for (int i = 0; i < d; i++) {
        if (projected[i] >= 0.f)
            out.qjl_signs[i / 8] |= (uint8_t)(1 << (i % 8));
    }

    // 6. Pack
    out.norm_fp16  = fp32_to_fp16(norm);
    out.rnorm_fp16 = fp32_to_fp16(rnorm);
    pack_3bit(indices, out.indices, d);
    return out;
}

// ---------------------------------------------------------------------------
// Dequantize TurboQuantK → float vector  (CPU debug path; GPU uses Metal)
// ---------------------------------------------------------------------------

static inline void turbo_dequantize_k(const TurboQuantK& k, float* dst, int d) {
    uint8_t indices[TURBO_D];
    unpack_3bit(k.indices, indices, d);

    float norm  = fp16_to_fp32(k.norm_fp16);
    float rnorm = fp16_to_fp32(k.rnorm_fp16);

    // Stage 1: PolarQuant reconstruction
    float mse_recon[TURBO_D];
    for (int i = 0; i < d; i++) mse_recon[i] = CENTROIDS_3BIT[indices[i]];
    turbo_rotate_inverse(mse_recon, d);

    // Stage 2: QJL reconstruction
    float signs[TURBO_D];
    for (int i = 0; i < d; i++)
        signs[i] = (k.qjl_signs[i / 8] & (1 << (i % 8))) ? 1.f : -1.f;

    // Apply inverse QJL WHT
    turbo_qjl_rotate(signs, d);   // WHT is self-inverse up to normalization
    const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;
    for (int i = 0; i < d; i++) signs[i] *= qjl_scale;

    // Combine and scale by original norm
    for (int i = 0; i < d; i++) dst[i] = (mse_recon[i] + signs[i]) * norm;
}

// ---------------------------------------------------------------------------
// MLX array-level API (used by KVCache.swift via C bridge)
// ---------------------------------------------------------------------------

/**
 * Encode a batch of KV vectors into TurboQuant format.
 *
 * keys:   [batch, num_heads, seq_len, head_dim] fp16/bf16/fp32
 * values: same shape
 *
 * Returns the compressed storage as opaque uint8 buffers.
 * The Metal attention kernel reads these directly during SDPA.
 */
struct TurboQuantKV {
    // Packed storage: each entry is one head_dim-sized compressed vector
    std::vector<TurboQuantK> k_data;  // K cache (PolarQuant + QJL)
    std::vector<TurboQuantV> v_data;  // V cache (PolarQuant only)

    int num_heads;
    int seq_len;
    int head_dim;
};

} // namespace mlx::core::fast
