// Copyright © 2026 Schtack.

import Foundation

/// TurboQuant preset requested by higher-level runtime code.
///
/// This additive Swift API deliberately routes through MLX's native packed
/// quantization primitives so callers can use one stable surface while lower
/// level PolarQuant/QJL Metal kernels evolve.
public enum TurboQuantPreset: String, Codable, Sendable, CaseIterable {
    case turbo2_5
    case turbo3_5

    public var displayName: String {
        switch self {
        case .turbo2_5:
            "TurboQuant 2.5-bit"
        case .turbo3_5:
            "TurboQuant 3.5-bit"
        }
    }

    /// Current native MLX packed-lane width used by this preset.
    ///
    /// MLX's public packed quantized matmul kernels accept integer lane widths.
    /// The 3.5-bit preset therefore uses 4-bit packed lanes until the lower
    /// level mixed 3/4-bit TurboQuant kernels are added to Cmlx/Metal.
    public var effectiveBits: Int {
        switch self {
        case .turbo2_5:
            2
        case .turbo3_5:
            4
        }
    }
}

public enum TurboQuantTensorRole: String, Codable, Sendable, CaseIterable {
    case key
    case value
    case vector
}

public struct TurboQuantConfiguration: Hashable, Codable, Sendable {
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var groupSize: Int
    public var mode: QuantizationMode

    public init(
        preset: TurboQuantPreset = .turbo3_5,
        role: TurboQuantTensorRole = .vector,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine
    ) {
        self.preset = preset
        self.role = role
        self.groupSize = groupSize
        self.mode = mode
    }

    public var effectiveBits: Int { preset.effectiveBits }
}

public typealias TurboQuantPackedTensor = (
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?
)

public func turboQuantized(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(),
    stream: StreamOrDevice = .default
) -> TurboQuantPackedTensor {
    let packed = quantized(
        array,
        groupSize: configuration.groupSize,
        bits: configuration.effectiveBits,
        mode: configuration.mode,
        stream: stream
    )
    return (packed.wq, packed.scales, packed.biases)
}

public func turboDequantized(
    _ packed: TurboQuantPackedTensor,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(),
    dtype: DType? = nil,
    stream: StreamOrDevice = .default
) -> MLXArray {
    dequantized(
        packed.weight,
        scales: packed.scales,
        biases: packed.biases,
        groupSize: configuration.groupSize,
        bits: configuration.effectiveBits,
        mode: configuration.mode,
        dtype: dtype,
        stream: stream
    )
}

public func turboQuantizedMM(
    _ x: MLXArray,
    _ packed: TurboQuantPackedTensor,
    transpose: Bool = true,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(),
    stream: StreamOrDevice = .default
) -> MLXArray {
    quantizedMM(
        x,
        packed.weight,
        scales: packed.scales,
        biases: packed.biases,
        transpose: transpose,
        groupSize: configuration.groupSize,
        bits: configuration.effectiveBits,
        mode: configuration.mode,
        stream: stream
    )
}
