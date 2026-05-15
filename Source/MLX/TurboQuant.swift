// Copyright © 2026 Schtack.

import Cmlx
import Foundation

/// TurboQuant preset requested by higher-level runtime code.
///
/// This additive Swift API gives callers one stable surface for the fast packed
/// MLX path, a deterministic PolarQuant/QJL reference codec, and the future
/// paper-exact Metal backend.
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

    public var baseMagnitudeBits: Int {
        switch self {
        case .turbo2_5:
            2
        case .turbo3_5:
            3
        }
    }

    public var highMagnitudeBits: Int {
        switch self {
        case .turbo2_5:
            3
        case .turbo3_5:
            4
        }
    }

    public var targetMagnitudeBits: Float {
        switch self {
        case .turbo2_5:
            2.5
        case .turbo3_5:
            3.5
        }
    }
}

public enum TurboQuantTensorRole: String, Codable, Sendable, CaseIterable {
    case key
    case value
    case vector
}

public enum TurboQuantBackend: String, Codable, Sendable, CaseIterable {
    /// MLX's native packed quantization and quantized matrix-multiply kernels.
    ///
    /// This is the production backend Pine uses today on iOS.
    case mlxPacked

    /// Deterministic CPU reference implementation for the mixed-bit PolarQuant
    /// layout and QJL residual sign path. It is intentionally correctness-first
    /// and exists to anchor fixtures while Metal kernels are implemented.
    case polarQJLReference

    /// Reserved for paper-exact Cmlx/Metal kernels.
    case metalPolarQJL
}

public struct TurboQuantKernelAvailability: Equatable, Codable, Sendable {
    public var supportsMLXPacked: Bool
    public var supportsPolarQJLReference: Bool
    public var supportsMetalPolarQJLCodec: Bool
    public var supportsMetalPolarQJLAttention: Bool
    public var supportsMetalPolarQJL: Bool

    public init(
        supportsMLXPacked: Bool = true,
        supportsPolarQJLReference: Bool = true,
        supportsMetalPolarQJLCodec: Bool = false,
        supportsMetalPolarQJLAttention: Bool = false,
        supportsMetalPolarQJL: Bool = false
    ) {
        self.supportsMLXPacked = supportsMLXPacked
        self.supportsPolarQJLReference = supportsPolarQJLReference
        self.supportsMetalPolarQJLCodec = supportsMetalPolarQJLCodec
        self.supportsMetalPolarQJLAttention = supportsMetalPolarQJLAttention
        self.supportsMetalPolarQJL = supportsMetalPolarQJL
    }

    public static var current: TurboQuantKernelAvailability {
        let metalAvailable = metalRuntimeAvailable()
        return TurboQuantKernelAvailability(
            supportsMetalPolarQJLCodec: metalAvailable,
            supportsMetalPolarQJLAttention: metalAvailable,
            supportsMetalPolarQJL: metalAvailable
        )
    }

    public func supports(_ backend: TurboQuantBackend) -> Bool {
        switch backend {
        case .mlxPacked:
            supportsMLXPacked
        case .polarQJLReference:
            supportsPolarQJLReference
        case .metalPolarQJL:
            supportsMetalPolarQJL
        }
    }

    public func runtimeBackend(for requestedBackend: TurboQuantBackend) -> TurboQuantBackend {
        if supports(requestedBackend) {
            requestedBackend
        } else {
            .mlxPacked
        }
    }

    public func fallbackReason(for requestedBackend: TurboQuantBackend) -> String? {
        guard !supports(requestedBackend) else { return nil }

        switch requestedBackend {
        case .mlxPacked:
            return nil
        case .polarQJLReference:
            return "PolarQuant/QJL reference backend unavailable; using MLX packed TurboQuant lanes."
        case .metalPolarQJL:
            return "Paper-exact PolarQuant/QJL Metal kernels unavailable; using MLX packed TurboQuant lanes."
        }
    }
}

public enum TurboQuantError: Error, Equatable, CustomStringConvertible {
    case invalidGroupSize(Int)
    case invalidMetalConfiguration(String)
    case invalidQualityInput(String)
    case invalidReferenceCode(String)
    case unsupportedBackend(TurboQuantBackend, String)

    public var description: String {
        switch self {
        case .invalidGroupSize(let groupSize):
            "TurboQuant group size must be positive, got \(groupSize)."
        case .invalidMetalConfiguration(let message):
            "Invalid TurboQuant Metal configuration: \(message)"
        case .invalidQualityInput(let message):
            "Invalid TurboQuant quality input: \(message)"
        case .invalidReferenceCode(let message):
            "Invalid TurboQuant reference code: \(message)"
        case .unsupportedBackend(let backend, let message):
            "Unsupported TurboQuant backend \(backend.rawValue): \(message)"
        }
    }
}

public struct TurboQuantConfiguration: Hashable, Codable, Sendable {
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var groupSize: Int
    public var mode: QuantizationMode
    public var backend: TurboQuantBackend
    public var seed: UInt64
    public var qjlResidualScale: Float

    public init(
        preset: TurboQuantPreset = .turbo3_5,
        role: TurboQuantTensorRole = .vector,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        qjlResidualScale: Float = 0.5
    ) {
        self.preset = preset
        self.role = role
        self.groupSize = groupSize
        self.mode = mode
        self.backend = backend
        self.seed = seed
        self.qjlResidualScale = qjlResidualScale
    }

    public var effectiveBits: Int { preset.effectiveBits }

    public var runtimeBackend: TurboQuantBackend {
        TurboQuantKernelAvailability.current.runtimeBackend(for: backend)
    }

    public var runtimeFallbackReason: String? {
        TurboQuantKernelAvailability.current.fallbackReason(for: backend)
    }

    public static func deterministicSeed(
        modelID: String,
        revision: String,
        cacheLayoutVersion: Int
    ) -> UInt64 {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        for byte in "\(modelID)#\(revision)#\(cacheLayoutVersion)".utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01B3
        }
        return hash == 0 ? 0x9E37_79B9_7F4A_7C15 : hash
    }
}

public typealias TurboQuantPackedTensor = (
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray?
)

public struct TurboQuantReferenceCode: Hashable, Codable, Sendable {
    public var shape: [Int]
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var groupSize: Int
    public var seed: UInt64
    public var residualScale: Float
    public var baseMagnitudeBits: Int
    public var highMagnitudeBits: Int
    public var valueCount: Int
    public var baseScales: [Float]
    public var highScales: [Float]
    public var residualScales: [Float]
    public var signs: Data
    public var highPrecisionMask: Data
    public var residualSigns: Data
    public var packedMagnitudes: Data

    private enum CodingKeys: String, CodingKey {
        case shape
        case preset
        case role
        case groupSize
        case seed
        case residualScale
        case baseMagnitudeBits
        case highMagnitudeBits
        case valueCount
        case baseScales
        case highScales
        case residualScales
        case signs
        case highPrecisionMask
        case residualSigns
        case packedMagnitudes
    }

    public init(
        shape: [Int],
        preset: TurboQuantPreset,
        role: TurboQuantTensorRole,
        groupSize: Int,
        seed: UInt64,
        residualScale: Float,
        baseMagnitudeBits: Int,
        highMagnitudeBits: Int,
        valueCount: Int,
        baseScales: [Float],
        highScales: [Float],
        residualScales: [Float]? = nil,
        signs: Data,
        highPrecisionMask: Data,
        residualSigns: Data,
        packedMagnitudes: Data
    ) {
        self.shape = shape
        self.preset = preset
        self.role = role
        self.groupSize = groupSize
        self.seed = seed
        self.residualScale = residualScale
        self.baseMagnitudeBits = baseMagnitudeBits
        self.highMagnitudeBits = highMagnitudeBits
        self.valueCount = valueCount
        self.baseScales = baseScales
        self.highScales = highScales
        self.residualScales = residualScales ?? []
        self.signs = signs
        self.highPrecisionMask = highPrecisionMask
        self.residualSigns = residualSigns
        self.packedMagnitudes = packedMagnitudes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        shape = try container.decode([Int].self, forKey: .shape)
        preset = try container.decode(TurboQuantPreset.self, forKey: .preset)
        role = try container.decode(TurboQuantTensorRole.self, forKey: .role)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        seed = try container.decode(UInt64.self, forKey: .seed)
        residualScale = try container.decodeIfPresent(Float.self, forKey: .residualScale) ?? 0.5
        baseMagnitudeBits = try container.decode(Int.self, forKey: .baseMagnitudeBits)
        highMagnitudeBits = try container.decode(Int.self, forKey: .highMagnitudeBits)
        valueCount = try container.decode(Int.self, forKey: .valueCount)
        baseScales = try container.decode([Float].self, forKey: .baseScales)
        highScales = try container.decode([Float].self, forKey: .highScales)
        residualScales = try container.decodeIfPresent([Float].self, forKey: .residualScales) ?? []
        signs = try container.decode(Data.self, forKey: .signs)
        highPrecisionMask = try container.decode(Data.self, forKey: .highPrecisionMask)
        residualSigns = try container.decode(Data.self, forKey: .residualSigns)
        packedMagnitudes = try container.decode(Data.self, forKey: .packedMagnitudes)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(shape, forKey: .shape)
        try container.encode(preset, forKey: .preset)
        try container.encode(role, forKey: .role)
        try container.encode(groupSize, forKey: .groupSize)
        try container.encode(seed, forKey: .seed)
        try container.encode(residualScale, forKey: .residualScale)
        try container.encode(baseMagnitudeBits, forKey: .baseMagnitudeBits)
        try container.encode(highMagnitudeBits, forKey: .highMagnitudeBits)
        try container.encode(valueCount, forKey: .valueCount)
        try container.encode(baseScales, forKey: .baseScales)
        try container.encode(highScales, forKey: .highScales)
        try container.encode(residualScales, forKey: .residualScales)
        try container.encode(signs, forKey: .signs)
        try container.encode(highPrecisionMask, forKey: .highPrecisionMask)
        try container.encode(residualSigns, forKey: .residualSigns)
        try container.encode(packedMagnitudes, forKey: .packedMagnitudes)
    }

    public var storageByteCount: Int {
        packedMagnitudes.count
            + signs.count
            + highPrecisionMask.count
            + residualSigns.count
            + (baseScales.count + highScales.count + residualScales.count)
                * MemoryLayout<Float>.stride
    }

    public var approximateBitsPerValue: Double {
        guard valueCount > 0 else { return 0 }
        return Double(storageByteCount * 8) / Double(valueCount)
    }
}

public struct TurboQuantMetalCode {
    public var shape: [Int]
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var groupSize: Int
    public var seed: UInt64
    public var valueCount: Int
    public var groupCount: Int
    public var magnitudeWordsPerGroup: Int
    public var bitsetWordsPerGroup: Int
    public var packedMagnitudes: MLXArray
    public var signs: MLXArray
    public var highPrecisionMask: MLXArray
    public var residualSigns: MLXArray
    public var scales: MLXArray

    public var storageByteCount: Int {
        packedMagnitudes.nbytes
            + signs.nbytes
            + highPrecisionMask.nbytes
            + residualSigns.nbytes
            + scales.nbytes
    }

    public var approximateBitsPerValue: Double {
        guard valueCount > 0 else { return 0 }
        return Double(storageByteCount * 8) / Double(valueCount)
    }
}

public enum TurboQuantAttentionPath: String, Codable, Sendable, CaseIterable {
    case onlineFused
    case twoStageCompressed
    case mlxPackedFallback
    case baseline
}

public struct TurboQuantAttentionLayout: Hashable, Codable, Sendable {
    public static let currentVersion = 2

    public var layoutVersion: Int
    public var batchSize: Int
    public var kvHeadCount: Int
    public var capacity: Int
    public var logicalLength: Int
    public var ringOffset: Int
    public var headDimension: Int
    public var groupsPerVector: Int
    public var magnitudeWordsPerGroup: Int
    public var bitsetWordsPerGroup: Int

    public init(
        layoutVersion: Int = TurboQuantAttentionLayout.currentVersion,
        batchSize: Int,
        kvHeadCount: Int,
        capacity: Int,
        logicalLength: Int,
        ringOffset: Int = 0,
        headDimension: Int,
        groupsPerVector: Int,
        magnitudeWordsPerGroup: Int,
        bitsetWordsPerGroup: Int
    ) {
        self.layoutVersion = layoutVersion
        self.batchSize = batchSize
        self.kvHeadCount = kvHeadCount
        self.capacity = capacity
        self.logicalLength = logicalLength
        self.ringOffset = ringOffset
        self.headDimension = headDimension
        self.groupsPerVector = groupsPerVector
        self.magnitudeWordsPerGroup = magnitudeWordsPerGroup
        self.bitsetWordsPerGroup = bitsetWordsPerGroup
    }

    public var logicalShape: [Int] {
        [batchSize, kvHeadCount, logicalLength, headDimension]
    }

    public var storageShape: [Int] {
        [batchSize, kvHeadCount, capacity, headDimension]
    }
}

public struct TurboQuantAttentionCode {
    public var layout: TurboQuantAttentionLayout
    public var preset: TurboQuantPreset
    public var role: TurboQuantTensorRole
    public var groupSize: Int
    public var seed: UInt64
    public var packedMagnitudes: MLXArray
    public var signs: MLXArray
    public var highPrecisionMask: MLXArray
    public var residualSigns: MLXArray
    public var scales: MLXArray

    public init(
        layout: TurboQuantAttentionLayout,
        preset: TurboQuantPreset,
        role: TurboQuantTensorRole,
        groupSize: Int,
        seed: UInt64,
        packedMagnitudes: MLXArray,
        signs: MLXArray,
        highPrecisionMask: MLXArray,
        residualSigns: MLXArray,
        scales: MLXArray
    ) {
        self.layout = layout
        self.preset = preset
        self.role = role
        self.groupSize = groupSize
        self.seed = seed
        self.packedMagnitudes = packedMagnitudes
        self.signs = signs
        self.highPrecisionMask = highPrecisionMask
        self.residualSigns = residualSigns
        self.scales = scales
    }

    public var storageByteCount: Int {
        packedMagnitudes.nbytes
            + signs.nbytes
            + highPrecisionMask.nbytes
            + residualSigns.nbytes
            + scales.nbytes
    }

    public var approximateBitsPerValue: Double {
        let values = layout.batchSize * layout.kvHeadCount
            * Swift.max(layout.logicalLength, 1) * layout.headDimension
        return Double(storageByteCount * 8) / Double(values)
    }
}

public struct TurboQuantQualityThresholds: Hashable, Codable, Sendable {
    public var maxRelativeMSE: Float
    public var minCosineSimilarity: Float
    public var maxInnerProductRelativeError: Float

    public init(
        maxRelativeMSE: Float = 0.02,
        minCosineSimilarity: Float = 0.99,
        maxInnerProductRelativeError: Float = 0.08
    ) {
        self.maxRelativeMSE = maxRelativeMSE
        self.minCosineSimilarity = minCosineSimilarity
        self.maxInnerProductRelativeError = maxInnerProductRelativeError
    }
}

public struct TurboQuantQualityReport: Hashable, Codable, Sendable {
    public var mse: Float
    public var relativeMSE: Float
    public var maxAbsoluteError: Float
    public var cosineSimilarity: Float
    public var innerProductRelativeError: Float
    public var thresholds: TurboQuantQualityThresholds

    public var passes: Bool {
        relativeMSE <= thresholds.maxRelativeMSE
            && cosineSimilarity >= thresholds.minCosineSimilarity
            && innerProductRelativeError <= thresholds.maxInnerProductRelativeError
    }
}

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

public func turboQuantReferenceEncode(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(
        backend: .polarQJLReference
    )
) throws -> TurboQuantReferenceCode {
    guard configuration.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(configuration.groupSize)
    }

    let values = array.asArray(Float.self)
    return try encodeTurboQuantReference(values: values, shape: array.shape, configuration: configuration)
}

public func turboQuantReferenceDecode(
    _ code: TurboQuantReferenceCode
) throws -> MLXArray {
    let values = try decodeTurboQuantReference(code)
    return MLXArray(values, code.shape)
}

public func turboQuantReferenceQuality(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(
        backend: .polarQJLReference
    ),
    thresholds: TurboQuantQualityThresholds = TurboQuantQualityThresholds()
) throws -> TurboQuantQualityReport {
    let original = array.asArray(Float.self)
    let code = try turboQuantReferenceEncode(array, configuration: configuration)
    let decoded = try turboQuantReferenceDecode(code).asArray(Float.self)
    return try turboQuantQuality(
        original: original,
        decoded: decoded,
        seed: configuration.seed,
        thresholds: thresholds
    )
}

public func turboQuantMetalEncode(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(backend: .metalPolarQJL),
    stream: StreamOrDevice = .default
) throws -> TurboQuantMetalCode {
    try validateMetalConfiguration(array: array, configuration: configuration)

    let valueCount = array.size
    let groupSize = configuration.groupSize
    let groupCount = (valueCount + groupSize - 1) / groupSize
    let magnitudeWordsPerGroup = metalMagnitudeWordsPerGroup(
        groupSize: groupSize,
        preset: configuration.preset
    )
    let bitsetWordsPerGroup = (groupSize + 31) / 32
    let threadGroupSize = Swift.max(1, Swift.min(groupCount, 64))

    let outputs = TurboQuantMetalKernels.encode(
        [array],
        template: metalTemplate(
            configuration: configuration,
            valueCount: valueCount,
            groupCount: groupCount,
            magnitudeWordsPerGroup: magnitudeWordsPerGroup,
            bitsetWordsPerGroup: bitsetWordsPerGroup
        ),
        grid: (groupCount, 1, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [
            [groupCount * magnitudeWordsPerGroup],
            [groupCount * bitsetWordsPerGroup],
            [groupCount * bitsetWordsPerGroup],
            [groupCount * bitsetWordsPerGroup],
            [groupCount, 3],
        ],
        outputDTypes: [.uint32, .uint32, .uint32, .uint32, .float32],
        initValue: 0,
        stream: stream
    )

    return TurboQuantMetalCode(
        shape: array.shape,
        preset: configuration.preset,
        role: configuration.role,
        groupSize: groupSize,
        seed: configuration.seed,
        valueCount: valueCount,
        groupCount: groupCount,
        magnitudeWordsPerGroup: magnitudeWordsPerGroup,
        bitsetWordsPerGroup: bitsetWordsPerGroup,
        packedMagnitudes: outputs[0],
        signs: outputs[1],
        highPrecisionMask: outputs[2],
        residualSigns: outputs[3],
        scales: outputs[4]
    )
}

public func turboQuantMetalDecode(
    _ code: TurboQuantMetalCode,
    dtype: DType = .float32,
    stream: StreamOrDevice = .default
) throws -> MLXArray {
    guard code.valueCount > 0 else {
        throw TurboQuantError.invalidMetalConfiguration("empty arrays are not supported")
    }
    guard code.groupSize > 0, code.groupSize <= 128, code.groupSize % 32 == 0 else {
        throw TurboQuantError.invalidGroupSize(code.groupSize)
    }
    guard dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration("decode output dtype must be floating point")
    }

    let threadGroupSize = Swift.max(1, Swift.min(code.valueCount, 256))
    let configuration = TurboQuantConfiguration(
        preset: code.preset,
        role: code.role,
        groupSize: code.groupSize,
        backend: .metalPolarQJL,
        seed: code.seed
    )
    let outputs = TurboQuantMetalKernels.decode(
        [
            code.packedMagnitudes,
            code.signs,
            code.highPrecisionMask,
            code.residualSigns,
            code.scales,
        ],
        template: metalTemplate(
            configuration: configuration,
            valueCount: code.valueCount,
            groupCount: code.groupCount,
            magnitudeWordsPerGroup: code.magnitudeWordsPerGroup,
            bitsetWordsPerGroup: code.bitsetWordsPerGroup
        ),
        grid: (code.valueCount, 1, 1),
        threadGroup: (threadGroupSize, 1, 1),
        outputShapes: [code.shape],
        outputDTypes: [dtype],
        stream: stream
    )

    return outputs[0]
}

public func turboQuantEmptyAttentionCode(
    layout: TurboQuantAttentionLayout,
    preset: TurboQuantPreset = .turbo3_5,
    role: TurboQuantTensorRole,
    groupSize: Int = 64,
    seed: UInt64 = 0x9E37_79B9_7F4A_7C15
) throws -> TurboQuantAttentionCode {
    try validateAttentionLayout(layout, role: role, groupSize: groupSize)
    return TurboQuantAttentionCode(
        layout: layout,
        preset: preset,
        role: role,
        groupSize: groupSize,
        seed: seed,
        packedMagnitudes: MLXArray.zeros(
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.magnitudeWordsPerGroup,
            ],
            dtype: .uint32
        ),
        signs: MLXArray.zeros(
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.bitsetWordsPerGroup,
            ],
            dtype: .uint32
        ),
        highPrecisionMask: MLXArray.zeros(
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.bitsetWordsPerGroup,
            ],
            dtype: .uint32
        ),
        residualSigns: MLXArray.zeros(
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.bitsetWordsPerGroup,
            ],
            dtype: .uint32
        ),
        scales: MLXArray.zeros(
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, 3,
            ],
            dtype: .float32
        )
    )
}

public func turboQuantAttentionLayout(
    for array: MLXArray,
    preset: TurboQuantPreset = .turbo3_5,
    groupSize: Int = 64,
    capacity: Int? = nil,
    logicalLength: Int? = nil,
    ringOffset: Int = 0
) throws -> TurboQuantAttentionLayout {
    try validateAttentionArray(array, groupSize: groupSize)
    let headDimension = array.dim(3)
    let groupsPerVector = (headDimension + groupSize - 1) / groupSize
    let resolvedCapacity = capacity ?? array.dim(2)
    let resolvedLogicalLength = logicalLength ?? array.dim(2)
    let layout = TurboQuantAttentionLayout(
        batchSize: array.dim(0),
        kvHeadCount: array.dim(1),
        capacity: resolvedCapacity,
        logicalLength: resolvedLogicalLength,
        ringOffset: ringOffset,
        headDimension: headDimension,
        groupsPerVector: groupsPerVector,
        magnitudeWordsPerGroup: metalMagnitudeWordsPerGroup(groupSize: groupSize, preset: preset),
        bitsetWordsPerGroup: (groupSize + 31) / 32
    )
    try validateAttentionLayout(layout, role: .key, groupSize: groupSize)
    return layout
}

public func turboQuantMetalEncodeAttention(
    _ array: MLXArray,
    configuration: TurboQuantConfiguration = TurboQuantConfiguration(
        role: .key,
        backend: .metalPolarQJL
    ),
    capacity: Int? = nil,
    logicalLength: Int? = nil,
    ringOffset: Int = 0,
    stream: StreamOrDevice = .default
) throws -> TurboQuantAttentionCode {
    try validateAttentionArray(array, groupSize: configuration.groupSize)
    try requireTurboQuantMetalAttention()

    let layout = try turboQuantAttentionLayout(
        for: array,
        preset: configuration.preset,
        groupSize: configuration.groupSize,
        capacity: capacity,
        logicalLength: logicalLength,
        ringOffset: ringOffset
    )
    guard layout.logicalLength <= layout.capacity else {
        throw TurboQuantError.invalidMetalConfiguration(
            "logical length cannot exceed compressed attention capacity"
        )
    }

    let rowGroupCount = layout.batchSize * layout.kvHeadCount
        * array.dim(2) * layout.groupsPerVector
    let outputs = TurboQuantMetalKernels.encodeAttention(
        [array],
        template: attentionTemplate(
            configuration: configuration,
            layout: layout,
            inputLength: array.dim(2),
            outputLength: array.dim(2),
            queryHeadCount: 0,
            queryLength: 0,
            outputDType: .float32,
            causal: false
        ),
        grid: (rowGroupCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(rowGroupCount, 256)), 1, 1),
        outputShapes: [
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.magnitudeWordsPerGroup,
            ],
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.bitsetWordsPerGroup,
            ],
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.bitsetWordsPerGroup,
            ],
            [
                layout.batchSize, layout.kvHeadCount, layout.capacity,
                layout.groupsPerVector, layout.bitsetWordsPerGroup,
            ],
            [layout.batchSize, layout.kvHeadCount, layout.capacity, layout.groupsPerVector, 3],
        ],
        outputDTypes: [.uint32, .uint32, .uint32, .uint32, .float32],
        initValue: 0,
        stream: stream
    )

    return TurboQuantAttentionCode(
        layout: layout,
        preset: configuration.preset,
        role: configuration.role,
        groupSize: configuration.groupSize,
        seed: configuration.seed,
        packedMagnitudes: outputs[0],
        signs: outputs[1],
        highPrecisionMask: outputs[2],
        residualSigns: outputs[3],
        scales: outputs[4]
    )
}

public func turboQuantMetalQK(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    stream: StreamOrDevice = .default
) throws -> MLXArray {
    try validateAttentionQuery(queries, code: keyCode)
    try requireTurboQuantMetalAttention()
    guard keyCode.role == .key else {
        throw TurboQuantError.invalidMetalConfiguration("QK requires a key code")
    }

    let outputShape = [
        queries.dim(0), queries.dim(1), queries.dim(2), keyCode.layout.logicalLength,
    ]
    let elementCount = outputShape.reduce(1, *)
    var scores = TurboQuantMetalKernels.qk(
        [
            queries,
            keyCode.packedMagnitudes,
            keyCode.signs,
            keyCode.highPrecisionMask,
            keyCode.residualSigns,
            keyCode.scales,
        ],
        template: attentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: keyCode.preset,
                role: keyCode.role,
                groupSize: keyCode.groupSize,
                backend: .metalPolarQJL,
                seed: keyCode.seed
            ),
            layout: keyCode.layout,
            inputLength: keyCode.layout.logicalLength,
            outputLength: keyCode.layout.logicalLength,
            queryHeadCount: queries.dim(1),
            queryLength: queries.dim(2),
            outputDType: .float32,
            causal: false
        ) + [("ATTENTION_SCALE_BITS", Int(scale.bitPattern))],
        grid: (elementCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(elementCount, 256)), 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [.float32],
        stream: stream
    )[0]

    applyAttentionMask(&scores, mask: mask, stream: stream)
    return scores
}

public func turboQuantMetalAV(
    attentionWeights: MLXArray,
    valueCode: TurboQuantAttentionCode,
    outputDType: DType = .float32,
    stream: StreamOrDevice = .default
) throws -> MLXArray {
    try requireTurboQuantMetalAttention()
    guard valueCode.role == .value else {
        throw TurboQuantError.invalidMetalConfiguration("AV requires a value code")
    }
    guard attentionWeights.ndim == 4 else {
        throw TurboQuantError.invalidMetalConfiguration("attention weights must be [B, Hq, L, T]")
    }
    guard attentionWeights.dim(0) == valueCode.layout.batchSize,
        attentionWeights.dim(3) == valueCode.layout.logicalLength
    else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention weights do not match the compressed value layout"
        )
    }
    guard attentionWeights.dim(1) % valueCode.layout.kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads"
        )
    }

    let outputShape = [
        attentionWeights.dim(0), attentionWeights.dim(1), attentionWeights.dim(2),
        valueCode.layout.headDimension,
    ]
    let elementCount = outputShape.reduce(1, *)
    return TurboQuantMetalKernels.av(
        [
            attentionWeights,
            valueCode.packedMagnitudes,
            valueCode.signs,
            valueCode.highPrecisionMask,
            valueCode.residualSigns,
            valueCode.scales,
        ],
        template: attentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: valueCode.preset,
                role: valueCode.role,
                groupSize: valueCode.groupSize,
                backend: .metalPolarQJL,
                seed: valueCode.seed
            ),
            layout: valueCode.layout,
            inputLength: valueCode.layout.logicalLength,
            outputLength: valueCode.layout.logicalLength,
            queryHeadCount: attentionWeights.dim(1),
            queryLength: attentionWeights.dim(2),
            outputDType: outputDType,
            causal: false
        ),
        grid: (elementCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(elementCount, 256)), 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

public func turboQuantMetalScaledDotProductAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    preferOnlineFused: Bool = true,
    stream: StreamOrDevice = .default
) throws -> MLXArray {
    try validateAttentionPair(keyCode: keyCode, valueCode: valueCode)
    try validateAttentionQuery(queries, code: keyCode)
    try requireTurboQuantMetalAttention()

    if preferOnlineFused,
        turboQuantMetalSupportsOnlineFusedAttention(queries: queries, keyCode: keyCode, mask: mask)
    {
        return try turboQuantMetalOnlineFusedAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: mask,
            outputDType: queries.dtype,
            stream: stream
        )
    }

    let scores = try turboQuantMetalQK(
        queries: queries,
        keyCode: keyCode,
        scale: scale,
        mask: mask,
        stream: stream
    )
    let weights = softmax(scores.asType(.float32), axis: -1, stream: stream)
    return try turboQuantMetalAV(
        attentionWeights: weights,
        valueCode: valueCode,
        outputDType: queries.dtype,
        stream: stream
    )
}

public func turboQuantMetalSupportsOnlineFusedAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> Bool {
    guard queries.ndim == 4 else { return false }
    guard queries.dim(0) == 1, queries.dim(2) <= 8 else { return false }
    guard [64, 80, 96, 128, 256].contains(queries.dim(3)) else { return false }
    guard queries.dim(3) == keyCode.layout.headDimension else { return false }
    switch mask {
    case .none, .causal:
        return true
    case .array, .arrays:
        return false
    }
}

private func turboQuantMetalOnlineFusedAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    outputDType: DType,
    stream: StreamOrDevice
) throws -> MLXArray {
    let outputShape = [queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)]
    let rowCount = queries.dim(0) * queries.dim(1) * queries.dim(2)
    let causal: Bool
    switch mask {
    case .causal:
        causal = true
    case .none:
        causal = false
    case .array, .arrays:
        throw TurboQuantError.invalidMetalConfiguration(
            "online fused TurboQuant attention does not support materialized masks"
        )
    }

    return TurboQuantMetalKernels.fusedAttention(
        [
            queries,
            keyCode.packedMagnitudes,
            keyCode.signs,
            keyCode.highPrecisionMask,
            keyCode.residualSigns,
            keyCode.scales,
            valueCode.packedMagnitudes,
            valueCode.signs,
            valueCode.highPrecisionMask,
            valueCode.residualSigns,
            valueCode.scales,
        ],
        template: attentionTemplate(
            configuration: TurboQuantConfiguration(
                preset: keyCode.preset,
                role: .key,
                groupSize: keyCode.groupSize,
                backend: .metalPolarQJL,
                seed: keyCode.seed
            ),
            layout: keyCode.layout,
            inputLength: keyCode.layout.logicalLength,
            outputLength: keyCode.layout.logicalLength,
            queryHeadCount: queries.dim(1),
            queryLength: queries.dim(2),
            outputDType: outputDType,
            causal: causal
        ) + [
            ("VALUE_SEED", Int(UInt32(truncatingIfNeeded: valueCode.seed))),
            ("ATTENTION_SCALE_BITS", Int(scale.bitPattern)),
        ],
        grid: (rowCount, 1, 1),
        threadGroup: (Swift.max(1, Swift.min(rowCount, 256)), 1, 1),
        outputShapes: [outputShape],
        outputDTypes: [outputDType],
        stream: stream
    )[0]
}

public func requireTurboQuantBackend(_ backend: TurboQuantBackend) throws {
    let availability = TurboQuantKernelAvailability.current
    guard availability.supports(backend) else {
        throw TurboQuantError.unsupportedBackend(
            backend,
            availability.fallbackReason(for: backend) ?? "Backend unavailable."
        )
    }
}

public func requireTurboQuantMetalAttention() throws {
    guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarQJL,
            "Metal runtime is unavailable for PolarQuant/QJL compressed attention."
        )
    }
}

public func requireTurboQuantMetalCodec() throws {
    guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
        throw TurboQuantError.unsupportedBackend(
            .metalPolarQJL,
            "Metal runtime is unavailable for the PolarQuant/QJL codec."
        )
    }
}

private func encodeTurboQuantReference(
    values: [Float],
    shape: [Int],
    configuration: TurboQuantConfiguration
) throws -> TurboQuantReferenceCode {
    let expectedCount = shape.reduce(1, *)
    guard expectedCount == values.count else {
        throw TurboQuantError.invalidReferenceCode(
            "shape \(shape) contains \(expectedCount) values but input has \(values.count)"
        )
    }

    let groupSize = configuration.groupSize
    let baseBits = configuration.preset.baseMagnitudeBits
    let highBits = configuration.preset.highMagnitudeBits
    let groupCount = (values.count + groupSize - 1) / groupSize
    var baseScales = Array(repeating: Float(1), count: groupCount)
    var highScales = Array(repeating: Float(1), count: groupCount)
    var residualScales = Array(repeating: Float(0), count: groupCount)
    var signs = [UInt8](repeating: 0, count: packedBitByteCount(values.count))
    var highPrecisionMask = [UInt8](repeating: 0, count: packedBitByteCount(values.count))
    var residualSigns = [UInt8](repeating: 0, count: packedBitByteCount(values.count))
    var magnitudes = [UInt8]()
    var magnitudeBitOffset = 0

    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * groupSize
        let end = Swift.min(start + groupSize, values.count)
        let count = end - start
        guard count > 0 else { continue }

        var transformed = Array(repeating: Float(0), count: count)
        var maxAbs = Float(0)
        for localIndex in 0 ..< count {
            let absoluteIndex = start + localIndex
            let value = preconditionedValue(
                values[absoluteIndex],
                index: absoluteIndex,
                seed: configuration.seed
            )
            transformed[localIndex] = value
            maxAbs = Swift.max(maxAbs, Swift.abs(value))
        }

        let baseMax = Float((1 << baseBits) - 1)
        let highMax = Float((1 << highBits) - 1)
        let safeMaxAbs = Swift.max(maxAbs, Float.leastNonzeroMagnitude)
        baseScales[groupIndex] = safeMaxAbs / baseMax
        highScales[groupIndex] = safeMaxAbs / highMax

        let highPrecisionCount = mixedPrecisionHighCount(
            valueCount: count,
            baseBits: baseBits,
            highBits: highBits,
            targetBits: configuration.preset.targetMagnitudeBits
        )
        var highPrecisionIndices = Set<Int>()
        if highPrecisionCount > 0 {
            let ranked = transformed.indices.sorted { lhs, rhs in
                let leftMagnitude = Swift.abs(transformed[lhs])
                let rightMagnitude = Swift.abs(transformed[rhs])
                if leftMagnitude == rightMagnitude {
                    return lhs < rhs
                }
                return leftMagnitude > rightMagnitude
            }
            highPrecisionIndices = Set(ranked.prefix(highPrecisionCount))
        }

        var residuals = Array(repeating: Float(0), count: count)
        var residualMagnitudeSum = Float(0)
        for localIndex in 0 ..< count {
            let value = transformed[localIndex]
            let highPrecision = highPrecisionIndices.contains(localIndex)
            let bits = highPrecision ? highBits : baseBits
            let scale = highPrecision ? highScales[groupIndex] : baseScales[groupIndex]
            let levelMax = Float((1 << bits) - 1)
            let magnitude = Swift.abs(value)
            let quantizedMagnitude = UInt8(
                Swift.max(0, Swift.min(Int((magnitude / scale).rounded()), Int(levelMax)))
            )
            let signedDecoded = (value.sign == .minus ? -1 : 1) * Float(quantizedMagnitude) * scale
            let residual = value - signedDecoded
            residuals[localIndex] = residual
            residualMagnitudeSum += Swift.abs(residual)
        }
        if configuration.role != .value {
            residualScales[groupIndex] = residualMagnitudeSum / Float(count)
        }

        for localIndex in 0 ..< count {
            let absoluteIndex = start + localIndex
            let value = transformed[localIndex]
            let highPrecision = highPrecisionIndices.contains(localIndex)
            let bits = highPrecision ? highBits : baseBits
            let scale = highPrecision ? highScales[groupIndex] : baseScales[groupIndex]
            let levelMax = Float((1 << bits) - 1)
            let magnitude = Swift.abs(value)
            let quantizedMagnitude = UInt8(
                Swift.max(0, Swift.min(Int((magnitude / scale).rounded()), Int(levelMax)))
            )
            setPackedBit(&signs, index: absoluteIndex, value: value.sign == .minus)
            setPackedBit(&highPrecisionMask, index: absoluteIndex, value: highPrecision)
            if configuration.role != .value {
                setPackedBit(&residualSigns, index: absoluteIndex, value: residuals[localIndex].sign == .minus)
            }
            appendPackedBits(
                UInt32(quantizedMagnitude),
                bitCount: bits,
                bytes: &magnitudes,
                bitOffset: &magnitudeBitOffset
            )
        }
    }

    if configuration.role == .value {
        residualSigns.removeAll(keepingCapacity: false)
    }

    return TurboQuantReferenceCode(
        shape: shape,
        preset: configuration.preset,
        role: configuration.role,
        groupSize: groupSize,
        seed: configuration.seed,
        residualScale: configuration.qjlResidualScale,
        baseMagnitudeBits: baseBits,
        highMagnitudeBits: highBits,
        valueCount: values.count,
        baseScales: baseScales,
        highScales: highScales,
        residualScales: residualScales,
        signs: Data(signs),
        highPrecisionMask: Data(highPrecisionMask),
        residualSigns: Data(residualSigns),
        packedMagnitudes: Data(magnitudes)
    )
}

private func decodeTurboQuantReference(_ code: TurboQuantReferenceCode) throws -> [Float] {
    guard code.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(code.groupSize)
    }
    guard code.shape.reduce(1, *) == code.valueCount else {
        throw TurboQuantError.invalidReferenceCode(
            "shape \(code.shape) does not match value count \(code.valueCount)"
        )
    }

    let groupCount = (code.valueCount + code.groupSize - 1) / code.groupSize
    guard code.baseScales.count == groupCount, code.highScales.count == groupCount else {
        throw TurboQuantError.invalidReferenceCode("scale table count does not match groups")
    }
    guard code.residualScales.isEmpty || code.residualScales.count == groupCount else {
        throw TurboQuantError.invalidReferenceCode("residual scale table count does not match groups")
    }
    guard code.signs.count >= packedBitByteCount(code.valueCount),
        code.highPrecisionMask.count >= packedBitByteCount(code.valueCount)
    else {
        throw TurboQuantError.invalidReferenceCode("bitset storage is truncated")
    }
    if code.role != .value && code.residualSigns.count < packedBitByteCount(code.valueCount) {
        throw TurboQuantError.invalidReferenceCode("residual sign storage is truncated")
    }

    var values = Array(repeating: Float(0), count: code.valueCount)
    var magnitudeBitOffset = 0

    for groupIndex in 0 ..< groupCount {
        let start = groupIndex * code.groupSize
        let end = Swift.min(start + code.groupSize, code.valueCount)
        for absoluteIndex in start ..< end {
            let highPrecision = getPackedBit(code.highPrecisionMask, index: absoluteIndex)
            let bits = highPrecision ? code.highMagnitudeBits : code.baseMagnitudeBits
            let scale = highPrecision ? code.highScales[groupIndex] : code.baseScales[groupIndex]
            let magnitude = Float(
                try readPackedBits(
                    code.packedMagnitudes,
                    bitOffset: &magnitudeBitOffset,
                    bitCount: bits
                )
            )
            let sign: Float = getPackedBit(code.signs, index: absoluteIndex) ? -1 : 1
            var reconstructed = sign * magnitude * scale

            if code.role != .value {
                let residualSign: Float =
                    getPackedBit(code.residualSigns, index: absoluteIndex) ? -1 : 1
                let residualScale = code.residualScales.isEmpty
                    ? code.residualScale * scale
                    : code.residualScales[groupIndex]
                reconstructed += residualSign * residualScale
            }

            values[absoluteIndex] = unpreconditionedValue(
                reconstructed,
                index: absoluteIndex,
                seed: code.seed
            )
        }
    }

    return values
}

private func turboQuantQuality(
    original: [Float],
    decoded: [Float],
    seed: UInt64,
    thresholds: TurboQuantQualityThresholds
) throws -> TurboQuantQualityReport {
    guard !original.isEmpty else {
        throw TurboQuantError.invalidQualityInput("quality input must not be empty")
    }
    guard original.count == decoded.count else {
        throw TurboQuantError.invalidQualityInput("original and decoded counts differ")
    }

    var squaredError = Float(0)
    var squaredSignal = Float(0)
    var maxAbsoluteError = Float(0)
    var dot = Float(0)
    var originalNormSquared = Float(0)
    var decodedNormSquared = Float(0)
    var probeOriginalDot = Float(0)
    var probeDecodedDot = Float(0)

    for index in original.indices {
        let lhs = original[index]
        let rhs = decoded[index]
        let delta = lhs - rhs
        squaredError += delta * delta
        squaredSignal += lhs * lhs
        maxAbsoluteError = Swift.max(maxAbsoluteError, Swift.abs(delta))
        dot += lhs * rhs
        originalNormSquared += lhs * lhs
        decodedNormSquared += rhs * rhs

        let probe = deterministicProbeValue(index: index, seed: seed)
        probeOriginalDot += probe * lhs
        probeDecodedDot += probe * rhs
    }

    let count = Float(original.count)
    let mse = squaredError / count
    let relativeMSE = squaredError / Swift.max(squaredSignal, Float.leastNonzeroMagnitude)
    let cosineDenominator = sqrt(originalNormSquared) * sqrt(decodedNormSquared)
    let cosineSimilarity = dot / Swift.max(cosineDenominator, Float.leastNonzeroMagnitude)
    let innerProductRelativeError = Swift.abs(probeOriginalDot - probeDecodedDot)
        / Swift.max(Swift.abs(probeOriginalDot), Float.leastNonzeroMagnitude)

    return TurboQuantQualityReport(
        mse: mse,
        relativeMSE: relativeMSE,
        maxAbsoluteError: maxAbsoluteError,
        cosineSimilarity: cosineSimilarity,
        innerProductRelativeError: innerProductRelativeError,
        thresholds: thresholds
    )
}

private func deterministicProbeValue(index: Int, seed: UInt64) -> Float {
    var state = seed ^ 0xD1B5_4A32_D192_ED03
    state &+= UInt64(index) &* 0x9E37_79B9_7F4A_7C15
    state ^= state >> 30
    state &*= 0xBF58_476D_1CE4_E5B9
    state ^= state >> 27
    state &*= 0x94D0_49BB_1331_11EB
    state ^= state >> 31
    let unit = Float(UInt32(truncatingIfNeeded: state)) / Float(UInt32.max)
    return unit * 2 - 1
}

private func mixedPrecisionHighCount(
    valueCount: Int,
    baseBits: Int,
    highBits: Int,
    targetBits: Float
) -> Int {
    guard highBits > baseBits else { return 0 }
    let fraction = (targetBits - Float(baseBits)) / Float(highBits - baseBits)
    let clampedFraction = Swift.max(0, Swift.min(1, fraction))
    return Int((Float(valueCount) * clampedFraction).rounded())
}

private func packedBitByteCount(_ bitCount: Int) -> Int {
    (bitCount + 7) / 8
}

private func setPackedBit(_ bytes: inout [UInt8], index: Int, value: Bool) {
    guard value else { return }
    let byteIndex = index / 8
    let bitIndex = index % 8
    bytes[byteIndex] |= UInt8(1 << bitIndex)
}

private func getPackedBit(_ data: Data, index: Int) -> Bool {
    let byteIndex = index / 8
    let bitIndex = index % 8
    guard byteIndex < data.count else { return false }
    return (data[byteIndex] & UInt8(1 << bitIndex)) != 0
}

private func appendPackedBits(
    _ value: UInt32,
    bitCount: Int,
    bytes: inout [UInt8],
    bitOffset: inout Int
) {
    for localBit in 0 ..< bitCount {
        if bitOffset / 8 == bytes.count {
            bytes.append(0)
        }
        let bitSet = (value & (1 << UInt32(localBit))) != 0
        if bitSet {
            bytes[bitOffset / 8] |= UInt8(1 << (bitOffset % 8))
        }
        bitOffset += 1
    }
}

private func readPackedBits(
    _ data: Data,
    bitOffset: inout Int,
    bitCount: Int
) throws -> UInt32 {
    var value: UInt32 = 0
    for localBit in 0 ..< bitCount {
        let byteIndex = bitOffset / 8
        guard byteIndex < data.count else {
            throw TurboQuantError.invalidReferenceCode("packed magnitude storage is truncated")
        }
        if (data[byteIndex] & UInt8(1 << (bitOffset % 8))) != 0 {
            value |= 1 << UInt32(localBit)
        }
        bitOffset += 1
    }
    return value
}

private func preconditionedValue(_ value: Float, index: Int, seed: UInt64) -> Float {
    randomSign(index: index, seed: seed) ? -value : value
}

private func unpreconditionedValue(_ value: Float, index: Int, seed: UInt64) -> Float {
    randomSign(index: index, seed: seed) ? -value : value
}

private func randomSign(index: Int, seed: UInt64) -> Bool {
    var state = seed &+ UInt64(index) &* 0x9E37_79B9_7F4A_7C15
    state ^= state >> 30
    state &*= 0xBF58_476D_1CE4_E5B9
    state ^= state >> 27
    state &*= 0x94D0_49BB_1331_11EB
    state ^= state >> 31
    return (state & 1) == 1
}

private func metalRuntimeAvailable() -> Bool {
    var result = false
    return mlx_metal_is_available(&result) == 0 && result
}

private func validateMetalConfiguration(
    array: MLXArray,
    configuration: TurboQuantConfiguration
) throws {
    guard array.size > 0 else {
        throw TurboQuantError.invalidMetalConfiguration("empty arrays are not supported")
    }
    guard array.dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration("input dtype must be floating point")
    }
    guard configuration.groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(configuration.groupSize)
    }
    guard configuration.groupSize <= 128, configuration.groupSize % 32 == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "group size must be 32, 64, 96, or 128 for the Metal codec"
        )
    }
    try requireTurboQuantMetalCodec()
}

private func metalMagnitudeWordsPerGroup(
    groupSize: Int,
    preset: TurboQuantPreset
) -> Int {
    let highCount = mixedPrecisionHighCount(
        valueCount: groupSize,
        baseBits: preset.baseMagnitudeBits,
        highBits: preset.highMagnitudeBits,
        targetBits: preset.targetMagnitudeBits
    )
    let bitCount = groupSize * preset.baseMagnitudeBits
        + highCount * (preset.highMagnitudeBits - preset.baseMagnitudeBits)
    return (bitCount + 31) / 32
}

private func metalTemplate(
    configuration: TurboQuantConfiguration,
    valueCount: Int,
    groupCount: Int,
    magnitudeWordsPerGroup: Int,
    bitsetWordsPerGroup: Int
) -> [(String, any KernelTemplateArg)] {
    [
        ("GROUP_SIZE", configuration.groupSize),
        ("VALUE_COUNT", valueCount),
        ("GROUP_COUNT", groupCount),
        ("BASE_BITS", configuration.preset.baseMagnitudeBits),
        ("HIGH_BITS", configuration.preset.highMagnitudeBits),
        ("HIGH_NUMERATOR", 1),
        ("HIGH_DENOMINATOR", 2),
        ("MAG_WORDS_PER_GROUP", magnitudeWordsPerGroup),
        ("BITSET_WORDS_PER_GROUP", bitsetWordsPerGroup),
        ("ROLE", metalRoleValue(configuration.role)),
        ("SEED", Int(UInt32(truncatingIfNeeded: configuration.seed))),
    ]
}

private func metalRoleValue(_ role: TurboQuantTensorRole) -> Int {
    switch role {
    case .key:
        0
    case .value:
        1
    case .vector:
        2
    }
}

private func validateAttentionArray(_ array: MLXArray, groupSize: Int) throws {
    guard array.ndim == 4 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "attention tensors must have shape [B, H, T, D]"
        )
    }
    guard array.size > 0 else {
        throw TurboQuantError.invalidMetalConfiguration("empty attention tensors are not supported")
    }
    guard array.dtype.isFloatingPoint else {
        throw TurboQuantError.invalidMetalConfiguration("attention tensor dtype must be floating point")
    }
    guard groupSize > 0 else {
        throw TurboQuantError.invalidGroupSize(groupSize)
    }
    guard groupSize <= 128, groupSize % 32 == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "group size must be 32, 64, 96, or 128 for compressed attention"
        )
    }
    guard [64, 80, 96, 128, 256].contains(array.dim(3)) else {
        throw TurboQuantError.invalidMetalConfiguration(
            "head dimension \(array.dim(3)) is not supported by compressed attention"
        )
    }
}

private func validateAttentionLayout(
    _ layout: TurboQuantAttentionLayout,
    role: TurboQuantTensorRole,
    groupSize: Int
) throws {
    guard role == .key || role == .value else {
        throw TurboQuantError.invalidMetalConfiguration(
            "compressed attention codes must be encoded as key or value"
        )
    }
    guard layout.layoutVersion == TurboQuantAttentionLayout.currentVersion else {
        throw TurboQuantError.invalidMetalConfiguration(
            "unsupported compressed attention layout version \(layout.layoutVersion)"
        )
    }
    guard layout.batchSize > 0, layout.kvHeadCount > 0, layout.capacity > 0,
        layout.logicalLength >= 0, layout.logicalLength <= layout.capacity,
        layout.headDimension > 0
    else {
        throw TurboQuantError.invalidMetalConfiguration("invalid compressed attention layout shape")
    }
    guard layout.ringOffset >= 0, layout.ringOffset < layout.capacity else {
        throw TurboQuantError.invalidMetalConfiguration("ring offset is outside cache capacity")
    }
    guard layout.groupsPerVector == (layout.headDimension + groupSize - 1) / groupSize else {
        throw TurboQuantError.invalidMetalConfiguration("groups per vector does not match layout")
    }
}

private func validateAttentionQuery(
    _ queries: MLXArray,
    code: TurboQuantAttentionCode
) throws {
    try validateAttentionArray(queries, groupSize: code.groupSize)
    guard queries.dim(0) == code.layout.batchSize else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query batch size does not match compressed attention cache"
        )
    }
    guard queries.dim(3) == code.layout.headDimension else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query head dimension does not match compressed attention cache"
        )
    }
    guard queries.dim(1) % code.layout.kvHeadCount == 0 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "query heads must be a multiple of KV heads"
        )
    }
}

private func validateAttentionPair(
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode
) throws {
    try validateAttentionLayout(keyCode.layout, role: keyCode.role, groupSize: keyCode.groupSize)
    try validateAttentionLayout(valueCode.layout, role: valueCode.role, groupSize: valueCode.groupSize)
    guard keyCode.role == .key, valueCode.role == .value else {
        throw TurboQuantError.invalidMetalConfiguration("compressed attention requires key and value codes")
    }
    guard keyCode.layout == valueCode.layout else {
        throw TurboQuantError.invalidMetalConfiguration("key and value compressed layouts differ")
    }
    guard keyCode.preset == valueCode.preset, keyCode.groupSize == valueCode.groupSize else {
        throw TurboQuantError.invalidMetalConfiguration("key and value compressed presets differ")
    }
}

private func applyAttentionMask(
    _ scores: inout MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    stream: StreamOrDevice
) {
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1),
            expandedDimensions(kIndices, axis: -2),
            stream: stream
        )
        scores = `where`(
            causalMask,
            scores,
            MLXArray(-Float.greatestFiniteMagnitude),
            stream: stream
        )

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = `where`(
                maskArray,
                scores,
                MLXArray(-Float.greatestFiniteMagnitude),
                stream: stream
            )
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                scores = `where`(
                    maskArray,
                    scores,
                    MLXArray(-Float.greatestFiniteMagnitude),
                    stream: stream
                )
            } else {
                scores = scores + maskArray
            }
        }

    case .none:
        break
    }
}

private func attentionTemplate(
    configuration: TurboQuantConfiguration,
    layout: TurboQuantAttentionLayout,
    inputLength: Int,
    outputLength: Int,
    queryHeadCount: Int,
    queryLength: Int,
    outputDType: DType,
    causal: Bool
) -> [(String, any KernelTemplateArg)] {
    [
        ("BATCH_SIZE", layout.batchSize),
        ("KV_HEADS", layout.kvHeadCount),
        ("QUERY_HEADS", queryHeadCount),
        ("INPUT_LENGTH", inputLength),
        ("OUTPUT_LENGTH", outputLength),
        ("CAPACITY", layout.capacity),
        ("LOGICAL_LENGTH", layout.logicalLength),
        ("RING_OFFSET", layout.ringOffset),
        ("QUERY_LENGTH", queryLength),
        ("HEAD_DIM", layout.headDimension),
        ("GROUP_SIZE", configuration.groupSize),
        ("GROUPS_PER_VECTOR", layout.groupsPerVector),
        ("BASE_BITS", configuration.preset.baseMagnitudeBits),
        ("HIGH_BITS", configuration.preset.highMagnitudeBits),
        ("MAG_WORDS_PER_GROUP", layout.magnitudeWordsPerGroup),
        ("BITSET_WORDS_PER_GROUP", layout.bitsetWordsPerGroup),
        ("ROLE", metalRoleValue(configuration.role)),
        ("SEED", Int(UInt32(truncatingIfNeeded: configuration.seed))),
        ("OUTPUT_DTYPE", outputDType),
        ("DO_CAUSAL", causal),
    ]
}

private enum TurboQuantMetalKernels {
    static let encode = MLXFast.metalKernel(
        name: "turboquant_polar_qjl_encode",
        inputNames: ["x"],
        outputNames: ["packed", "signs", "high_mask", "residual_signs", "scales"],
        source: encodeSource
    )

    static let decode = MLXFast.metalKernel(
        name: "turboquant_polar_qjl_decode",
        inputNames: ["packed", "signs", "high_mask", "residual_signs", "scales"],
        outputNames: ["out"],
        source: decodeSource
    )

    static let encodeAttention = MLXFast.metalKernel(
        name: "turboquant_attention_encode",
        inputNames: ["x"],
        outputNames: ["packed", "signs", "high_mask", "residual_signs", "scales"],
        source: encodeAttentionSource,
        header: attentionHeader
    )

    static let qk = MLXFast.metalKernel(
        name: "turboquant_attention_qk",
        inputNames: ["q", "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales"],
        outputNames: ["scores"],
        source: qkSource,
        header: attentionHeader
    )

    static let av = MLXFast.metalKernel(
        name: "turboquant_attention_av",
        inputNames: ["weights", "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales"],
        outputNames: ["out"],
        source: avSource,
        header: attentionHeader
    )

    static let fusedAttention = MLXFast.metalKernel(
        name: "turboquant_attention_fused_decode",
        inputNames: [
            "q",
            "k_packed", "k_signs", "k_high_mask", "k_residual_signs", "k_scales",
            "v_packed", "v_signs", "v_high_mask", "v_residual_signs", "v_scales",
        ],
        outputNames: ["out"],
        source: fusedAttentionSource,
        header: attentionHeader
    )

    private static let encodeSource = """
        uint group_id = thread_position_in_grid.x;
        if (group_id >= GROUP_COUNT) {
            return;
        }

        uint start = group_id * GROUP_SIZE;
        uint count = min(uint(GROUP_SIZE), uint(VALUE_COUNT) - start);
        if (count == 0) {
            return;
        }

        thread float values[GROUP_SIZE];
        thread float magnitudes[GROUP_SIZE];
        float max_abs = 0.0f;

        for (uint local = 0; local < count; local++) {
            uint index = start + local;
            uint mixed = uint(SEED) + index * 0x9E3779B9u;
            mixed ^= mixed >> 16;
            mixed *= 0x7FEB352Du;
            mixed ^= mixed >> 15;
            mixed *= 0x846CA68Bu;
            mixed ^= mixed >> 16;

            float value = float(x[index]);
            if ((mixed & 1u) != 0u) {
                value = -value;
            }
            values[local] = value;
            float magnitude = fabs(value);
            magnitudes[local] = magnitude;
            max_abs = max(max_abs, magnitude);
        }

        float base_max = float((1 << BASE_BITS) - 1);
        float high_max = float((1 << HIGH_BITS) - 1);
        float safe_max = max(max_abs, 1.17549435e-38f);
        float base_scale = safe_max / base_max;
        float high_scale = safe_max / high_max;
        uint scale_base = group_id * 3;
        scales[scale_base] = base_scale;
        scales[scale_base + 1] = high_scale;
        scales[scale_base + 2] = 0.0f;

        uint bitset_base = group_id * BITSET_WORDS_PER_GROUP;
        for (uint word = 0; word < BITSET_WORDS_PER_GROUP; word++) {
            signs[bitset_base + word] = 0u;
            high_mask[bitset_base + word] = 0u;
            residual_signs[bitset_base + word] = 0u;
        }

        uint packed_base = group_id * MAG_WORDS_PER_GROUP;
        for (uint word = 0; word < MAG_WORDS_PER_GROUP; word++) {
            packed[packed_base + word] = 0u;
        }

        uint high_count = uint(round(float(count * HIGH_NUMERATOR) / float(HIGH_DENOMINATOR)));
        float residual_sum = 0.0f;
        for (uint local = 0; local < count; local++) {
            float magnitude = magnitudes[local];
            uint rank = 0;
            for (uint other = 0; other < count; other++) {
                bool greater = magnitudes[other] > magnitude;
                bool tied_before = magnitudes[other] == magnitude && other < local;
                if (greater || tied_before) {
                    rank += 1;
                }
            }

            bool high_precision = rank < high_count;
            uint bits = high_precision ? uint(HIGH_BITS) : uint(BASE_BITS);
            float scale = high_precision ? high_scale : base_scale;
            uint level_max = (1u << bits) - 1u;
            uint quantized = uint(clamp(round(magnitude / scale), 0.0f, float(level_max)));
            if (ROLE != 1) {
                float signed_decode = (values[local] < 0.0f ? -1.0f : 1.0f)
                    * float(quantized) * scale;
                residual_sum += fabs(values[local] - signed_decode);
            }
        }
        if (ROLE != 1) {
            scales[scale_base + 2] = residual_sum / float(count);
        }

        uint bit_offset = 0;
        for (uint local = 0; local < count; local++) {
            float magnitude = magnitudes[local];
            uint rank = 0;
            for (uint other = 0; other < count; other++) {
                bool greater = magnitudes[other] > magnitude;
                bool tied_before = magnitudes[other] == magnitude && other < local;
                if (greater || tied_before) {
                    rank += 1;
                }
            }

            bool high_precision = rank < high_count;
            uint bits = high_precision ? uint(HIGH_BITS) : uint(BASE_BITS);
            float scale = high_precision ? high_scale : base_scale;
            uint level_max = (1u << bits) - 1u;
            uint quantized = uint(clamp(round(magnitude / scale), 0.0f, float(level_max)));

            uint word_index = local >> 5;
            uint word_bit = local & 31u;
            uint mask_bit = 1u << word_bit;
            if (values[local] < 0.0f) {
                signs[bitset_base + word_index] |= mask_bit;
            }
            if (high_precision) {
                high_mask[bitset_base + word_index] |= mask_bit;
            }

            if (ROLE != 1) {
                float signed_decode = (values[local] < 0.0f ? -1.0f : 1.0f)
                    * float(quantized) * scale;
                float residual = values[local] - signed_decode;
                if (residual < 0.0f) {
                    residual_signs[bitset_base + word_index] |= mask_bit;
                }
            }

            for (uint bit = 0; bit < bits; bit++) {
                if ((quantized & (1u << bit)) != 0u) {
                    uint global_bit = bit_offset + bit;
                    uint packed_word = global_bit >> 5;
                    uint packed_bit = global_bit & 31u;
                    packed[packed_base + packed_word] |= 1u << packed_bit;
                }
            }
            bit_offset += bits;
        }
        """

    private static let decodeSource = """
        uint index = thread_position_in_grid.x;
        if (index >= VALUE_COUNT) {
            return;
        }

        uint group_id = index / GROUP_SIZE;
        uint local = index - group_id * GROUP_SIZE;
        uint bitset_base = group_id * BITSET_WORDS_PER_GROUP;
        uint word_index = local >> 5;
        uint word_bit = local & 31u;
        uint mask_bit = 1u << word_bit;
        bool high_precision = (high_mask[bitset_base + word_index] & mask_bit) != 0u;
        uint bits = high_precision ? uint(HIGH_BITS) : uint(BASE_BITS);
        uint scale_base = group_id * 3;
        float scale = high_precision ? scales[scale_base + 1] : scales[scale_base];

        uint bit_offset = 0;
        for (uint prior = 0; prior < local; prior++) {
            uint prior_word = prior >> 5;
            uint prior_bit = prior & 31u;
            bool prior_high = (high_mask[bitset_base + prior_word] & (1u << prior_bit)) != 0u;
            bit_offset += prior_high ? uint(HIGH_BITS) : uint(BASE_BITS);
        }

        uint packed_base = group_id * MAG_WORDS_PER_GROUP;
        uint quantized = 0u;
        for (uint bit = 0; bit < bits; bit++) {
            uint global_bit = bit_offset + bit;
            uint packed_word = global_bit >> 5;
            uint packed_bit = global_bit & 31u;
            if ((packed[packed_base + packed_word] & (1u << packed_bit)) != 0u) {
                quantized |= 1u << bit;
            }
        }

        float sign = (signs[bitset_base + word_index] & mask_bit) != 0u ? -1.0f : 1.0f;
        float value = sign * float(quantized) * scale;
        if (ROLE != 1) {
            float residual_sign =
                (residual_signs[bitset_base + word_index] & mask_bit) != 0u ? -1.0f : 1.0f;
            value += residual_sign * scales[scale_base + 2];
        }

        uint mixed = uint(SEED) + index * 0x9E3779B9u;
        mixed ^= mixed >> 16;
        mixed *= 0x7FEB352Du;
        mixed ^= mixed >> 15;
        mixed *= 0x846CA68Bu;
        mixed ^= mixed >> 16;
        if ((mixed & 1u) != 0u) {
            value = -value;
        }

        out[index] = value;
        """

    private static let attentionHeader = """
        inline uint tq_mix(uint seed, uint index) {
            uint mixed = seed + index * 0x9E3779B9u;
            mixed ^= mixed >> 16;
            mixed *= 0x7FEB352Du;
            mixed ^= mixed >> 15;
            mixed *= 0x846CA68Bu;
            mixed ^= mixed >> 16;
            return mixed;
        }

        inline bool tq_random_sign(uint seed, uint index) {
            return (tq_mix(seed, index) & 1u) != 0u;
        }

        inline uint tq_bitset_offset(uint batch, uint head, uint token, uint group, uint word) {
            return (((batch * uint(KV_HEADS) + head) * uint(CAPACITY) + token)
                * uint(GROUPS_PER_VECTOR) + group) * uint(BITSET_WORDS_PER_GROUP) + word;
        }

        inline uint tq_packed_offset(uint batch, uint head, uint token, uint group, uint word) {
            return (((batch * uint(KV_HEADS) + head) * uint(CAPACITY) + token)
                * uint(GROUPS_PER_VECTOR) + group) * uint(MAG_WORDS_PER_GROUP) + word;
        }

        inline uint tq_scale_offset(uint batch, uint head, uint token, uint group, uint scale_index) {
            return ((((batch * uint(KV_HEADS) + head) * uint(CAPACITY) + token)
                * uint(GROUPS_PER_VECTOR) + group) * 3u) + scale_index;
        }

        inline uint tq_physical_token(uint logical_token) {
            return (uint(RING_OFFSET) + logical_token) % uint(CAPACITY);
        }

        inline uint tq_read_magnitude(
            device const uint* packed,
            device const uint* high_mask,
            uint batch,
            uint head,
            uint token,
            uint group,
            uint local
        ) {
            uint bitset_word = local >> 5;
            uint bitset_bit = local & 31u;
            bool high_precision =
                (high_mask[tq_bitset_offset(batch, head, token, group, bitset_word)]
                    & (1u << bitset_bit)) != 0u;
            uint bits = high_precision ? uint(HIGH_BITS) : uint(BASE_BITS);

            uint bit_offset = 0u;
            for (uint prior = 0; prior < local; prior++) {
                uint prior_word = prior >> 5;
                uint prior_bit = prior & 31u;
                bool prior_high =
                    (high_mask[tq_bitset_offset(batch, head, token, group, prior_word)]
                        & (1u << prior_bit)) != 0u;
                bit_offset += prior_high ? uint(HIGH_BITS) : uint(BASE_BITS);
            }

            uint quantized = 0u;
            for (uint bit = 0; bit < bits; bit++) {
                uint global_bit = bit_offset + bit;
                uint packed_word = global_bit >> 5;
                uint packed_bit = global_bit & 31u;
                if ((packed[tq_packed_offset(batch, head, token, group, packed_word)]
                    & (1u << packed_bit)) != 0u) {
                    quantized |= 1u << bit;
                }
            }
            return quantized;
        }

        inline float tq_decode_attention_value(
            device const uint* packed,
            device const uint* signs,
            device const uint* high_mask,
            device const uint* residual_signs,
            device const float* scales,
            uint batch,
            uint head,
            uint token,
            uint dimension,
            uint seed,
            uint role
        ) {
            uint group = dimension / uint(GROUP_SIZE);
            uint local = dimension - group * uint(GROUP_SIZE);
            uint bitset_word = local >> 5;
            uint bitset_bit = local & 31u;
            uint bit_mask = 1u << bitset_bit;
            bool high_precision =
                (high_mask[tq_bitset_offset(batch, head, token, group, bitset_word)] & bit_mask) != 0u;
            float scale = high_precision
                ? scales[tq_scale_offset(batch, head, token, group, 1u)]
                : scales[tq_scale_offset(batch, head, token, group, 0u)];
            uint quantized = tq_read_magnitude(packed, high_mask, batch, head, token, group, local);
            float sign =
                (signs[tq_bitset_offset(batch, head, token, group, bitset_word)] & bit_mask) != 0u
                    ? -1.0f : 1.0f;
            float value = sign * float(quantized) * scale;

            if (role != 1u) {
                float residual_sign =
                    (residual_signs[tq_bitset_offset(batch, head, token, group, bitset_word)]
                        & bit_mask) != 0u ? -1.0f : 1.0f;
                value += residual_sign * scales[tq_scale_offset(batch, head, token, group, 2u)];
            }

            if (tq_random_sign(seed, dimension)) {
                value = -value;
            }
            return value;
        }
        """

    private static let encodeAttentionSource = """
        uint row_group_id = thread_position_in_grid.x;
        uint total = uint(BATCH_SIZE) * uint(KV_HEADS) * uint(INPUT_LENGTH) * uint(GROUPS_PER_VECTOR);
        if (row_group_id >= total) {
            return;
        }

        uint group = row_group_id % uint(GROUPS_PER_VECTOR);
        uint token = (row_group_id / uint(GROUPS_PER_VECTOR)) % uint(INPUT_LENGTH);
        uint head = (row_group_id / (uint(GROUPS_PER_VECTOR) * uint(INPUT_LENGTH))) % uint(KV_HEADS);
        uint batch = row_group_id / (uint(GROUPS_PER_VECTOR) * uint(INPUT_LENGTH) * uint(KV_HEADS));
        if (token >= uint(CAPACITY)) {
            return;
        }

        uint group_start = group * uint(GROUP_SIZE);
        uint count = min(uint(GROUP_SIZE), uint(HEAD_DIM) - group_start);
        thread float values[GROUP_SIZE];
        thread float magnitudes[GROUP_SIZE];
        float max_abs = 0.0f;

        for (uint local = 0; local < count; local++) {
            uint dimension = group_start + local;
            uint input_index =
                (((batch * uint(KV_HEADS) + head) * uint(INPUT_LENGTH) + token)
                    * uint(HEAD_DIM)) + dimension;
            float value = float(x[input_index]);
            if (tq_random_sign(uint(SEED), dimension)) {
                value = -value;
            }
            values[local] = value;
            float magnitude = fabs(value);
            magnitudes[local] = magnitude;
            max_abs = max(max_abs, magnitude);
        }

        float base_max = float((1 << BASE_BITS) - 1);
        float high_max = float((1 << HIGH_BITS) - 1);
        float safe_max = max(max_abs, 1.17549435e-38f);
        float base_scale = safe_max / base_max;
        float high_scale = safe_max / high_max;
        scales[tq_scale_offset(batch, head, token, group, 0u)] = base_scale;
        scales[tq_scale_offset(batch, head, token, group, 1u)] = high_scale;
        scales[tq_scale_offset(batch, head, token, group, 2u)] = 0.0f;

        for (uint word = 0; word < uint(BITSET_WORDS_PER_GROUP); word++) {
            signs[tq_bitset_offset(batch, head, token, group, word)] = 0u;
            high_mask[tq_bitset_offset(batch, head, token, group, word)] = 0u;
            residual_signs[tq_bitset_offset(batch, head, token, group, word)] = 0u;
        }
        for (uint word = 0; word < uint(MAG_WORDS_PER_GROUP); word++) {
            packed[tq_packed_offset(batch, head, token, group, word)] = 0u;
        }

        uint high_count = uint(round(float(count) * 0.5f));
        float residual_sum = 0.0f;
        for (uint local = 0; local < count; local++) {
            float magnitude = magnitudes[local];
            uint rank = 0u;
            for (uint other = 0; other < count; other++) {
                bool greater = magnitudes[other] > magnitude;
                bool tied_before = magnitudes[other] == magnitude && other < local;
                if (greater || tied_before) {
                    rank += 1u;
                }
            }
            bool high_precision = rank < high_count;
            uint bits = high_precision ? uint(HIGH_BITS) : uint(BASE_BITS);
            float scale = high_precision ? high_scale : base_scale;
            uint level_max = (1u << bits) - 1u;
            uint quantized = uint(clamp(round(magnitude / scale), 0.0f, float(level_max)));
            if (ROLE != 1) {
                float signed_decode = (values[local] < 0.0f ? -1.0f : 1.0f)
                    * float(quantized) * scale;
                residual_sum += fabs(values[local] - signed_decode);
            }
        }
        if (ROLE != 1) {
            scales[tq_scale_offset(batch, head, token, group, 2u)] = residual_sum / float(count);
        }

        uint bit_offset = 0u;
        for (uint local = 0; local < count; local++) {
            float magnitude = magnitudes[local];
            uint rank = 0u;
            for (uint other = 0; other < count; other++) {
                bool greater = magnitudes[other] > magnitude;
                bool tied_before = magnitudes[other] == magnitude && other < local;
                if (greater || tied_before) {
                    rank += 1u;
                }
            }
            bool high_precision = rank < high_count;
            uint bits = high_precision ? uint(HIGH_BITS) : uint(BASE_BITS);
            float scale = high_precision ? high_scale : base_scale;
            uint level_max = (1u << bits) - 1u;
            uint quantized = uint(clamp(round(magnitude / scale), 0.0f, float(level_max)));

            uint word = local >> 5;
            uint bit = local & 31u;
            uint mask = 1u << bit;
            if (values[local] < 0.0f) {
                signs[tq_bitset_offset(batch, head, token, group, word)] |= mask;
            }
            if (high_precision) {
                high_mask[tq_bitset_offset(batch, head, token, group, word)] |= mask;
            }
            if (ROLE != 1) {
                float signed_decode = (values[local] < 0.0f ? -1.0f : 1.0f)
                    * float(quantized) * scale;
                float residual = values[local] - signed_decode;
                if (residual < 0.0f) {
                    residual_signs[tq_bitset_offset(batch, head, token, group, word)] |= mask;
                }
            }

            for (uint packed_bit = 0; packed_bit < bits; packed_bit++) {
                if ((quantized & (1u << packed_bit)) != 0u) {
                    uint global_bit = bit_offset + packed_bit;
                    uint packed_word = global_bit >> 5;
                    uint packed_word_bit = global_bit & 31u;
                    packed[tq_packed_offset(batch, head, token, group, packed_word)] |=
                        1u << packed_word_bit;
                }
            }
            bit_offset += bits;
        }
        """

    private static let qkSource = """
        uint index = thread_position_in_grid.x;
        uint total = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH) * uint(LOGICAL_LENGTH);
        if (index >= total) {
            return;
        }

        float attention_scale = as_type<float>(uint(ATTENTION_SCALE_BITS));
        uint logical_token = index % uint(LOGICAL_LENGTH);
        uint q_token = (index / uint(LOGICAL_LENGTH)) % uint(QUERY_LENGTH);
        uint q_head = (index / (uint(LOGICAL_LENGTH) * uint(QUERY_LENGTH))) % uint(QUERY_HEADS);
        uint batch = index / (uint(LOGICAL_LENGTH) * uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint physical_token = tq_physical_token(logical_token);

        float sum = 0.0f;
        for (uint dimension = 0; dimension < uint(HEAD_DIM); dimension++) {
            uint q_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + dimension;
            float key_value = tq_decode_attention_value(
                k_packed, k_signs, k_high_mask, k_residual_signs, k_scales,
                batch, kv_head, physical_token, dimension, uint(SEED), 0u);
            sum += float(q[q_index]) * key_value;
        }
        scores[index] = sum * attention_scale;
        """

    private static let avSource = """
        uint index = thread_position_in_grid.x;
        uint total = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH) * uint(HEAD_DIM);
        if (index >= total) {
            return;
        }

        uint dimension = index % uint(HEAD_DIM);
        uint q_token = (index / uint(HEAD_DIM)) % uint(QUERY_LENGTH);
        uint q_head = (index / (uint(HEAD_DIM) * uint(QUERY_LENGTH))) % uint(QUERY_HEADS);
        uint batch = index / (uint(HEAD_DIM) * uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;

        float sum = 0.0f;
        for (uint logical_token = 0; logical_token < uint(LOGICAL_LENGTH); logical_token++) {
            uint physical_token = tq_physical_token(logical_token);
            uint weight_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(LOGICAL_LENGTH)) + logical_token;
            float value = tq_decode_attention_value(
                v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                batch, kv_head, physical_token, dimension, uint(SEED), 1u);
            sum += float(weights[weight_index]) * value;
        }
        out[index] = sum;
        """

    private static let fusedAttentionSource = """
        uint row = thread_position_in_grid.x;
        uint total_rows = uint(BATCH_SIZE) * uint(QUERY_HEADS) * uint(QUERY_LENGTH);
        if (row >= total_rows) {
            return;
        }

        float attention_scale = as_type<float>(uint(ATTENTION_SCALE_BITS));
        uint q_token = row % uint(QUERY_LENGTH);
        uint q_head = (row / uint(QUERY_LENGTH)) % uint(QUERY_HEADS);
        uint batch = row / (uint(QUERY_LENGTH) * uint(QUERY_HEADS));
        uint repeats = uint(QUERY_HEADS) / uint(KV_HEADS);
        uint kv_head = q_head / repeats;
        uint causal_limit = uint(LOGICAL_LENGTH) - uint(QUERY_LENGTH) + q_token;

        thread float accum[HEAD_DIM];
        for (uint dimension = 0; dimension < uint(HEAD_DIM); dimension++) {
            accum[dimension] = 0.0f;
        }

        float row_max = -INFINITY;
        for (uint logical_token = 0; logical_token < uint(LOGICAL_LENGTH); logical_token++) {
            if (DO_CAUSAL && logical_token > causal_limit) {
                continue;
            }
            uint physical_token = tq_physical_token(logical_token);
            float score = 0.0f;
            for (uint dimension = 0; dimension < uint(HEAD_DIM); dimension++) {
                uint q_index =
                    (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                        * uint(HEAD_DIM)) + dimension;
                float key_value = tq_decode_attention_value(
                    k_packed, k_signs, k_high_mask, k_residual_signs, k_scales,
                    batch, kv_head, physical_token, dimension, uint(SEED), 0u);
                score += float(q[q_index]) * key_value;
            }
            row_max = max(row_max, score * attention_scale);
        }

        float row_sum = 0.0f;
        for (uint logical_token = 0; logical_token < uint(LOGICAL_LENGTH); logical_token++) {
            if (DO_CAUSAL && logical_token > causal_limit) {
                continue;
            }
            uint physical_token = tq_physical_token(logical_token);
            float score = 0.0f;
            for (uint dimension = 0; dimension < uint(HEAD_DIM); dimension++) {
                uint q_index =
                    (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                        * uint(HEAD_DIM)) + dimension;
                float key_value = tq_decode_attention_value(
                    k_packed, k_signs, k_high_mask, k_residual_signs, k_scales,
                    batch, kv_head, physical_token, dimension, uint(SEED), 0u);
                score += float(q[q_index]) * key_value;
            }
            float weight = exp(score * attention_scale - row_max);
            row_sum += weight;
            for (uint dimension = 0; dimension < uint(HEAD_DIM); dimension++) {
                float value = tq_decode_attention_value(
                    v_packed, v_signs, v_high_mask, v_residual_signs, v_scales,
                    batch, kv_head, physical_token, dimension, uint(VALUE_SEED), 1u);
                accum[dimension] += weight * value;
            }
        }

        float inv_sum = 1.0f / max(row_sum, 1.17549435e-38f);
        for (uint dimension = 0; dimension < uint(HEAD_DIM); dimension++) {
            uint out_index =
                (((batch * uint(QUERY_HEADS) + q_head) * uint(QUERY_LENGTH) + q_token)
                    * uint(HEAD_DIM)) + dimension;
            out[out_index] = accum[dimension] * inv_sum;
        }
        """
}
