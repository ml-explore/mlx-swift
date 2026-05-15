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
    public var supportsMetalPolarQJL: Bool

    public init(
        supportsMLXPacked: Bool = true,
        supportsPolarQJLReference: Bool = true,
        supportsMetalPolarQJLCodec: Bool = false,
        supportsMetalPolarQJL: Bool = false
    ) {
        self.supportsMLXPacked = supportsMLXPacked
        self.supportsPolarQJLReference = supportsPolarQJLReference
        self.supportsMetalPolarQJLCodec = supportsMetalPolarQJLCodec
        self.supportsMetalPolarQJL = supportsMetalPolarQJL
    }

    public static var current: TurboQuantKernelAvailability {
        TurboQuantKernelAvailability(supportsMetalPolarQJLCodec: metalRuntimeAvailable())
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
    case invalidReferenceCode(String)
    case unsupportedBackend(TurboQuantBackend, String)

    public var description: String {
        switch self {
        case .invalidGroupSize(let groupSize):
            "TurboQuant group size must be positive, got \(groupSize)."
        case .invalidMetalConfiguration(let message):
            "Invalid TurboQuant Metal configuration: \(message)"
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
    public var signs: Data
    public var highPrecisionMask: Data
    public var residualSigns: Data
    public var packedMagnitudes: Data

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
        self.signs = signs
        self.highPrecisionMask = highPrecisionMask
        self.residualSigns = residualSigns
        self.packedMagnitudes = packedMagnitudes
    }

    public var storageByteCount: Int {
        packedMagnitudes.count
            + signs.count
            + highPrecisionMask.count
            + residualSigns.count
            + (baseScales.count + highScales.count) * MemoryLayout<Float>.stride
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
            [groupCount, 2],
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

public func requireTurboQuantBackend(_ backend: TurboQuantBackend) throws {
    let availability = TurboQuantKernelAvailability.current
    guard availability.supports(backend) else {
        throw TurboQuantError.unsupportedBackend(
            backend,
            availability.fallbackReason(for: backend) ?? "Backend unavailable."
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
            let signedDecoded = (value.sign == .minus ? -1 : 1) * Float(quantizedMagnitude) * scale
            let residual = value - signedDecoded

            setPackedBit(&signs, index: absoluteIndex, value: value.sign == .minus)
            setPackedBit(&highPrecisionMask, index: absoluteIndex, value: highPrecision)
            if configuration.role != .value {
                setPackedBit(&residualSigns, index: absoluteIndex, value: residual.sign == .minus)
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
                reconstructed += residualSign * code.residualScale * scale
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
    guard configuration.qjlResidualScale == 0.5 else {
        throw TurboQuantError.invalidMetalConfiguration(
            "Metal codec currently supports qjlResidualScale == 0.5"
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
        scales[group_id * 2] = base_scale;
        scales[group_id * 2 + 1] = high_scale;

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
        float scale = high_precision ? scales[group_id * 2 + 1] : scales[group_id * 2];

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
            value += residual_sign * 0.5f * scale;
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
}
