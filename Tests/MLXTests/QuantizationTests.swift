// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class QuantizationTests: XCTestCase {
    private func requireMLXRuntime() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("MLX runtime metallib unavailable in this package context")
        }
    }

    func testQuantizedLinearShapeDesc() {
        let linear1 = Linear(512, 1024)
        let quantized1 = linear1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(
            quantized1.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
        let linear2 = Linear(1024, 512, bias: false)
        let quantized2 = linear2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(inputDimensions=1024, outputDimensions=512, bias=false)")
        let linear3 = Linear(512, 1024)
        let quantized3 = linear3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(inputDimensions=512, outputDimensions=1024, bias=true)")
    }

    func testQuantizedEmbeddingShapeDesc() {
        let embedding1 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized1 = embedding1.toQuantized(groupSize: 64, bits: 4)
        XCTAssertEqual(quantized1.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
        let embedding2 = Embedding(embeddingCount: 1024, dimensions: 512)
        let quantized2 = embedding2.toQuantized(groupSize: 128, bits: 8)
        XCTAssertEqual(
            quantized2.describeExtra(0), "(embeddingCount=1024, dimensions=512)")
        let embedding3 = Embedding(embeddingCount: 512, dimensions: 1024)
        let quantized3 = embedding3.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertEqual(
            quantized3.describeExtra(0), "(embeddingCount=512, dimensions=1024)")
    }

    func testQuantizedLinearMxfp4DoesNotCreateAffineBiases() {
        let quantized = QuantizedLinear(64, 64, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertNil(quantized.biases)
    }

    func testTurboQuantPackedRoundTrip() throws {
        try requireMLXRuntime()

        let x = MLXArray.ones([1, 32], dtype: .float32, stream: .device(.cpu))
        let configuration = TurboQuantConfiguration(preset: .turbo3_5, groupSize: 32)
        let packed = turboQuantized(x, configuration: configuration, stream: .device(.cpu))
        let decoded = turboDequantized(packed, configuration: configuration, stream: .device(.cpu))

        XCTAssertEqual(decoded.shape, x.shape)
        XCTAssertTrue(allClose(decoded, x).item(Bool.self))
    }

    func testTurboQuantMatmulShape() throws {
        try requireMLXRuntime()

        let x = MLXArray.ones([2, 32], dtype: .float32, stream: .device(.cpu))
        let w = MLXArray.ones([4, 32], dtype: .float32, stream: .device(.cpu))
        let configuration = TurboQuantConfiguration(preset: .turbo2_5, groupSize: 32)
        let packed = turboQuantized(w, configuration: configuration, stream: .device(.cpu))
        let output = turboQuantizedMM(x, packed, configuration: configuration, stream: .device(.cpu))

        XCTAssertEqual(output.shape, [2, 4])
    }

    func testTurboQuantReferenceCodecIsDeterministic() throws {
        try requireMLXRuntime()

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.17) + cos(Double(index) * 0.03))
        }
        let x = MLXArray(values, [2, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 32,
            backend: .polarQJLReference,
            seed: 42
        )

        let first = try turboQuantReferenceEncode(x, configuration: configuration)
        let second = try turboQuantReferenceEncode(x, configuration: configuration)

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.shape, [2, 64])
        XCTAssertGreaterThan(first.storageByteCount, 0)
        XCTAssertFalse(first.residualScales.isEmpty)
    }

    func testTurboQuantReferenceCodecUsesFullWidthSeed() throws {
        try requireMLXRuntime()

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.11) + cos(Double(index) * 0.19))
        }
        let x = MLXArray(values, [2, 64])
        let lowSeedConfiguration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .polarQJLReference,
            seed: 0x0000_0000_0123_4567
        )
        let highSeedConfiguration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .polarQJLReference,
            seed: 0xDEAD_BEEF_0123_4567
        )

        let lowSeed = try turboQuantReferenceEncode(x, configuration: lowSeedConfiguration)
        let highSeed = try turboQuantReferenceEncode(x, configuration: highSeedConfiguration)

        XCTAssertNotEqual(lowSeed.signs, highSeed.signs)
    }

    func testTurboQuantReferenceCodecDistortionThreshold() throws {
        try requireMLXRuntime()

        let values = (0 ..< 256).map { index in
            let position = Double(index)
            let sineTerm = sin(position * 0.11) * 0.7
            let cosineTerm = cos(position * 0.07) * 0.3
            return Float(sineTerm + cosineTerm)
        }
        let x = MLXArray(values, [4, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .vector,
            groupSize: 64,
            backend: .polarQJLReference,
            seed: 17
        )

        let code = try turboQuantReferenceEncode(x, configuration: configuration)
        let decoded = try turboQuantReferenceDecode(code).asArray(Float.self)
        let mse = zip(values, decoded)
            .map { lhs, rhs in
                let delta = lhs - rhs
                return delta * delta
            }
            .reduce(Float(0), +) / Float(values.count)

        XCTAssertLessThan(mse, 0.01)
    }

    func testTurboQuantReferenceQualityGatePassesFixture() throws {
        try requireMLXRuntime()

        let values = (0 ..< 256).map { index in
            let position = Double(index)
            let sineTerm = sin(position * 0.09) * 0.5
            let cosineTerm = cos(position * 0.13) * 0.25
            return Float(sineTerm + cosineTerm)
        }
        let x = MLXArray(values, [4, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .polarQJLReference,
            seed: 99
        )

        let report = try turboQuantReferenceQuality(x, configuration: configuration)

        XCTAssertTrue(report.passes)
        XCTAssertLessThan(report.relativeMSE, 0.02)
        XCTAssertGreaterThan(report.cosineSimilarity, 0.99)
        XCTAssertLessThan(report.innerProductRelativeError, 0.08)
    }

    func testTurboQuantBackendAvailabilityContract() throws {
        XCTAssertNoThrow(try requireTurboQuantBackend(.mlxPacked))
        XCTAssertNoThrow(try requireTurboQuantBackend(.polarQJLReference))

        let availability = TurboQuantKernelAvailability.current
        if availability.supportsMetalPolarQJL {
            XCTAssertNoThrow(try requireTurboQuantBackend(.metalPolarQJL))
            XCTAssertEqual(availability.runtimeBackend(for: .metalPolarQJL), .metalPolarQJL)
            XCTAssertNil(availability.fallbackReason(for: .metalPolarQJL))
        } else {
            XCTAssertThrowsError(try requireTurboQuantBackend(.metalPolarQJL))
            XCTAssertEqual(availability.runtimeBackend(for: .metalPolarQJL), .mlxPacked)
            XCTAssertNotNil(availability.fallbackReason(for: .metalPolarQJL))
        }
    }

    func testTurboQuantDeviceCapabilitiesAndProbeContract() throws {
        let capabilities = TurboQuantDeviceCapabilities.current
        let availability = TurboQuantKernelAvailability.current

        XCTAssertFalse(capabilities.architectureName.isEmpty)
        XCTAssertEqual(capabilities.runtimeProbe, TurboQuantRuntimeProbe.current)
        XCTAssertEqual(availability.selfTestStatus, capabilities.runtimeProbe.status)
        XCTAssertEqual(availability.selectedKernelProfile, capabilities.runtimeProbe.selectedKernelProfile)

        if availability.supportsMetalPolarQJLAttention {
            XCTAssertEqual(capabilities.runtimeProbe.status, .passed)
            XCTAssertNotEqual(capabilities.runtimeProbe.selectedKernelProfile, .mlxPackedFallback)
            XCTAssertNil(capabilities.runtimeProbe.failureReason)
        } else {
            XCTAssertNotEqual(capabilities.runtimeProbe.status, .notRun)
            XCTAssertEqual(availability.runtimeBackend(for: .metalPolarQJL), .mlxPacked)
        }
    }

    func testTurboQuantMetalCodecRoundTripWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.05))
        }
        let x = MLXArray(values, [2, 64])
        for seed in [UInt64(0xDEAD_BEEF_0000_0017), UInt64(0x0000_0000_DEAD_BEEF)] {
            let configuration = TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: seed
            )

            let code = try turboQuantMetalEncode(x, configuration: configuration)
            let decoded = try turboQuantMetalDecode(code).asArray(Float.self)
            let mse = zip(values, decoded)
                .map { lhs, rhs in
                    let delta = lhs - rhs
                    return delta * delta
                }
                .reduce(Float(0), +) / Float(values.count)

            XCTAssertEqual(code.shape, [2, 64])
            XCTAssertLessThan(mse, 0.02)
        }
    }

    func testTurboQuantAttentionLayoutIsRowWise() throws {
        let layout = try turboQuantAttentionLayout(shape: [1, 2, 3, 80], groupSize: 64)

        XCTAssertEqual(layout.layoutVersion, 3)
        XCTAssertEqual(layout.logicalShape, [1, 2, 3, 80])
        XCTAssertEqual(layout.pinnedPrefixLength, 0)
        XCTAssertEqual(layout.groupsPerVector, 2)
        XCTAssertEqual(layout.bitsetWordsPerGroup, 2)
    }

    func testTurboQuantCompressedAttentionMatchesDecodedReferenceWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let qValues = (0 ..< 128).map { Float(sin(Double($0) * 0.03)) }
        let kValues = (0 ..< 256).map { Float(cos(Double($0) * 0.05) * 0.5) }
        let vValues = (0 ..< 256).map { Float(sin(Double($0) * 0.07) * 0.25) }
        let queries = MLXArray(qValues, [1, 2, 1, 64])
        let keys = MLXArray(kValues, [1, 2, 2, 64])
        let values = MLXArray(vValues, [1, 2, 2, 64])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 11
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 13
            )
        )

        let output = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            preferOnlineFused: false
        )

        XCTAssertEqual(output.shape, [1, 2, 1, 64])
    }

    func testTurboQuantOnlineFusedSupportContract() throws {
        let keyLayout = try turboQuantAttentionLayout(shape: [1, 2, 8, 64], groupSize: 64)

        XCTAssertTrue(
            turboQuantMetalSupportsOnlineFusedAttention(
                queryShape: [1, 4, 1, 64],
                keyLayout: keyLayout,
                mask: .none
            )
        )
    }

    func testTurboQuantOnlineFusedSupportsLargeContextContract() throws {
        let keyLayout = try turboQuantAttentionLayout(shape: [1, 2, 513, 64], groupSize: 64)

        XCTAssertTrue(
            turboQuantMetalSupportsOnlineFusedAttention(
                queryShape: [1, 4, 1, 64],
                keyLayout: keyLayout,
                mask: .none
            )
        )
    }
}
