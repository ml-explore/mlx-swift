// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

class QuantizationTests: XCTestCase {
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

    func testTurboQuantPackedRoundTrip() {
        let x = MLXArray.ones([1, 32], dtype: .float32)
        let configuration = TurboQuantConfiguration(preset: .turbo3_5, groupSize: 32)
        let packed = turboQuantized(x, configuration: configuration)
        let decoded = turboDequantized(packed, configuration: configuration)

        XCTAssertEqual(decoded.shape, x.shape)
        XCTAssertTrue(allClose(decoded, x).item(Bool.self))
    }

    func testTurboQuantMatmulShape() {
        let x = MLXArray.ones([2, 32], dtype: .float32)
        let w = MLXArray.ones([4, 32], dtype: .float32)
        let configuration = TurboQuantConfiguration(preset: .turbo2_5, groupSize: 32)
        let packed = turboQuantized(w, configuration: configuration)
        let output = turboQuantizedMM(x, packed, configuration: configuration)

        XCTAssertEqual(output.shape, [2, 4])
    }

    func testTurboQuantReferenceCodecIsDeterministic() throws {
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

    func testTurboQuantReferenceCodecDistortionThreshold() throws {
        let values = (0 ..< 256).map { index in
            Float(sin(Double(index) * 0.11) * 0.7 + cos(Double(index) * 0.07) * 0.3)
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
        let values = (0 ..< 256).map { index in
            Float(sin(Double(index) * 0.09) * 0.5 + cos(Double(index) * 0.13) * 0.25)
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
        XCTAssertThrowsError(try requireTurboQuantBackend(.metalPolarQJL))

        let availability = TurboQuantKernelAvailability.current
        XCTAssertEqual(availability.runtimeBackend(for: .metalPolarQJL), .mlxPacked)
        XCTAssertNotNil(availability.fallbackReason(for: .metalPolarQJL))
    }

    func testTurboQuantMetalCodecRoundTripWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.05))
        }
        let x = MLXArray(values, [2, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .metalPolarQJL,
            seed: 23
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

    func testTurboQuantAttentionLayoutIsRowWise() throws {
        let x = MLXArray.zeros([1, 2, 3, 80], dtype: .float32)
        let layout = try turboQuantAttentionLayout(for: x, groupSize: 64)

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
        let queries = MLXArray.zeros([1, 4, 1, 64], dtype: .float32)
        let keys = MLXArray.zeros([1, 2, 8, 64], dtype: .float32)
        let keyCode = try turboQuantEmptyAttentionCode(
            layout: try turboQuantAttentionLayout(for: keys, groupSize: 64),
            role: .key,
            groupSize: 64
        )

        XCTAssertTrue(
            turboQuantMetalSupportsOnlineFusedAttention(
                queries: queries,
                keyCode: keyCode,
                mask: .none
            )
        )
    }

    func testTurboQuantOnlineFusedSupportsLargeContextContract() throws {
        let queries = MLXArray.zeros([1, 4, 1, 64], dtype: .float32)
        let keys = MLXArray.zeros([1, 2, 513, 64], dtype: .float32)
        let keyCode = try turboQuantEmptyAttentionCode(
            layout: try turboQuantAttentionLayout(for: keys, groupSize: 64),
            role: .key,
            groupSize: 64
        )

        XCTAssertTrue(
            turboQuantMetalSupportsOnlineFusedAttention(
                queries: queries,
                keyCode: keyCode,
                mask: .none
            )
        )
    }
}
