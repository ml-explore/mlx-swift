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

    func testTurboQuantBackendAvailabilityContract() throws {
        XCTAssertNoThrow(try requireTurboQuantBackend(.mlxPacked))
        XCTAssertNoThrow(try requireTurboQuantBackend(.polarQJLReference))
        XCTAssertThrowsError(try requireTurboQuantBackend(.metalPolarQJL))

        let availability = TurboQuantKernelAvailability.current
        XCTAssertEqual(availability.runtimeBackend(for: .metalPolarQJL), .mlxPacked)
        XCTAssertNotNil(availability.fallbackReason(for: .metalPolarQJL))
    }
}
