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

    private func relativeMSE(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let squaredError = zip(lhs, rhs).reduce(Float(0)) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + delta * delta
        }
        let signal = lhs.reduce(Float(0)) { $0 + $1 * $1 }
        return squaredError / max(signal, Float.leastNonzeroMagnitude)
    }

    private func pearsonCorrelation(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let count = Float(lhs.count)
        let lhsMean = lhs.reduce(Float(0), +) / count
        let rhsMean = rhs.reduce(Float(0), +) / count
        var numerator = Float(0)
        var lhsVariance = Float(0)
        var rhsVariance = Float(0)
        for (left, right) in zip(lhs, rhs) {
            let lhsCentered = left - lhsMean
            let rhsCentered = right - rhsMean
            numerator += lhsCentered * rhsCentered
            lhsVariance += lhsCentered * lhsCentered
            rhsVariance += rhsCentered * rhsCentered
        }
        return numerator / max(sqrt(lhsVariance * rhsVariance), Float.leastNonzeroMagnitude)
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
        let output = turboQuantizedMM(
            x, packed, configuration: configuration, stream: .device(.cpu))

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
        XCTAssertEqual(first.format, TurboQuantReferenceFormat.turboQuantProd)
        XCTAssertGreaterThan(first.storageByteCount, 0)
        XCTAssertFalse(first.highScales.isEmpty)
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
        let mse =
            zip(values, decoded)
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

        XCTAssertLessThan(report.relativeMSE, 0.085)
        XCTAssertGreaterThan(report.cosineSimilarity, 0.955)
    }

    func testTurboQuantReferenceValueBitsStorageAccounting() throws {
        try requireMLXRuntime()

        let values = (0 ..< 256).map { index in
            let position = Double(index)
            let sineTerm = 0.4 * sin(position * 0.07)
            let cosineTerm = 0.15 * cos(position * 0.17)
            return Float(sineTerm + cosineTerm)
        }
        let x = MLXArray(values, [4, 64])
        let twoBit = try turboQuantReferenceEncode(
            x,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .polarQJLReference,
                valueBits: 2
            )
        )
        let fourBit = try turboQuantReferenceEncode(
            x,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .polarQJLReference,
                valueBits: 4
            )
        )

        XCTAssertEqual(twoBit.format, TurboQuantReferenceFormat.affineValue)
        XCTAssertEqual(fourBit.format, TurboQuantReferenceFormat.affineValue)
        XCTAssertLessThan(twoBit.approximateBitsPerValue, 3.1)
        XCTAssertLessThan(fourBit.approximateBitsPerValue, 5.1)
        XCTAssertLessThan(twoBit.storageByteCount, fourBit.storageByteCount)
    }

    func testTurboQuantProductInnerProductBiasAndRetrieval() throws {
        try requireMLXRuntime()

        let queryValues = (0 ..< 64).map { index in
            let position = Double(index)
            let sineTerm = 0.35 * sin(position * 0.13)
            let cosineTerm = 0.2 * cos(position * 0.05)
            return Float(sineTerm + cosineTerm)
        }
        let needleValues = queryValues.map { $0 * 1.35 }
        let query = MLXArray(queryValues, [64])
        let keys = (0 ..< 16).map { keyIndex in
            (0 ..< 64).map { dim in
                if keyIndex == 7 { return needleValues[dim] }
                let position = Double(keyIndex * 64 + dim)
                return Float(0.25 * sin(position * 0.071) - 0.18 * cos(position * 0.113))
            }
        }

        var exactScores: [Float] = []
        var estimatedScores: [Float] = []
        for (keyIndex, keyValues) in keys.enumerated() {
            let exactScore = zip(queryValues, keyValues).reduce(Float(0)) { partial, pair in
                partial + pair.0 * pair.1
            }
            exactScores.append(exactScore)
            let code = try turboQuantReferenceEncode(
                MLXArray(keyValues, [64]),
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .polarQJLReference,
                    seed: UInt64(0x600D_0000 + keyIndex)
                )
            )
            estimatedScores.append(try turboQuantReferenceInnerProduct(query: query, code: code))
        }

        XCTAssertEqual(estimatedScores.enumerated().max(by: { $0.element < $1.element })?.offset, 7)
        XCTAssertGreaterThan(pearsonCorrelation(exactScores, estimatedScores), 0.7)

        let target = MLXArray(keys[3], [64])
        let exact = exactScores[3]
        let estimates = try (0 ..< 32).map { seedOffset in
            let code = try turboQuantReferenceEncode(
                target,
                configuration: TurboQuantConfiguration(
                    preset: .turbo3_5,
                    role: .key,
                    groupSize: 64,
                    backend: .polarQJLReference,
                    seed: UInt64(0xB1A5_0000 + seedOffset)
                )
            )
            return try turboQuantReferenceInnerProduct(query: query, code: code)
        }
        let average = estimates.reduce(Float(0), +) / Float(estimates.count)
        XCTAssertLessThan(abs(average - exact) / max(abs(exact), Float.leastNonzeroMagnitude), 0.25)
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
        XCTAssertEqual(
            availability.selectedKernelProfile, capabilities.runtimeProbe.selectedKernelProfile)

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
            XCTAssertEqual(code.shape, [2, 64])
            XCTAssertLessThan(relativeMSE(values, decoded), 0.1)
        }
    }

    func testTurboQuantMetalCodecUsesGPUStreamWhenDefaultDeviceIsCPU() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let values = (0 ..< 128).map { index in
            Float(sin(Double(index) * 0.07))
        }
        let x = MLXArray(values, [2, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .key,
            groupSize: 64,
            backend: .metalPolarQJL,
            seed: 0xDEAD_BEEF_0000_0017
        )

        try Device.withDefaultDevice(.cpu) {
            XCTAssertTrue(StreamOrDevice.default.description.contains("cpu"))

            let code = try turboQuantMetalEncode(x, configuration: configuration)
            let decoded = try turboQuantMetalDecode(code).asArray(Float.self)

            XCTAssertEqual(code.shape, [2, 64])
            XCTAssertEqual(decoded.count, values.count)
        }
    }

    func testTurboQuantMetalMatmulMatchesDecodedReferenceWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else {
            throw XCTSkip("Metal runtime unavailable")
        }

        let xValues = (0 ..< 192).map { index in
            let position = Double(index)
            return Float(0.4 * sin(position * 0.07) + 0.2 * cos(position * 0.17))
        }
        let wValues = (0 ..< 320).map { index in
            let position = Double(index)
            return Float(0.3 * cos(position * 0.05) - 0.15 * sin(position * 0.11))
        }
        let x = MLXArray(xValues, [3, 64])
        let w = MLXArray(wValues, [5, 64])
        let configuration = TurboQuantConfiguration(
            preset: .turbo3_5,
            role: .vector,
            groupSize: 64,
            backend: .metalPolarQJL,
            seed: 0xC0FF_EE00_0000_0042
        )

        let code = try turboQuantMetalEncode(w, configuration: configuration)
        let decoded = try turboQuantMetalDecode(code, dtype: .float32)
        let reference = matmul(x, decoded.transposed())
        let output = try turboQuantizedMM(x, code, transpose: true, outputDType: .float32)

        XCTAssertEqual(output.shape, [3, 5])
        XCTAssertTrue(allClose(output, reference, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertEqual(code.magnitudeWordsPerGroup, 5)

        let columnMajorWeight = decoded.transposed()
        let columnCode = try turboQuantMetalEncode(columnMajorWeight, configuration: configuration)
        let columnReference = matmul(x, try turboQuantMetalDecode(columnCode, dtype: .float32))
        let columnOutput = try turboQuantizedMM(
            x, columnCode, transpose: false, outputDType: .float32)

        XCTAssertEqual(columnOutput.shape, [3, 5])
        XCTAssertTrue(
            allClose(columnOutput, columnReference, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    func testTurboQuantAttentionLayoutIsRowWise() throws {
        let layout = try turboQuantAttentionLayout(shape: [1, 2, 3, 80], groupSize: 64)

        XCTAssertEqual(layout.layoutVersion, 4)
        XCTAssertEqual(layout.logicalShape, [1, 2, 3, 80])
        XCTAssertEqual(layout.pinnedPrefixLength, 0)
        XCTAssertEqual(layout.groupsPerVector, 2)
        XCTAssertEqual(layout.bitsetWordsPerGroup, 2)
    }

    func testTurboQuantCompressedAttentionUsesProductEstimatorWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let qValues: [Float] = (0 ..< 512).map { index in
            let position = Double(index)
            return Float(sin(position * 0.03) + 0.2 * cos(position * 0.11))
        }
        let kValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(cos(position * 0.05) * 0.5 + sin(position * 0.17) * 0.1)
        }
        let vValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(sin(position * 0.07) * 0.25 - cos(position * 0.13) * 0.2)
        }
        let queries = MLXArray(qValues, [1, 4, 2, 64])
        let keys = MLXArray(kValues, [1, 2, 5, 64])
        let values = MLXArray(vValues, [1, 2, 5, 64])
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
        let fullPrecisionReference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(64)),
            mask: .causal
        )

        let twoStage = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: false
        )
        let fused = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: true
        )

        XCTAssertEqual(twoStage.shape, [1, 4, 2, 64])
        XCTAssertEqual(fused.shape, [1, 4, 2, 64])
        XCTAssertTrue(allClose(fused, twoStage, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertLessThan(
            relativeMSE(
                fullPrecisionReference.asArray(Float.self),
                fused.asArray(Float.self)
            ),
            0.12
        )
        XCTAssertLessThan(
            relativeMSE(
                fullPrecisionReference.asArray(Float.self),
                twoStage.asArray(Float.self)
            ),
            0.12
        )
    }

    func testTurboQuantCompressedAttentionSupportsBatchedInputsWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let qValues: [Float] = (0 ..< 1024).map { index in
            let position = Double(index)
            return Float(0.3 * sin(position * 0.021) + 0.17 * cos(position * 0.071))
        }
        let kValues: [Float] = (0 ..< 1280).map { index in
            let position = Double(index)
            return Float(0.25 * cos(position * 0.037) - 0.11 * sin(position * 0.097))
        }
        let vValues: [Float] = (0 ..< 1280).map { index in
            let position = Double(index)
            return Float(0.19 * sin(position * 0.043) + 0.13 * cos(position * 0.083))
        }
        let queries = MLXArray(qValues, [2, 4, 2, 64])
        let keys = MLXArray(kValues, [2, 2, 5, 64])
        let values = MLXArray(vValues, [2, 2, 5, 64])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 31
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 37
            )
        )
        let fullPrecisionReference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(64)),
            mask: .causal
        )

        let twoStage = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: false
        )
        let fused = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            preferOnlineFused: true
        )

        XCTAssertEqual(twoStage.shape, [2, 4, 2, 64])
        XCTAssertEqual(fused.shape, [2, 4, 2, 64])
        XCTAssertTrue(allClose(fused, twoStage, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertLessThan(
            relativeMSE(
                fullPrecisionReference.asArray(Float.self),
                fused.asArray(Float.self)
            ),
            0.12
        )
    }

    func testTurboQuantCompressedAttentionSupportsSinksWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let qValues: [Float] = (0 ..< 512).map { index in
            let position = Double(index)
            return Float(0.24 * sin(position * 0.031) + 0.12 * cos(position * 0.089))
        }
        let kValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(0.2 * cos(position * 0.047) - 0.08 * sin(position * 0.101))
        }
        let vValues: [Float] = (0 ..< 640).map { index in
            let position = Double(index)
            return Float(0.18 * sin(position * 0.053) + 0.09 * cos(position * 0.077))
        }
        let queries = MLXArray(qValues, [1, 4, 2, 64])
        let keys = MLXArray(kValues, [1, 2, 5, 64])
        let values = MLXArray(vValues, [1, 2, 5, 64])
        let sinks = MLXArray([0.3 as Float, -0.2, 0.1, -0.4])
        let keyCode = try turboQuantMetalEncodeAttention(
            keys,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .key,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 41
            )
        )
        let valueCode = try turboQuantMetalEncodeAttention(
            values,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 43
            )
        )
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            sinks: sinks
        )

        let output = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: 1 / sqrt(Float(64)),
            mask: .causal,
            sinks: sinks,
            preferOnlineFused: true
        )

        XCTAssertEqual(output.shape, [1, 4, 2, 64])
        XCTAssertLessThan(
            relativeMSE(
                reference.asArray(Float.self),
                output.asArray(Float.self)
            ),
            0.12
        )
    }

    func testTurboQuantAttentionDecodeHonorsRotatingLayoutWhenAvailable() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            throw XCTSkip("Metal compressed attention unavailable")
        }

        let capacity = 6
        let headDimension = 64
        let physicalValues = (0 ..< capacity).flatMap { token in
            Array(repeating: Float(token + 1) * 0.25, count: headDimension)
        }
        let physical = MLXArray(physicalValues, [1, 1, capacity, headDimension])
        let code = try turboQuantMetalEncodeAttention(
            physical,
            configuration: TurboQuantConfiguration(
                preset: .turbo3_5,
                role: .value,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 29
            ),
            capacity: capacity,
            logicalLength: capacity,
            ringOffset: 2,
            pinnedPrefixLength: 2
        )

        let decoded = try turboQuantMetalDecodeAttention(code, outputDType: .float32)
        let expectedTokenOrder = [0, 1, 4, 5, 2, 3]
        let expectedValues = expectedTokenOrder.flatMap { token in
            Array(repeating: Float(token + 1) * 0.25, count: headDimension)
        }
        let expected = MLXArray(expectedValues, [1, 1, capacity, headDimension])

        XCTAssertTrue(allClose(decoded, expected, rtol: 1e-6, atol: 1e-6).item(Bool.self))
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
