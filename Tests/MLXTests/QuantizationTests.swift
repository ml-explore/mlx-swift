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

    func testQuantizedLinearStoresGlobalScale() {
        let globalScale = MLXArray(1.0, dtype: .float32)
        let quantized = QuantizedLinear(
            weight: MLXArray.zeros([8, 4], dtype: .uint32),
            bias: nil,
            scales: MLXArray.ones([8, 4], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            globalScale: globalScale)

        XCTAssertNotNil(quantized.globalScale)
        XCTAssertEqual(quantized.globalScale?.dtype, .float32)
        XCTAssertNotNil(quantized.parameters()["global_scale"])
        XCTAssertNil(quantized.parameters()["globalScale"])
    }

    func testQuantizedEmbeddingStoresGlobalScale() {
        let globalScale = MLXArray(1.0, dtype: .float32)
        let quantized = QuantizedEmbedding(
            weight: MLXArray.zeros([8, 2], dtype: .uint32),
            scales: MLXArray.ones([8, 2], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            globalScale: globalScale)

        XCTAssertNotNil(quantized.globalScale)
        XCTAssertEqual(quantized.globalScale?.dtype, .float32)
        XCTAssertNotNil(quantized.parameters()["global_scale"])
        XCTAssertNil(quantized.parameters()["globalScale"])
    }

    func testQuantizedGlobalScaleIsOptionalParameter() {
        let linear = QuantizedLinear(
            weight: MLXArray.zeros([8, 4], dtype: .uint32),
            bias: nil,
            scales: MLXArray.ones([8, 4], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4)
        let embedding = QuantizedEmbedding(
            weight: MLXArray.zeros([8, 2], dtype: .uint32),
            scales: MLXArray.ones([8, 2], dtype: .uint8),
            biases: nil,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4)

        XCTAssertNil(linear.globalScale)
        XCTAssertNil(linear.parameters()["global_scale"])
        XCTAssertNil(embedding.globalScale)
        XCTAssertNil(embedding.parameters()["global_scale"])
    }
}
