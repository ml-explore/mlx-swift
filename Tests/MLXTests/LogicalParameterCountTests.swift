// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@Suite
struct LogicalParameterCountTests {

    // MARK: - Test layers

    /// A `QuantizedLinear` subclass with an extra parameter, e.g. shaped like
    /// a QLoRA adapter.  The extra parameter must be counted.
    final class QuantizedLinearWithExtra: QuantizedLinear {
        let extra: MLXArray

        init(_ inputDimensions: Int, _ outputDimensions: Int, extraRows: Int) {
            self.extra = MLXArray.zeros([extraRows, inputDimensions])
            super.init(
                weight: MLXRandom.uniform(0 ..< 1, [outputDimensions, inputDimensions]),
                bias: nil)
        }
    }

    /// A ``Quantized`` layer that does not derive from `QuantizedLinear`, similar
    /// to `QuantizedSwitchLinear` in mlx-swift-lm (a stack of expert weights).
    final class QuantizedExperts: Module, Quantized {
        let groupSize: Int
        let bits: Int
        let mode: QuantizationMode

        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray?
        let bias: MLXArray

        init(experts: Int, inputDimensions: Int, outputDimensions: Int, bits: Int = 4) {
            self.groupSize = 64
            self.bits = bits
            self.mode = .affine

            let w = MLXRandom.uniform(0 ..< 1, [experts, outputDimensions, inputDimensions])
            let (wq, scales, biases) = MLX.quantized(w, groupSize: groupSize, bits: bits)
            self.weight = wq
            self.scales = scales
            self.biases = biases
            self.bias = MLXArray.zeros([experts, outputDimensions])
        }
    }

    final class Model: Module {
        @ModuleInfo var embedding: Embedding
        @ModuleInfo var layers: [Linear]
        @ModuleInfo var head: Linear

        override init() {
            self.embedding = Embedding(embeddingCount: 100, dimensions: 128)
            self.layers = [Linear(128, 256), Linear(256, 128, bias: false)]
            self.head = Linear(128, 100)
        }
    }

    // MARK: - Tests

    @Test func testLinear() {
        #expect(Linear(128, 32).logicalParameterCount == 128 * 32 + 32)
        #expect(Linear(128, 32, bias: false).logicalParameterCount == 128 * 32)
    }

    @Test(arguments: [
        (4, QuantizationMode.affine, 64),
        (8, QuantizationMode.affine, 64),
        (3, QuantizationMode.affine, 64),
        (4, QuantizationMode.mxfp4, 32),
    ])
    func testQuantizedLinear(bits: Int, mode: QuantizationMode, groupSize: Int) {
        let withBias = QuantizedLinear(
            256, 64, bias: true, groupSize: groupSize, bits: bits, mode: mode)
        #expect(withBias.logicalParameterCount == 256 * 64 + 64)

        let noBias = QuantizedLinear(
            256, 64, bias: false, groupSize: groupSize, bits: bits, mode: mode)
        #expect(noBias.logicalParameterCount == 256 * 64)
    }

    @Test func testQuantizedEmbedding() {
        let e = QuantizedEmbedding(embeddingCount: 100, dimensions: 128)
        #expect(e.logicalParameterCount == 100 * 128)
    }

    @Test func testQuantizedSubclassExtraParameters() {
        let q = QuantizedLinearWithExtra(256, 64, extraRows: 8)
        #expect(q.logicalParameterCount == 256 * 64 + 8 * 256)
    }

    @Test func testCustomQuantizedLayer() {
        let q = QuantizedExperts(experts: 4, inputDimensions: 256, outputDimensions: 64)
        #expect(q.logicalParameterCount == 4 * 256 * 64 + 4 * 64)
    }

    @Test func testNested() {
        let model = Model()
        let expected =
            100 * 128  // embedding
            + 128 * 256 + 256  // layers[0]
            + 256 * 128  // layers[1]
            + 128 * 100 + 100  // head
        #expect(model.logicalParameterCount == expected)
    }

    @Test func testQuantizePreservesCount() {
        // the logical count should not change when a model is quantized
        let model = Model()
        let before = model.logicalParameterCount

        quantize(model: model, groupSize: 64, bits: 4, mode: .affine)
        #expect(model.head is QuantizedLinear)
        #expect(model.embedding is QuantizedEmbedding)

        #expect(model.logicalParameterCount == before)
    }

    @Test func testMaterializedModule() {
        let model = Model()
        quantize(model: model, groupSize: 64, bits: 4, mode: .affine)
        let expected = model.logicalParameterCount

        let mm = MaterializedModule(model)
        #expect(mm.logicalParameterCount == expected)
    }
}
