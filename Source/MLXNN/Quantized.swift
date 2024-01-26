import Foundation
import MLX
import MLXRandom

public class QuantizedLinear: Linear {

    let groupSize: Int
    let bits: Int

    let scales: MLXArray
    let biases: MLXArray

    public convenience init(
        _ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true, groupSize: Int = 64,
        bits: Int = 4
    ) {
        let scale = sqrt(1 / Float(inputDimensions))
        let weight = uniform(low: -scale, high: scale, [outputDimensions, inputDimensions])

        let bias = bias ? MLXArray.zeros([outputDimensions]) : nil

        self.init(weight: weight, bias: bias, groupSize: groupSize, bits: bits)
    }

    public init(weight: MLXArray, bias: MLXArray?, groupSize: Int = 64, bits: Int = 4) {
        self.groupSize = groupSize
        self.bits = bits

        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight, groupSize: groupSize, bits: bits)

        self.scales = scales
        self.biases = biases

        super.init(weight: quantizedWeight, bias: bias)

        self.freeze()
    }

    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) {
        super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = quantizedMatmul(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits
        )
        if let bias {
            x = x + bias
        }
        return x
    }

    static public func from(linear: Module, groupSize: Int = 64, bits: Int = 4) -> QuantizedLinear?
    {
        guard let linear = linear as? Linear else { return nil }

        return QuantizedLinear(
            weight: linear.weight, bias: linear.bias, groupSize: groupSize, bits: bits)
    }

    static public func quantize(
        model: Module,
        groupSize: Int = 64,
        bits: Int = 4,
        predicate: (Module) -> Bool = { $0 is Linear }
    ) {
        let updates = model.leafModules().compactMapValues { m -> Module? in
            Self.from(linear: m, groupSize: groupSize, bits: bits)
        }

        model.update(modules: updates)
    }
}
