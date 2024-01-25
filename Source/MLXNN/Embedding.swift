import Foundation
import MLX
import MLXRandom

/// Implements a simple lookup table that maps each input integer to a high-dimensional vector.
///
/// Typically used to embed discrete tokens for processing by neural networks.
public class Embedding: Module, UnaryModel {

    let weight: MLXArray

    public init(embeddingCount: Int, dimensions: Int) {
        let scale = sqrt(1 / Float(dimensions))
        self.weight = MLXRandom.normal([embeddingCount, dimensions]) * scale
    }

    public override func describeExtra(_ indent: Int) -> String {
        weight.shape.description
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        weight[x]
    }
}
