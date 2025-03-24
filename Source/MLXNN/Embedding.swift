// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Implements a simple lookup table that maps each input integer to a high-dimensional vector.
///
/// Typically used to embed discrete tokens for processing by neural networks.
open class Embedding: Module, UnaryLayer, Quantizable {
    public let weight: MLXArray

    /// Implements a simple lookup table that maps each input integer to a high-dimensional vector.
    ///
    /// Typically used to embed discrete tokens for processing by neural networks.
    ///
    /// - Parameters:
    ///   - embeddingCount: How many possible discrete tokens can we embed.  Usually called the vocabulary size.
    ///   - dimensions: dimensionality of the embeddings.
    public init(embeddingCount: Int, dimensions: Int) {
        let scale = sqrt(1 / Float(dimensions))
        self.weight = MLXRandom.normal([embeddingCount, dimensions]) * scale
    }

    /// Initializer meant for subclasses to provide weight directly.
    public init(weight: MLXArray) {
        self.weight = weight
    }

    /// Describe the shape of the `weight`.
    public override func describeExtra(_ indent: Int) -> String {
        weight.shape.description
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        weight[x]
    }

    /// Call the embedding layer as a linear layer.
    ///
    /// Use this for example when input embedding and output projection
    /// weights are tied.
    open func asLinear(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T)
    }

    public func toQuantized(groupSize: Int, bits: Int) -> Module {
        QuantizedEmbedding(self, groupSize: groupSize, bits: bits)
    }
}
