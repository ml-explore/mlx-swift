// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXRandom

/// Implements a simple lookup table that maps each input integer to a high-dimensional vector.
///
/// Typically used to embed discrete tokens for processing by neural networks.
open class Embedding: Module, UnaryLayer {

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

    /// Describe the shape of the `weight`.
    public override func describeExtra(_ indent: Int) -> String {
        weight.shape.description
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        weight[x]
    }
}
