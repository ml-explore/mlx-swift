// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Upsample the input signal spatially.
///
/// See ``Upsample/init(scaleFactor:mode:)`` for more information.
open class Upsample: Module, UnaryLayer {

    public enum Mode {
        /// Nearest neighbor upsampling
        case nearest

        /// Linear interpolation subsampling.  If `alignCorners` is `true` then the top
        /// and left edge of the input and output will match as will the bottom right edge.
        case linear(alignCorners: Bool = false)
    }

    public let scaleFactor: FloatOrArray
    public let mode: Mode

    /// Upsample the input signal spatially.
    ///
    /// The spatial dimensions are by convention dimensions `1` to `x.ndim - 2`.
    /// The first is the batch dimension and the last is the feature dimension.
    ///
    /// For example, an audio signal would be 3D with 1 spatial dimension, an image
    /// 4D with 2 and so on and so forth.
    ///
    /// There are two upsampling algorithms implemented nearest neighbor
    /// (``Upsample/Mode/nearest``) upsampling
    /// and linear interpolation (``Upsample/Mode/linear(alignCorners:)``). Both can be
    /// applied to any number of spatial
    /// dimensions and the linear interpolation will be bilinear, trilinear etc
    /// when applied to more than one spatial dimension.
    ///
    /// - Parameters:
    ///   - scaleFactor: The multiplier for the spatial size.
    ///     If a `float` is provided, it is the multiplier for all spatial dimensions.
    ///     Otherwise, the number of scale factors provided must match the
    ///     number of spatial dimensions.
    ///   - mode: The upsampling algorithm
    public init(scaleFactor: FloatOrArray, mode: Mode = .nearest) {
        self.scaleFactor = scaleFactor
        self.mode = mode
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dimensions = x.ndim - 2
        if dimensions <= 0 {
            fatalError(
                """
                [Upsample] The input should have at least 1 spatial
                dimension which means it should be at least 3D but
                \(x.ndim)D was provided
                """)
        }

        let scaleFactor = self.scaleFactor.asArray(dimensions: dimensions)

        switch mode {
        case .nearest:
            return upsampleNearest(x, scale: scaleFactor)
        case .linear(let alignCorners):
            return upsampleLinear(x, scale: scaleFactor, alignCorners: alignCorners)
        }
    }
}

private func upsampleNearest(_ x: MLXArray, scale: [Float]) -> MLXArray {
    let dimensions = x.ndim - 2
    precondition(
        dimensions == scale.count, "A scale needs to be provided for each spatial dimension")

    // get a truncated version of the scales
    let intScale = scale.map { Int($0) }
    let intFloatScale = intScale.map { Float($0) }

    if intFloatScale == scale {
        // Int scale means we can simply expand-broadcast and reshape
        var shape = x.shape
        for d in 0 ..< dimensions {
            shape.insert(1, at: 2 + 2 * d)
        }
        var x = x.reshaped(shape)

        for d in 0 ..< dimensions {
            shape[2 + 2 * d] = intScale[d]
        }
        x = broadcast(x, to: shape)

        for d in 0 ..< dimensions {
            shape[d + 1] *= shape[d + 2]
            shape.remove(at: d + 2)
        }
        x = x.reshaped(shape)

        return x
    } else {
        // Float scales
        let N = x.shape.dropFirst().dropLast()

        var indices: [MLXArrayIndex] = [0...]
        for (i, (n, s)) in zip(N, scale).enumerated() {
            indices.append(nearestIndices(dimension: n, scale: s, dim: i, ndim: dimensions))
        }

        return x[indices]
    }
}

private typealias IndexWeight = (MLXArray, MLXArray)

private func upsampleLinear(_ x: MLXArray, scale: [Float], alignCorners: Bool) -> MLXArray {
    let dimensions = x.ndim - 2
    precondition(
        dimensions == scale.count, "A scale needs to be provided for each spatial dimension")

    let N = x.shape.dropFirst().dropLast()

    // compute the sampling grid
    var indexWeights = [(IndexWeight, IndexWeight)]()
    for (i, (n, s)) in zip(N, scale).enumerated() {
        indexWeights.append(
            linearIndices(
                dimension: n, scale: s, alignCorners: alignCorners, dim: i, ndim: dimensions))
    }

    // sample and compute the weights
    var samples = [MLXArray]()
    var weights = [MLXArray]()
    for indexWeight in product(pairs: indexWeights) {
        let index = indexWeight.map { $0.0 }
        let weight = indexWeight.map { $0.1 }
        samples.append(x[[0...] + index])
        weights.append(weight.dropFirst().reduce(weight[0], { $0 * $1 }))
    }

    // interpolate
    return zip(weights.dropFirst(), samples.dropFirst())
        .reduce(weights[0] * samples[0]) { result, pair in
            result + (pair.0 * pair.1)
        }
}

private func product(pairs: [(IndexWeight, IndexWeight)]) -> [[IndexWeight]] {
    // in the python code this uses itertools.product() but we can get by with a
    // simple implementation for this constrained case.
    //
    // Given [(A, B), (C, D)] this will produce an array of all the
    // combinations of the pairs, e.g. [[A, C], [A, D], [B, C], [B, D]]

    precondition(pairs.count <= UInt.bitWidth)

    // all combinations of the L0 or R0 ... Ln / Rn
    var result = [[IndexWeight]]()

    // we can generate all the combinations with bits -- we are
    // selecting between the first or second of the input tuples,
    // so we can map like this:
    //
    // [(A, B), (C, D)] ->
    //  00 - [A, C]
    //  01 - [A, D]
    //  10 - [B, C]
    //  11 - [B, D]

    // the pattern to generate the bits and the mask for the width
    var pattern: UInt = 0
    let mask: UInt = (1 << pairs.count) - 1

    while true {
        let item =
            pairs
            .enumerated()
            .map { (index, pair) in

                // select the .0 or .1 element depending on the bit
                if pattern & (1 << index) == 0 {
                    return pair.0
                } else {
                    return pair.1
                }
            }

        result.append(item)

        // next bit pattern
        pattern = (pattern + 1) & mask

        // if it "wraps" around to 0 that means we covered all the patterns
        if pattern == 0 {
            break
        }
    }

    return result
}

private func nearestIndices(dimension: Int, scale: Float, dim: Int, ndim: Int) -> MLXArray {
    scaledIndices(dimension: dimension, scale: scale, alignCorners: true, dim: dim, ndim: ndim)
        .asType(.int32)
}

private func linearIndices(dimension: Int, scale: Float, alignCorners: Bool, dim: Int, ndim: Int)
    -> (IndexWeight, IndexWeight)
{
    let indices = scaledIndices(
        dimension: dimension, scale: scale, alignCorners: alignCorners, dim: dim, ndim: ndim)
    let indicesLeft = floor(indices)
    let indicesRight = ceil(indices)
    let weight = expandedDimensions(indices - indicesLeft, axis: -1)

    return (
        (indicesLeft.asType(.int32), 1 - weight),
        (indicesRight.asType(.int32), weight)
    )
}

private func scaledIndices(dimension N: Int, scale: Float, alignCorners: Bool, dim: Int, ndim: Int)
    -> MLXArray
{
    let M = Int(scale * Float(N))

    var indices: MLXArray
    if alignCorners {
        indices = MLXArray(0 ..< M).asType(.float32) * ((Float(N) - 1) / (Float(M) - 1))
    } else {
        let step = 1 / scale
        let start = ((Float(M) - 1) * step - Float(N) + 1) / 2
        indices = MLXArray(0 ..< M).asType(.float32) * step - start
        indices = clip(indices, min: 0, max: N - 1)
    }

    var shape = Array(repeating: 1, count: ndim)
    shape[dim] = -1

    return indices.reshaped(shape)
}
