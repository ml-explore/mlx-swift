// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Upsample the input signal spatially.
///
/// See ``Upsample/init(scaleFactor:mode:)`` for more information.
open class Upsample: Module, UnaryLayer {

    public enum Mode: Sendable {
        /// Nearest neighbor upsampling
        case nearest

        /// Linear interpolation subsampling.  If `alignCorners` is `true` then the top
        /// and left edge of the input and output will match as will the bottom right edge.
        case linear(alignCorners: Bool = false)

        /// Cubic interpolation subsampling.  If `alignCorners` is `true` then the top
        /// and left edge of the input and output will match as will the bottom right edge.
        case cubic(alignCorners: Bool = false)
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
    /// There are three upsampling algorithms implemented:
    ///
    /// - ``Upsample/Mode-swift.enum/nearest``
    /// - ``Upsample/Mode-swift.enum/linear(alignCorners:)``
    /// - ``Upsample/Mode-swift.enum/cubic(alignCorners:)``
    ///
    /// All can be applied to any number of spatial
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
            return interpolate(
                x, scale: scaleFactor, indexes: linearIndices, alignCorners: alignCorners)
        case .cubic(let alignCorners):
            return interpolate(
                x, scale: scaleFactor, indexes: cubicIndices, alignCorners: alignCorners)
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

private func interpolate(
    _ x: MLXArray, scale: [Float], indexes: (Int, Float, Bool, Int, Int) -> [IndexWeight],
    alignCorners: Bool
) -> MLXArray {
    let dimensions = x.ndim - 2
    precondition(
        dimensions == scale.count, "A scale needs to be provided for each spatial dimension")

    let N = x.shape.dropFirst().dropLast()

    // compute the sampling grid
    var indexWeights = [[IndexWeight]]()
    for (i, (n, s)) in zip(N, scale).enumerated() {
        indexWeights.append(indexes(n, s, alignCorners, i, dimensions))
    }

    // sample and compute the weights
    var samples = [MLXArray]()
    var weights = [MLXArray]()
    for indexWeight in product(values: indexWeights) {
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

/// Return the product (element-wise permutation) across the given arrays.
///
/// Given an values like this:
///
/// ```swift
/// [
///     [1, 2, 3],
///     [4, 5, 6],
///     [7, 8, 9],
/// ]
/// ```
///
/// this will produce all permutations of `values[0]` with `values[...N]`:
///
/// ```swift
/// [
///     [1, 4, 7],
///     [2, 4, 7],
///     [3, 4, 7],
///     [1, 5, 7],
///     ...
///     [3, 6, 9],
/// ]
/// ```
///
/// - Parameter values: input values
private func product<T>(values: [[T]]) -> [[T]] {
    guard !values.isEmpty else { return [] }

    // if there are N items in values and M values per tuple there
    // will be M^N values in the result
    let perTuple = values[0].count
    let count = (0 ..< values.count).reduce(1) { c, _ in c * perTuple }

    var result = [[T]]()
    for resultIndex in 0 ..< count {
        var items = [T]()

        // use % and / to compute which item will be used from
        // each value[i]
        var indexGenerator = resultIndex
        for i in 0 ..< values.count {
            let index = indexGenerator % perTuple
            items.append(values[i][index])
            indexGenerator = indexGenerator / perTuple
        }

        result.append(items)
    }

    return result
}

private func nearestIndices(dimension N: Int, scale: Float, dim: Int, ndim: Int) -> MLXArray {
    let M = Int(scale * Float(N))
    var indices = arange(M, dtype: .float32)

    if M > N {
        indices = (indices + 0.5) * (Float(N) / Float(M)) - 0.5
        indices = indices.round()
    } else {
        indices = indices * (Float(N) / Float(M))
    }

    var shape = Array(repeating: 1, count: ndim)
    shape[dim] = -1

    return indices.asType(.uint32).reshaped(shape)
}

private func linearIndices(dimension: Int, scale: Float, alignCorners: Bool, dim: Int, ndim: Int)
    -> [IndexWeight]
{
    var indices = scaledIndices(
        dimension: dimension, scale: scale, alignCorners: alignCorners, dim: dim, ndim: ndim)
    indices = clip(indices, min: 0, max: dimension - 1)
    let indicesLeft = floor(indices)
    let indicesRight = ceil(indices)
    let weight = expandedDimensions(indices - indicesLeft, axis: -1)

    return [
        (indicesLeft.asType(.int32), 1 - weight),
        (indicesRight.asType(.int32), weight),
    ]
}

private let compiledGetWeight1: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    // PyTorch uses -0.5 for antialiasing=true (compatibility with PIL)
    // and uses -0.75 for antialiasing=false (compatibility with OpenCV)

    compile(shapeless: true) { ind, grid in
        let a = -0.75
        let x = abs(ind - grid)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    }
}()

private let compiledGetWeight2: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    // PyTorch uses -0.5 for antialiasing=true (compatibility with PIL)
    // and uses -0.75 for antialiasing=false (compatibility with OpenCV)

    compile(shapeless: true) { ind, grid in
        let a = -0.75
        let x = abs(ind - grid)
        return (((x - 5) * x + 8) * x - 4) * a
    }
}()

private func cubicIndices(dimension: Int, scale: Float, alignCorners: Bool, dim: Int, ndim: Int)
    -> [IndexWeight]
{
    let indices = scaledIndices(
        dimension: dimension, scale: scale, alignCorners: alignCorners, dim: dim, ndim: ndim)

    var indicesL1 = floor(indices)
    var indicesR1 = floor(indices + 1)
    var indicesL2 = indicesL1 - 1
    var indicesR2 = indicesR1 + 1

    let weightL1 = compiledGetWeight1(indices, indicesL1)[.ellipsis, .newAxis]
    let weightR1 = compiledGetWeight1(indices, indicesR1)[.ellipsis, .newAxis]
    let weightL2 = compiledGetWeight2(indices, indicesL2)[.ellipsis, .newAxis]
    let weightR2 = compiledGetWeight2(indices, indicesR2)[.ellipsis, .newAxis]

    // padding with border value
    indicesL1 = clip(indicesL1, min: 0, max: dimension - 1)
    indicesR1 = clip(indicesR1, min: 0, max: dimension - 1)
    indicesL2 = clip(indicesL2, min: 0, max: dimension - 1)
    indicesR2 = clip(indicesR2, min: 0, max: dimension - 1)

    return [
        (indicesL1.asType(.int32), weightL1),
        (indicesR1.asType(.int32), weightR1),
        (indicesL2.asType(.int32), weightL2),
        (indicesR2.asType(.int32), weightR2),
    ]
}

private func scaledIndices(dimension N: Int, scale: Float, alignCorners: Bool, dim: Int, ndim: Int)
    -> MLXArray
{
    let M = Int(scale * Float(N))

    var indices: MLXArray
    if alignCorners {
        indices = arange(M, dtype: .float32) * ((Float(N) - 1) / (Float(M) - 1))
    } else {
        let step = 1 / scale
        let start = ((Float(M) - 1) * step - Float(N) + 1) / 2
        indices = arange(M, dtype: .float32) * step - start
    }

    var shape = Array(repeating: 1, count: ndim)
    shape[dim] = -1

    return indices.reshaped(shape)
}
