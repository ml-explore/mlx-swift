// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// A layer that calls the passed ``UnaryLayer`` in order.
///
/// `Sequential` can be constructed either with an array of layers or using a ``SequentialBuilder``:
///
/// ```swift
/// // a nonsensical Sequential layer, but it demonstrates how
/// // to constructuct something with an interesting structure
/// let b = Bool.random()
/// let s = Sequential {
///     Tanh()
///     if b {
///         Tanh()
///     } else {
///         Sigmoid()
///     }
///     for _ in 0 ..< 3 {
///         Linear(10, 20)
///     }
/// }
/// ```
///
/// produces:
///
/// ```swift
/// Sequential {
///   layers: [
///     Tanh,
///     Sigmoid,
///     Linear(inputDimensions=10, outputDimensions=20, bias=true),
///     Linear(inputDimensions=10, outputDimensions=20, bias=true),
///     Linear(inputDimensions=10, outputDimensions=20, bias=true)
///   ],
/// }
/// ```
open class Sequential: Module, UnaryLayer {

    @ModuleInfo public var layers: [UnaryLayer]

    public init(layers: [UnaryLayer]) {
        self.layers = layers
    }

    public init(layers: UnaryLayer...) {
        self.layers = layers
    }

    /// A convenient way to write code that builds a Sequential layer:
    ///
    /// ```swift
    /// // a nonsensical Sequential layer, but it demonstrates how
    /// // to constructuct something with an interesting structure
    /// let b = Bool.random()
    /// let s = Sequential {
    ///     Tanh()
    ///     if b {
    ///         Tanh()
    ///     } else {
    ///         Sigmoid()
    ///     }
    ///     for _ in 0 ..< 3 {
    ///         Linear(10, 20)
    ///     }
    /// }
    /// ```
    ///
    /// produces:
    ///
    /// ```swift
    /// Sequential {
    ///   layers: [
    ///     Tanh,
    ///     Sigmoid,
    ///     Linear(inputDimensions=10, outputDimensions=20, bias=true),
    ///     Linear(inputDimensions=10, outputDimensions=20, bias=true),
    ///     Linear(inputDimensions=10, outputDimensions=20, bias=true)
    ///   ],
    /// }
    /// ```
    public init(@SequentialBuilder layers: () -> [UnaryLayer]) {
        self.layers = layers()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            x = layer(x)
        }
        return x
    }
}

/// A way to build ``Sequential``.
///
/// See ``Sequential/init(layers:)-43yu``
@resultBuilder
public struct SequentialBuilder {

    public static func buildArray(_ array: [UnaryLayer]) -> [UnaryLayer] {
        array
    }

    public static func buildArray(_ value: [[UnaryLayer]]) -> [UnaryLayer] {
        value.flatMap { $0 }
    }

    public static func buildExpression(_ value: UnaryLayer) -> [UnaryLayer] {
        [value]
    }

    public static func buildPartialBlock(accumulated: [UnaryLayer], next: [UnaryLayer])
        -> [UnaryLayer]
    {
        accumulated + next
    }

    public static func buildPartialBlock(first: [UnaryLayer]) -> [UnaryLayer] {
        first
    }

    public static func buildEither(first: [UnaryLayer]) -> [UnaryLayer] {
        first
    }

    public static func buildEither(second: [UnaryLayer]) -> [UnaryLayer] {
        second
    }

    public static func buildOptional(_ component: [UnaryLayer]?) -> [UnaryLayer] {
        [Identity()]
    }
}
