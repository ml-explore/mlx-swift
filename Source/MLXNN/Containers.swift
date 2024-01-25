import Foundation
import MLX

public class Sequential: Module, UnaryModel {

    let layers: [UnaryModel]

    public init(layers: [UnaryModel]) {
        self.layers = layers
    }

    public init(layers: UnaryModel...) {
        self.layers = layers
    }

    /// ```swift
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
    public init(@SequentialBuilder layers: () -> [UnaryModel]) {
        self.layers = layers()
        print(self.layers)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            x = layer(x)
        }
        return x
    }
}

@resultBuilder
public struct SequentialBuilder {

    public static func buildArray(_ array: [UnaryModel]) -> [UnaryModel] {
        array
    }

    public static func buildArray(_ value: [[UnaryModel]]) -> [UnaryModel] {
        value.flatMap { $0 }
    }

    public static func buildExpression(_ value: UnaryModel) -> [UnaryModel] {
        [value]
    }

    public static func buildPartialBlock(accumulated: [UnaryModel], next: [UnaryModel])
        -> [UnaryModel]
    {
        accumulated + next
    }

    public static func buildPartialBlock(first: [UnaryModel]) -> [UnaryModel] {
        first
    }

    public static func buildEither(first: [UnaryModel]) -> [UnaryModel] {
        first
    }

    public static func buildEither(second: [UnaryModel]) -> [UnaryModel] {
        second
    }

    public static func buildOptional(_ component: [UnaryModel]?) -> [UnaryModel] {
        [Identity()]
    }
}
