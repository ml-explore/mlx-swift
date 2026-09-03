import Cmlx
import Foundation

@usableFromInline
final internal class UncheckedSendableBox<T>: @unchecked Sendable {
    @usableFromInline
    let value: T

    @usableFromInline
    init(_ value: T) {
        self.value = value
    }
}

/// Internal vmap implementation.  This takes a closure with sendability erased and produces
/// a (declared) sendable closure.  Callers must declare their return sendability correctly.
/// For example, ``vmap(_:inAxes:outAxes:)`` produces
/// a non-Sendable closure from a non-Sendable input while
/// ``vmapPure(_:inAxes:outAxes:)`` produces
/// Sendable from Sendable.
private func vmapInternal(
    _ f: @escaping ([MLXArray]) -> [MLXArray],
    inAxes: some Sequence<Int?> & Sendable = [0],
    outAxes: some Sequence<Int?> & Sendable = [0]
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    let box = UncheckedSendableBox(f)
    return { arrays in
        let inAxes32 = inAxes.map { Int32($0 ?? -1) }
        let outAxes32 = outAxes.map { Int32($0 ?? -1) }

        let inputs = new_mlx_vector_array(arrays)
        defer { mlx_vector_array_free(inputs) }

        var traceInputs = mlx_vector_array_new()
        var traceOutputs = mlx_vector_array_new()

        withEvalLock {
            let closure = new_mlx_closure(box.value)
            _ = inAxes32.withUnsafeBufferPointer { inAxesBuf in
                mlx_detail_vmap_trace(
                    &traceInputs, &traceOutputs, closure, inputs, inAxesBuf.baseAddress,
                    inAxesBuf.count
                )
            }
            mlx_closure_free(closure)
        }

        defer {
            mlx_vector_array_free(traceInputs)
            mlx_vector_array_free(traceOutputs)
        }

        var result = mlx_vector_array_new()
        _ = inAxes32.withUnsafeBufferPointer { inAxesBuf in
            outAxes32.withUnsafeBufferPointer { outAxesBuf in
                mlx_detail_vmap_replace(
                    &result,
                    inputs,
                    traceInputs,
                    traceOutputs,
                    inAxesBuf.baseAddress,
                    inAxesBuf.count,
                    outAxesBuf.baseAddress,
                    outAxesBuf.count
                )
            }
        }

        defer { mlx_vector_array_free(result) }
        return mlx_vector_array_values(result)
    }
}

/// Returns a vectorized version of `f()`.
///
/// The returned function applies `f()` independently over the axis
/// specified by `inAxes` and stacks the results along `outAxes`.
///
/// - Parameters:
///   - f: Function to vectorize
///   - inAxes: Axis of each input to map over. `nil` disables mapping for that input.
///   - outAxes: Axis of each output to stack the results along
/// - Returns: A vectorized function
///
/// ### See Also
/// - <doc:vmap>
public func vmap(
    _ f: @escaping ([MLXArray]) -> [MLXArray],
    inAxes: some Sequence<Int?> & Sendable = [0],
    outAxes: some Sequence<Int?> & Sendable = [0]
) -> ([MLXArray]) -> [MLXArray] {
    vmapInternal(f, inAxes: inAxes, outAxes: outAxes)
}

/// Returns a vectorized version of `f()`, a pure function (`Sendable`).
///
/// The returned function applies `f()` independently over the axis
/// specified by `inAxes` and stacks the results along `outAxes`.
///
/// ```swift
/// @Sendable
/// func add(_ x: MLXArray, _ y: MLXArray) -> MLXArray { x + y }
/// let vf = vmapPure(add, inAxes: (0, nil))
/// ```
///
/// - Parameters:
///   - f: Function to vectorize
///   - inAxes: Axis of each input to map over. `nil` disables mapping for that input.
///   - outAxes: Axis of each output to stack the results along
/// - Returns: A vectorized function
///
/// ### See Also
/// - <doc:vmap>
public func vmapPure(
    _ f: @escaping @Sendable ([MLXArray]) -> [MLXArray],
    inAxes: some Sequence<Int?> & Sendable = [0],
    outAxes: some Sequence<Int?> & Sendable = [0]
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    vmapInternal(f, inAxes: inAxes, outAxes: outAxes)
}

/// Overload of ``vmap(_:inAxes:outAxes:)`` for a single ``MLXArray`` input and
/// output.
///
/// ### See Also
/// - <doc:vmap>
/// - ``vmap(_:inAxes:outAxes:)``
public func vmap(
    _ f: @escaping (MLXArray) -> MLXArray,
    inAxes: Int? = 0,
    outAxes: Int? = 0
) -> (MLXArray) -> MLXArray {
    let inner = vmap({ [f($0[0])] }, inAxes: [inAxes], outAxes: [outAxes])
    return { a in inner([a])[0] }
}

/// Overload of ``vmapPure(_:inAxes:outAxes:)``
/// for a single ``MLXArray`` input and output.
///
/// ### See Also
/// - <doc:vmap>
public func vmapPure(
    _ f: @escaping @Sendable (MLXArray) -> MLXArray,
    inAxes: Int? = 0,
    outAxes: Int? = 0
) -> @Sendable (MLXArray) -> MLXArray {
    let inner = vmapPure({ [f($0[0])] }, inAxes: [inAxes], outAxes: [outAxes])
    return { a in inner([a])[0] }
}

/// Overload of ``vmap(_:inAxes:outAxes:)`` for two ``MLXArray`` inputs and a
/// single ``MLXArray`` output.
///
/// ### See Also
/// - <doc:vmap>
/// - ``vmap(_:inAxes:outAxes:)``
public func vmap(
    _ f: @escaping (MLXArray, MLXArray) -> MLXArray,
    inAxes: (Int?, Int?) = (0, 0),
    outAxes: Int? = 0
) -> (MLXArray, MLXArray) -> MLXArray {
    let inner = vmap({ [f($0[0], $0[1])] }, inAxes: [inAxes.0, inAxes.1], outAxes: [outAxes])
    return { a, b in inner([a, b])[0] }
}

/// Overload of ``vmapPure(_:inAxes:outAxes:)``
/// for a two ``MLXArray`` inputs and a single output.
///
/// ### See Also
/// - <doc:vmap>
public func vmapPure(
    _ f: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray,
    inAxes: (Int?, Int?) = (0, 0),
    outAxes: Int? = 0
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    let inner = vmapPure({ [f($0[0], $0[1])] }, inAxes: [inAxes.0, inAxes.1], outAxes: [outAxes])
    return { a, b in inner([a, b])[0] }
}
