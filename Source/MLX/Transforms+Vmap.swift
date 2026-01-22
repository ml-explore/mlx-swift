import Cmlx
import Foundation

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
    inAxes: some Sequence<Int?> = [0],
    outAxes: some Sequence<Int?> = [0]
) -> ([MLXArray]) -> [MLXArray] {
    { arrays in
        let inAxes32 = inAxes.map { Int32($0 ?? -1) }
        let outAxes32 = outAxes.map { Int32($0 ?? -1) }

        let inputs = new_mlx_vector_array(arrays)
        defer { mlx_vector_array_free(inputs) }

        var traceInputs = mlx_vector_array_new()
        var traceOutputs = mlx_vector_array_new()

        evalLock.withLock {
            let closure = new_mlx_closure(f)
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
