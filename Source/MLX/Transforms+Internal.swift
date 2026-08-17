// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Synchronization

// see Transforms+Variants for generated grad() functions

/// Copy the backing MLX contexts in a parameter tree so later `_updateInternal` calls on the
/// original wrappers cannot mutate the snapshot. Used by MLXNN to restore a model after tracing.
@_documentation(visibility: internal)
public func _snapshotArrayContexts(
    _ parameters: NestedDictionary<String, MLXArray>
) -> NestedDictionary<String, MLXArray> {
    parameters.mapValues { $0.copyContext() }
}

/// Owns an MLX value-and-gradient transform for the lifetime of the Swift function returned to
/// the caller. The transform is immutable after construction and applications are serialized by
/// `evalLock`, so it is safe to reuse across calls and threads.
private final class ValueAndGradientTransform {
    private let transform: mlx_closure_value_and_grad

    init?(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: some Collection<Int>) {
        var transform = mlx_closure_value_and_grad_new()
        let closure = new_mlx_closure(f)
        let argumentNumbers = argumentNumbers.asInt32
        let status = evalLock.withLock {
            let status = mlx_value_and_grad(
                &transform, closure, argumentNumbers, argumentNumbers.count)
            mlx_closure_free(closure)
            return status
        }

        guard status == 0 else {
            mlx_closure_value_and_grad_free(transform)
            return nil
        }
        self.transform = transform
    }

    deinit {
        _ = evalLock.withLock {
            mlx_closure_value_and_grad_free(transform)
        }
    }

    func call(_ arrays: some Collection<MLXArray>) -> ([MLXArray], [MLXArray]) {
        let inputVector = new_mlx_vector_array(arrays)
        defer { mlx_vector_array_free(inputVector) }

        var values = mlx_vector_array_new()
        var gradients = mlx_vector_array_new()
        defer { mlx_vector_array_free(values) }
        defer { mlx_vector_array_free(gradients) }

        let status = evalLock.withLock {
            mlx_closure_value_and_grad_apply(&values, &gradients, transform, inputVector)
        }
        guard status == 0 else {
            return ([], [])
        }

        return (mlx_vector_array_values(values), mlx_vector_array_values(gradients))
    }
}

func buildGradient(_ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: some Collection<Int>)
    -> (
        [MLXArray]
    ) -> [MLXArray]
{
    guard let transform = ValueAndGradientTransform(f, argumentNumbers: argumentNumbers) else {
        return { _ in [] }
    }

    return { (arrays: [MLXArray]) in
        transform.call(arrays).1
    }
}

func buildValueAndGradient(
    _ f: @escaping ([MLXArray]) -> [MLXArray], argumentNumbers: some Collection<Int>
) -> (
    [MLXArray]
) -> ([MLXArray], [MLXArray]) {
    guard let transform = ValueAndGradientTransform(f, argumentNumbers: argumentNumbers) else {
        return { _ in ([], []) }
    }

    return { (arrays: [MLXArray]) in
        transform.call(arrays)
    }
}

/// Caches the transform used by the nested-parameter plus array-input overload. The parameter
/// topology is part of the transform because its tracer closure reconstructs that topology; a
/// topology change therefore creates a fresh transform while ordinary value/shape changes reuse
/// the existing one.
private final class NestedArrayValueAndGradientTransform {
    typealias Parameters = NestedDictionary<String, MLXArray>

    let f: (Parameters, [MLXArray]) -> [MLXArray]
    let lock = Mutex(())

    private var topology: NestedDictionary<String, Bool>?
    private var transform: ValueAndGradientTransform?

    init(_ f: @escaping (Parameters, [MLXArray]) -> [MLXArray]) {
        self.f = f
    }

    func call(_ parameters: Parameters, _ extraArrays: [MLXArray]) -> ([MLXArray], Parameters) {
        var result: ([MLXArray], Parameters) = ([], parameters)
        lock.withLock { _ in
            result = innerCall(parameters, extraArrays)
        }
        return result
    }

    private func innerCall(
        _ parameters: Parameters, _ extraArrays: [MLXArray]
    ) -> ([MLXArray], Parameters) {
        let currentTopology = parameters.mapValues { _ in false }
        let flattenedParameters = parameters.flattenedValues()

        if topology != currentTopology || transform == nil {
            let parameterCount = flattenedParameters.count
            let parameterTemplate = parameters
            let f = f

            transform = ValueAndGradientTransform(
                { inputs in
                    let flatParameters = Array(inputs.prefix(parameterCount))
                    let parameters = parameterTemplate.replacingValues(with: flatParameters)
                    let extras = Array(inputs.dropFirst(parameterCount))
                    return f(parameters, extras)
                }, argumentNumbers: 0 ..< parameterCount)
            topology = transform == nil ? nil : currentTopology
        }

        guard let transform else {
            return ([], parameters)
        }

        let (values, flatGradients) = transform.call(flattenedParameters + extraArrays)
        let gradients = parameters.replacingValues(with: flatGradients)
        return (values, gradients)
    }
}

func buildValueAndGradient(
    _ f: @escaping (NestedDictionary<String, MLXArray>, [MLXArray]) -> [MLXArray]
) -> (NestedDictionary<String, MLXArray>, [MLXArray]) -> (
    [MLXArray], NestedDictionary<String, MLXArray>
) {
    let transform = NestedArrayValueAndGradientTransform(f)
    return { parameters, arrays in
        transform.call(parameters, arrays)
    }
}

func buildValueAndGradient<T>(
    _ f: @escaping (NestedDictionary<String, MLXArray>, T) -> [MLXArray]
) -> (NestedDictionary<String, MLXArray>, T) -> (
    [MLXArray], NestedDictionary<String, MLXArray>
) {
    {
        (parameters: NestedDictionary<String, MLXArray>, arrays: T) -> (
            [MLXArray], NestedDictionary<String, MLXArray>
        ) in

        let flattenedArrays = parameters.flattenedValues()

        // this goes in the closure and is wrapped by mlx_value_and_grad
        //
        // Note: we pass the flattened array through the grad but
        // we capture the extra arrays used as arguments (matching
        // the python implementation).
        //
        // Potentially this could pass all the values and use the
        // arg indexes to indicate which ones to grad (it should work
        // as is)
        func inner(flattenedArrays: [MLXArray]) -> [MLXArray] {
            let parameters = parameters.replacingValues(with: flattenedArrays)
            return f(parameters, arrays)
        }

        guard let transform = ValueAndGradientTransform(
            inner, argumentNumbers: 0 ..< flattenedArrays.count)
        else {
            return ([], parameters)
        }

        let (values, flatGradients) = transform.call(flattenedArrays)
        let gradients = parameters.replacingValues(with: flatGradients)

        return (values, gradients)
    }
}
