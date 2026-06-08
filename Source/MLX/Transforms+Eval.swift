// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// lock to be held while doing any eval or asyncEval.  This is
/// a recursive lock to handle any cases where a closure might
/// call back into eval.
let evalLock = NSRecursiveLock()

/// Evaluate `array` and return a ``MaterializedArray`` snapshot of its contents.
///
/// The returned array is fully evaluated, immutable, and `Sendable`, so it can
/// be passed across task and actor boundaries.  See ``MaterializedArray`` for
/// the full set of guarantees.
///
/// `array` itself is unaffected — it remains the same lazy ``MLXArray`` it was
/// before the call — but because evaluation is forced, any pending graph work
/// behind it has been realized as a side effect.
///
/// ### See Also
/// - ``MaterializedArray``
/// - ``MLXArray/materialized()``
/// - ``materialize(_:)``
public func materialize(_ array: MLXArray) -> MaterializedArray {
    eval(array)
    var m = mlx_array_new()
    mlx_array_set(&m, array.ctx)
    return MaterializedArray(materialized: m)
}

/// Evaluate `arrays` and return a ``MaterializedArray`` snapshot for each one.
///
/// Equivalent to calling ``materialize(_:)`` on each element, but evaluates the
/// whole batch in a single ``eval(_:)-(Sequence<Any>)`` call so MLX can schedule the
/// work together.  Prefer this overload when you need to materialize several
/// arrays at once — for example, the parameters of a model.
///
/// ### See Also
/// - ``MaterializedArray``
/// - ``materialize(_:)``
public func materialize(_ arrays: [MLXArray]) -> [MaterializedArray] {
    eval(arrays)
    return arrays.map {
        var m = mlx_array_new()
        mlx_array_set(&m, $0.ctx)
        return MaterializedArray(materialized: m)
    }
}

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func eval(_ arrays: MLXArray...) {
    let vector_array = new_mlx_vector_array(arrays)
    _ = evalLock.withLock {
        mlx_eval(vector_array)
    }
    mlx_vector_array_free(vector_array)
}

/// Evaluate one or more `MLXArray`
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func eval(_ arrays: some Collection<MLXArray>) {
    let vector_array = new_mlx_vector_array(arrays)
    _ = evalLock.withLock {
        mlx_eval(vector_array)
    }
    mlx_vector_array_free(vector_array)
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// ### See Also
/// - <doc:lazy-evaluation>
/// - ``asyncEval(_:)-(Collection<MLXArray>)``
public func asyncEval(_ arrays: some Collection<MLXArray>) {
    let vector_array = new_mlx_vector_array(arrays)
    _ = evalLock.withLock {
        mlx_async_eval(vector_array)
    }
    mlx_vector_array_free(vector_array)
}

/// Evaluate one or more `MLXArray`.
///
/// This variant allows several structured types:
///
/// ```swift
/// let a: MLXArray
/// let b: [MLXArray]
/// let c: [String:MLXArray]
/// let d: [String:[MLXArray]]
/// let e: (MLXArray, MLXArray)
/// let f: [(String, MLXArray)]
/// let nested: [(MLXArray, [MLXArray])]
///
/// eval(a, b, c, d, e, f)
/// ```
///
/// Other structured types may be supported -- check the implementation.
///
/// ### See Also
/// - <doc:lazy-evaluation>
/// - ``asyncEval(_:)-(Collection<MLXArray>)``
public func eval(_ values: Any...) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    eval(arrays)
}

/// Evaluate one or more `MLXArray`.
///
/// See ``eval(_:)``
public func eval(_ values: some Sequence<Any>) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    eval(arrays)
}

/// Variant of ``eval(_:)-(Collection<MLXArray>)`` that checks for errors in MLX and throws.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func checkedEval(_ values: Any...) throws {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    try withError {
        eval(arrays)
    }
}

/// Variant of ``eval(_:)-(MLXArray...)`` that checks for errors in MLX and throws.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func checkedEval(_ values: some Sequence<Any>) throws {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    try withError {
        eval(arrays)
    }
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// This variant allows several structured types:
///
/// ```swift
/// let a: MLXArray
/// let b: [MLXArray]
/// let c: [String:MLXArray]
/// let d: [String:[MLXArray]]
/// let e: (MLXArray, MLXArray)
/// let f: [(String, MLXArray)]
/// let nested: [(MLXArray, [MLXArray])]
///
/// asyncEval(a, b, c, d, e, f)
/// ```
///
/// Other structured types may be supported -- check the implementation.
///
/// ### See Also
/// - <doc:lazy-evaluation>
public func asyncEval(_ values: Any...) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    asyncEval(arrays)
}

/// Evaluate one or more `MLXArray` asynchronously.
///
/// See ``asyncEval(_:)-(Collection<MLXArray>)``
public func asyncEval(_ values: some Sequence<Any>) {
    var arrays = [MLXArray]()

    for item in values {
        collect(item, into: &arrays)
    }

    asyncEval(arrays)
}

private func collect(_ item: Any, into arrays: inout [MLXArray]) {
    switch item {
    case let v as Evaluatable:
        arrays.append(contentsOf: v.innerState())

    case let v as NestedDictionary<String, MLXArray>:
        arrays.append(contentsOf: v.flattened().map { $0.1 })

    case let v as MLXArray:
        arrays.append(v)
    case let v as [MLXArray]:
        arrays.append(contentsOf: v)
    case let v as [Any]:
        for item in v {
            collect(item, into: &arrays)
        }
    case let v as [AnyHashable: Any]:
        for item in v.values {
            collect(item, into: &arrays)
        }
    case let v as (Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
    case let v as (Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
    case let v as (Any, Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
        collect(v.3, into: &arrays)
    case let v as (Any, Any, Any, Any, Any):
        collect(v.0, into: &arrays)
        collect(v.1, into: &arrays)
        collect(v.2, into: &arrays)
        collect(v.3, into: &arrays)
        collect(v.4, into: &arrays)
    case is String, is any BinaryInteger, is any BinaryFloatingPoint:
        // ignore, e.g. (String, MLXArray)
        break
    default:
        fatalError("Unable to extract MLXArray from \(item)")
    }
}
