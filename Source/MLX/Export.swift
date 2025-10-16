// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

/// Export a function to a `.mlxfn` file.
///
/// For example this defines a function, writes it to a file, imports
/// it and evaluates it.
///
/// ```swift
/// func f(arrays: [MLXArray]) -> [MLXArray] {
///     [arrays[0] * arrays[1]]
/// }
///
/// let x = MLXArray(1)
/// let y = MLXArray([1, 2, 3])
///
/// try exportFunction(to: url, f)(x, y: y)
///
/// // load it back in
/// let f2 = try importFunction(from: url)
///
/// let a = MLXArray(10)
/// let b = MLXArray([5, 10, 20])
///
/// // call it -- the shapes and labels have to match
/// let r = try f2(a, y: b)[0]
/// ```
///
/// - Parameters:
///   - url: file url to write the `.mlxfn` file
///   - shapeless: if `true` the function allows inputs with variable shapes
///   - f: the function to capture
/// - Returns: a helper (``FunctionExporterSingle``) that records the call
///
/// ### See Also
/// - ``exportFunctions(to:shapeless:_:build:)``
/// - ``importFunction(from:)``
public func exportFunction(
    to url: URL, shapeless: Bool = false, _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> FunctionExporterSingle {
    FunctionExporterSingle(url: url, shapeless: shapeless, f: f)
}

/// Export multiple traces of a function to a `.mlxfn` file.
///
/// For example this defines a function, writes it to a file, imports
/// it and evaluates it.
///
/// ```swift
/// func f(_ arrays: [MLXArray]) -> [MLXArray] {
///     [arrays.dropFirst().reduce(arrays[0], +)]
/// }
///
/// let x = MLXArray(1)
///
/// try exportFunctions(to: url, shapeless: true, f) { export in
///     try export(x)
///     try export(x, x)
///     try export(x, x, x)
/// }
///
/// // load it back in
/// let f2 = try importFunction(from: url)
///
/// let a = MLXArray([10, 10, 10])
/// let b = MLXArray([5, 10, 20])
/// let c = MLXArray([1, 2, 3])
///
/// // r1 = a
/// let r1 = try f2(a)[0]
///
/// // r2 = a + b
/// let r2 = try f2(a, b)[0]
///
/// // r3 = a + b + c
/// let r3 = try f2(a, b, c)[0]
/// ```
///
/// - Parameters:
///   - url: file url to write the `.mlxfn` file
///   - shapeless: if `true` the function allows inputs with variable shapes
///   - f: the function to capture
///   - build: closure for recording the calls
///
/// ### See Also
/// - ``exportFunction(to:shapeless:_:)``
/// - ``importFunction(from:)``
public func exportFunctions(
    to url: URL, shapeless: Bool = false, _ f: @escaping ([MLXArray]) -> [MLXArray],
    build: (FunctionExporterMultiple) throws -> Void
) throws {
    let exporter = try FunctionExporterMultiple(url: url, shapeless: shapeless, f: f)
    try build(exporter)
}

/// A helper for ``exportFunction(to:shapeless:_:)``.
///
/// This records the call to the function and saves it to the file.
///
/// ```swift
/// func f(arrays: [MLXArray]) -> [MLXArray] {
///     [arrays[0] * arrays[1]]
/// }
///
/// let x = MLXArray(1)
/// let y = MLXArray([1, 2, 3])
///
/// // the (x, y: y) is calling this object
/// try exportFunction(to: url, f)(x, y: y)
/// ```
///
/// ### See Also
/// - ``exportFunction(to:shapeless:_:)``
@dynamicCallable
public final class FunctionExporterSingle {
    let url: URL
    let shapeless: Bool
    let f: ([MLXArray]) -> [MLXArray]

    internal init(url: URL, shapeless: Bool, f: @escaping ([MLXArray]) -> [MLXArray]) {
        self.url = url
        self.shapeless = shapeless
        self.f = f
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws {
        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let keys = args.compactMap { $0.key.isEmpty ? nil : $0.key }
        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        let closure = new_mlx_kwargs_closure(keys: keys, f)
        defer { mlx_closure_kwargs_free(closure) }

        _ = try withError {
            mlx_export_function_kwargs(url.path, closure, positionalArgs, kwargs, shapeless)
        }
    }
}

/// A helper for ``exportFunctions(to:shapeless:_:build:)``.
///
/// This records the call to the function and saves it to the file.
///
/// ```swift
/// func f(_ arrays: [MLXArray]) -> [MLXArray] {
///     [arrays.dropFirst().reduce(arrays[0], +)]
/// }
///
/// let x = MLXArray(1)
///
/// // the export parameter is a FunctionExporterMultiple
/// try exportFunctions(to: url, shapeless: true, f) { export in
///     try export(x)
///     try export(x, x)
///     try export(x, x, x)
/// }
/// ```
///
/// ### See Also
/// - ``exportFunctions(to:shapeless:_:build:)``
@dynamicCallable
public final class FunctionExporterMultiple {
    let exporter: mlx_function_exporter

    internal init(url: URL, shapeless: Bool = false, f: @escaping ([MLXArray]) -> [MLXArray]) throws
    {
        let closure = new_mlx_closure(f)
        defer { mlx_closure_free(closure) }

        self.exporter = try withError {
            mlx_function_exporter_new(url.path, closure, shapeless)
        }
    }

    deinit {
        mlx_function_exporter_free(exporter)
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws {
        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        _ = try withError {
            mlx_function_exporter_apply_kwargs(exporter, positionalArgs, kwargs)
        }
    }
}

/// Imports a function from a `.mlxfn` file.
///
/// ```swift
/// // f is a callable that represents the loaded function
/// let f = try importFunction(from: url)
///
/// let a = MLXArray(10)
/// let b = MLXArray([5, 10, 20])
///
/// // call it -- the shapes and labels have to match
/// let r = try f(a, y: b)[0]
/// ```
///
/// - Parameter url: file to load from
/// - Returns: a callable that represents the loaded function
/// ### See Also
/// - ``exportFunction(to:shapeless:_:)``
/// - ``exportFunctions(to:shapeless:_:build:)``
public func importFunction(from url: URL) throws -> ImportedFunction {
    try ImportedFunction(url: url)
}

/// Helper for ``importFunction(from:)`` -- this holds the imported function.
///
/// This can be called with parameters that match the recorded parameters:
///
/// ```swift
/// func f(arrays: [MLXArray]) -> [MLXArray] {
///     [arrays[0] * arrays[1]]
/// }
///
/// let x = MLXArray(1)
/// let y = MLXArray([1, 2, 3])
///
/// // records with unnamed first parameter and a second parameter named `y`
/// try exportFunction(to: url, f)(x, y: y)
///
/// // f2 is a ImportedFunction
/// let f2 = try importFunction(from: url)
///
/// let a = MLXArray(10)
/// let b = MLXArray([5, 10, 20])
///
/// // call it -- the shapes and labels have to match
/// let r = try f2(a, y: b)[0]
/// ```
@dynamicCallable
public final class ImportedFunction {

    private let ctx: mlx_imported_function

    public init(url: URL) throws {
        self.ctx = try withError {
            mlx_imported_function_new(url.path)
        }
    }

    deinit {
        mlx_imported_function_free(ctx)
    }

    public func dynamicallyCall(withKeywordArguments args: KeyValuePairs<String, MLXArray>) throws
        -> [MLXArray]
    {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }

        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for (key, value) in args {
            if key.isEmpty {
                mlx_vector_array_append_value(positionalArgs, value.ctx)
            }
        }

        let kwargs = new_mlx_array_map(
            Dictionary(
                args.compactMap { $0.key.isEmpty ? nil : ($0.key, $0.value) },
                uniquingKeysWith: { a, b in a }))
        defer { mlx_map_string_to_array_free(kwargs) }

        _ = try withError {
            mlx_imported_function_apply_kwargs(&result, ctx, positionalArgs, kwargs)
        }

        return mlx_vector_array_values(result)
    }
}
