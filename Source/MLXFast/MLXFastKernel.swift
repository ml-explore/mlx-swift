// Copyright © 2024 Apple Inc.

import Cmlx
import MLX

/// Marker protocol for types that can be used in the `template` of a kernel call.
///
/// Currently:
/// - `Int`
/// - `Bool`
/// - `DType`
///
/// See also: ``MLXFastKernel``
public protocol KernelTemplateArg {}

extension Bool: KernelTemplateArg {}
extension Int: KernelTemplateArg {}
extension DType: KernelTemplateArg {}

/// Add a ``KernelTemplateArg`` to the tuple of vectors
private func add(
    name: String,
    value: any KernelTemplateArg,
    to vector: mlx_vector_tuple_string_variant_int_bool_array_dtype
) {
    let name = mlx_string_new(name.cString(using: .utf8))!
    defer { mlx_free(name) }

    let value =
        switch value {
        case let value as Bool:
            mlx_variant_int_bool_array_dtype_new_with_bool(value)!

        case let value as Int:
            mlx_variant_int_bool_array_dtype_new_with_int(Int32(value))!

        case let value as DType:
            mlx_variant_int_bool_array_dtype_new_with_array_dtype(value.cmlxDtype)!

        default:
            fatalError("Unable to handle KernelTemplateArg with type: \(type(of: value)).")
        }

    defer { mlx_free(value) }

    let tuple = mlx_tuple_string_variant_int_bool_array_dtype_new(name, value)!
    defer { mlx_free(tuple) }

    mlx_vector_tuple_string_variant_int_bool_array_dtype_add_value(vector, tuple)
}

/// Container for a kernel created by
/// ``metalKernel(name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:)``.
///
/// The ``callAsFunction(inputs:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:stream:)``
/// can be used to evaluate the kernel with inputs:
///
/// ```swift
/// let a = normal([2, 2])
/// let kernel = MLXFast.metalKernel(
///     name: "basic",
///     inputNames: ["a"],
///     outputNames: ["out1"],
///     source: """
///         uint elem = thread_position_in_grid.x;
///         out1[elem] = a[elem];
///     """)
///
/// let out = kernel(
///     inputs: [a],
///     grid: (4, 1, 1),
///     threadGroup: (2, 1, 1),
///     outputShapes: [[2, 2]],
///     outputDTypes: [.float32])
/// ```
open class MLXFastKernel {
    let kernel: mlx_closure_metal_kernel_function
    public let outputNames: [String]

    init(
        name: String, inputNames: [String], outputNames: [String],
        source: String, header: String = "",
        ensureRowContiguous: Bool = true,
        atomicOutputs: Bool = false
    ) {
        self.outputNames = outputNames

        let mlxName = mlx_string_new(name.cString(using: .utf8))!
        defer { mlx_free(mlxName) }

        let mlxInputNames = new_mlx_vector_string(inputNames)
        defer { mlx_free(mlxInputNames) }
        let mlxOutputNames = new_mlx_vector_string(outputNames)
        defer { mlx_free(mlxOutputNames) }

        let mlxSource = mlx_string_new(source.cString(using: .utf8))!
        defer { mlx_free(mlxSource) }
        let mlxHeader = mlx_string_new(header.cString(using: .utf8))!
        defer { mlx_free(mlxHeader) }

        self.kernel = mlx_fast_metal_kernel(
            mlxName, mlxInputNames, mlxOutputNames, mlxSource, mlxHeader, ensureRowContiguous,
            atomicOutputs)
    }

    deinit {
        mlx_free(kernel)
    }

    /// Call the prepared metal kernel.
    ///
    /// See ``MLXFastKernel`` for example.  Use
    /// ``metalKernel(name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:)``
    /// to create an instance.
    ///
    /// - Parameters:
    ///   - inputs: inputs passed to the metal kernel
    ///   - template: template arguments
    ///   - grid: 3-tuple specifying the grid to launch the kernel with
    ///   - threadGroup: 3-tuple specifying the threadgroup size to use
    ///   - outputShapes: list of shapes for each output in ``outputNames``
    ///   - outputDTypes: list of data types for each output in ``outputNames``
    ///   - initValue: optional value to use to initialize all of the output arrays
    ///   - verbose: if true will print the full generated source code of the kernel when run
    ///   - stream: stream to run on
    /// - Returns: array of `MLXArray`
    public func callAsFunction(
        inputs: [ScalarOrArray],
        template: [(String, KernelTemplateArg)]? = nil,
        grid: (Int, Int, Int),
        threadGroup: (Int, Int, Int),
        outputShapes: [[Int]],
        outputDTypes: [DType],
        initValue: Float? = nil,
        verbose: Bool = false,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        // convert all the inputs into the mlx-c types
        let inputs = new_mlx_vector_array(inputs.map { $0.asMLXArray(dtype: nil) })
        defer { mlx_free(inputs) }

        let outputShapes = new_mlx_vector_vector_int(outputShapes)
        defer { mlx_free(outputShapes) }

        let outputDTypes = new_mlx_vector_array_dtype(outputDTypes)
        defer { mlx_free(outputDTypes) }

        let grid = mlx_tuple_int_int_int_new(Int32(grid.0), Int32(grid.1), Int32(grid.2))!
        defer { mlx_free(grid) }

        let threadGroup = mlx_tuple_int_int_int_new(
            Int32(threadGroup.0), Int32(threadGroup.1), Int32(threadGroup.2))!
        defer { mlx_free(threadGroup) }

        let templateVector = mlx_vector_tuple_string_variant_int_bool_array_dtype_new()!
        defer { mlx_free(templateVector) }
        if let template {
            for (name, value) in template {
                add(name: name, value: value, to: templateVector)
            }
        }

        let initValue = mlx_optional_float(value: initValue ?? 0, has_value: initValue != nil)

        let result = mlx_closure_metal_kernel_function_apply(
            kernel,
            inputs,
            outputShapes,
            outputDTypes,
            grid,
            threadGroup,
            templateVector,
            initValue,
            verbose,
            stream.ctx)!
        defer { mlx_free(result) }

        return mlx_vector_array_values(result)
    }
}

/// A jit-compiled custom Metal kernel defined from a source string.
///
/// - Parameters:
///   - name: name for the kernel
///   - inputNames: parameter names of the inputs in the function signature
///   - outputNames: parameter names of the outputs in the function signature
///   - source: source code -- this is the body of a function in Metal,
///   the function signature will be automatically generated.
///   - header: header source code to include before the main function.  Useful
///   for helper functions or includes that should live outside of the main function body.
///   - ensureRowContiguous: whether to ensure the inputs are row contiguous
///   before the kernel runs (at a performance cost)
///   - atomicOutputs: whether to use atomic outputs in the function signature,
///   e.g. `device atomic<float>`
/// - Returns: an ``MLXFastKernel`` -- see that for information on how to call it
public func metalKernel(
    name: String, inputNames: [String], outputNames: [String],
    source: String, header: String = "",
    ensureRowContiguous: Bool = true,
    atomicOutputs: Bool = false
) -> MLXFastKernel {
    MLXFastKernel(
        name: name, inputNames: inputNames, outputNames: outputNames,
        source: source, header: header,
        ensureRowContiguous: ensureRowContiguous, atomicOutputs: atomicOutputs)
}
