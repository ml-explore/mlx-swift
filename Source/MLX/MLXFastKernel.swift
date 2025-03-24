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

extension MLXFast {

    /// Container for a kernel created by
    /// ``metalKernel(name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:)``
    ///
    /// The ``callAsFunction(_:stream:)`` can be used to evaluate the kernel with inputs:
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
    ///     """,
    ///     grid: (4, 1, 1),
    ///     threadGroup: (2, 1, 1),
    ///     outputShapes: [[2, 2]],
    ///     outputDTypes: [.float32])
    ///
    /// let out = kernel([a])
    /// ```
    open class MLXFastKernel {
        let kernel: mlx_fast_metal_kernel
        public let outputNames: [String]

        init(
            name: String, inputNames: [String], outputNames: [String],
            source: String, header: String = "",
            ensureRowContiguous: Bool = true,
            atomicOutputs: Bool = false,
            template: [(String, KernelTemplateArg)]? = nil,
            grid: (Int, Int, Int),
            threadGroup: (Int, Int, Int),
            outputShapes: [[Int]],
            outputDTypes: [DType],
            initValue: Float? = nil,
            verbose: Bool = false
        ) {
            self.outputNames = outputNames

            self.kernel = mlx_fast_metal_kernel_new(
                name.cString(using: .utf8),
                source.cString(using: .utf8),
                header.cString(using: .utf8))

            for name in inputNames {
                mlx_fast_metal_kernel_add_input_name(kernel, name)
            }
            for name in outputNames {
                mlx_fast_metal_kernel_add_output_name(kernel, name)
            }

            mlx_fast_metal_kernel_set_contiguous_rows(kernel, ensureRowContiguous)
            mlx_fast_metal_kernel_set_atomic_outputs(kernel, atomicOutputs)

            if let template {
                for (name, arg) in template {
                    switch arg {
                    case let value as Bool:
                        mlx_fast_metal_kernel_add_template_arg_bool(kernel, name, value)

                    case let value as Int:
                        mlx_fast_metal_kernel_add_template_arg_int(kernel, name, Int32(value))

                    case let value as DType:
                        mlx_fast_metal_kernel_add_template_arg_dtype(kernel, name, value.cmlxDtype)

                    default:
                        fatalError(
                            "Unable to handle KernelTemplateArg \(name) with type: \(type(of: arg))."
                        )
                    }
                }
            }

            mlx_fast_metal_kernel_set_grid(kernel, Int32(grid.0), Int32(grid.1), Int32(grid.2))
            mlx_fast_metal_kernel_set_thread_group(
                kernel, Int32(threadGroup.0), Int32(threadGroup.1), Int32(threadGroup.2))

            for (shape, dtype) in zip(outputShapes, outputDTypes) {
                mlx_fast_metal_kernel_add_output_arg(
                    kernel, shape.map { Int32($0) }, shape.count, dtype.cmlxDtype)
            }

            if let initValue {
                mlx_fast_metal_kernel_set_init_value(kernel, initValue)
            }

            mlx_fast_metal_kernel_set_verbose(kernel, verbose)
        }

        deinit {
            mlx_fast_metal_kernel_free(kernel)
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
            _ inputs: [ScalarOrArray],
            stream: StreamOrDevice = .default
        ) -> [MLXArray] {
            let inputs = new_mlx_vector_array(inputs.map { $0.asMLXArray(dtype: nil) })
            defer { mlx_vector_array_free(inputs) }

            var result = mlx_vector_array_new()
            mlx_fast_metal_kernel_apply(&result, kernel, inputs, stream.ctx)
            defer { mlx_vector_array_free(result) }

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
    public static func metalKernel(
        name: String, inputNames: [String], outputNames: [String],
        source: String, header: String = "",
        ensureRowContiguous: Bool = true,
        atomicOutputs: Bool = false,
        template: [(String, KernelTemplateArg)]? = nil,
        grid: (Int, Int, Int),
        threadGroup: (Int, Int, Int),
        outputShapes: [[Int]],
        outputDTypes: [DType],
        initValue: Float? = nil,
        verbose: Bool = false
    ) -> MLXFastKernel {
        MLXFastKernel(
            name: name, inputNames: inputNames, outputNames: outputNames,
            source: source, header: header,
            ensureRowContiguous: ensureRowContiguous, atomicOutputs: atomicOutputs,
            template: template, grid: grid, threadGroup: threadGroup,
            outputShapes: outputShapes, outputDTypes: outputDTypes,
            initValue: initValue, verbose: verbose
        )
    }

}  // MLXFast
