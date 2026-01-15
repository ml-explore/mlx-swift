// Copyright © 2024 Apple Inc.

import Cmlx

/// Marker protocol for types that can be used in the `template` of a kernel call.
///
/// Currently:
/// - `Int`
/// - `Bool`
/// - `DType`
///
/// See also: ``MLXFast/MLXFastKernel``
public protocol KernelTemplateArg {}

extension Bool: KernelTemplateArg {}
extension Int: KernelTemplateArg {}
extension DType: KernelTemplateArg {}

extension MLXFast {

    /// Container for a kernel created by
    /// ``callAsFunction(_:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:stream:)``
    ///
    /// The ``callAsFunction(_:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:stream:)`` can be used to evaluate the kernel with inputs:
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
    final public class MLXFastKernel: @unchecked Sendable {
        let kernel: mlx_fast_metal_kernel
        public let outputNames: [String]

        init(
            name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
            source: String, header: String = "",
            ensureRowContiguous: Bool = true,
            atomicOutputs: Bool = false
        ) {
            self.outputNames = Array(outputNames)

            let input_names = mlx_vector_string_new()
            defer { mlx_vector_string_free(input_names) }
            for name in inputNames {
                mlx_vector_string_append_value(input_names, name)
            }

            let output_names = mlx_vector_string_new()
            defer { mlx_vector_string_free(output_names) }
            for name in self.outputNames {
                mlx_vector_string_append_value(output_names, name)
            }

            self.kernel = mlx_fast_metal_kernel_new(
                name.cString(using: .utf8),
                input_names, output_names,
                source.cString(using: .utf8),
                header.cString(using: .utf8),
                ensureRowContiguous, atomicOutputs)
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
            _ inputs: [any ScalarOrArray],
            template: [(String, any KernelTemplateArg)]? = nil,
            grid: (Int, Int, Int),
            threadGroup: (Int, Int, Int),
            outputShapes: some Sequence<[Int]>,
            outputDTypes: some Sequence<DType>,
            initValue: Float? = nil,
            verbose: Bool = false,
            stream: StreamOrDevice = .default
        ) -> [MLXArray] {
            let config = mlx_fast_metal_kernel_config_new()
            defer { mlx_fast_metal_kernel_config_free(config) }

            if let template {
                for (name, arg) in template {
                    switch arg {
                    case let value as Bool:
                        mlx_fast_metal_kernel_config_add_template_arg_bool(config, name, value)

                    case let value as Int:
                        mlx_fast_metal_kernel_config_add_template_arg_int(
                            config, name, Int32(value))

                    case let value as DType:
                        mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            config, name, value.cmlxDtype)

                    default:
                        fatalError(
                            "Unable to handle KernelTemplateArg \(name) with type: \(type(of: arg))."
                        )
                    }
                }
            }

            mlx_fast_metal_kernel_config_set_grid(config, grid.0.int32, grid.1.int32, grid.2.int32)
            mlx_fast_metal_kernel_config_set_thread_group(
                config, threadGroup.0.int32, threadGroup.1.int32, threadGroup.2.int32)

            for (shape, dtype) in zip(outputShapes, outputDTypes) {
                mlx_fast_metal_kernel_config_add_output_arg(
                    config, shape.map { Int32($0) }, shape.count, dtype.cmlxDtype)
            }

            if let initValue {
                mlx_fast_metal_kernel_config_set_init_value(config, initValue)
            }

            mlx_fast_metal_kernel_config_set_verbose(config, verbose)

            let inputs = new_mlx_vector_array(inputs.map { $0.asMLXArray(dtype: nil) })
            defer { mlx_vector_array_free(inputs) }

            var result = mlx_vector_array_new()
            mlx_fast_metal_kernel_apply(&result, kernel, inputs, config, stream.ctx)
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
        name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
        source: String, header: String = "", ensureRowContiguous: Bool = true,
        atomicOutputs: Bool = false
    ) -> MLXFastKernel {
        MLXFastKernel(
            name: name, inputNames: inputNames, outputNames: outputNames,
            source: source, header: header,
            ensureRowContiguous: ensureRowContiguous, atomicOutputs: atomicOutputs
        )
    }

}  // MLXFast
