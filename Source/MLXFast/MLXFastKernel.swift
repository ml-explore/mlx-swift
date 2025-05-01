// Copyright © 2024 Apple Inc.

import Cmlx
import MLX

/// Container for a kernel created by
/// ``metalKernel(name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:)``
///
/// The ``MLXFast/MLXFastKernel`` can be used to evaluate the kernel with inputs:
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
@available(*, deprecated, renamed: "MLXFast.MLXFastKernel")
public typealias MLXFastKernel = MLXFast.MLXFastKernel

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
) -> MLXFast.MLXFastKernel {
    return MLX.MLXFast.metalKernel(
        name: name, inputNames: inputNames, outputNames: outputNames,
        source: source, header: header,
        ensureRowContiguous: ensureRowContiguous, atomicOutputs: atomicOutputs
    )
}
