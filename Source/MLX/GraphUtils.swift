// Copyright © 2025 Apple Inc.

import Cmlx
import Foundation

/// Assigns human readable names to arrays in a compute graph.
///
/// Arrays that aren't given a name are numbered automatically.  Pass a namer to
/// ``exportToDot(_:namer:)`` or ``graphDescription(_:namer:)`` to make their output easier
/// to read:
///
/// ```swift
/// let x = MLXArray([1, 2, 3])
/// let y = MLXArray([4, 5, 6])
///
/// let namer = NodeNamer()
/// namer.setName("x", for: x)
/// namer.setName("y", for: y)
///
/// print(exportToDot([x + y], namer: namer))
/// ```
///
/// ### See Also
/// - ``exportToDot(_:namer:)``
/// - ``graphDescription(_:namer:)``
public final class NodeNamer {

    let ctx: mlx_node_namer

    /// Create an empty namer.
    public init() {
        ctx = mlx_node_namer_new()
    }

    deinit {
        mlx_node_namer_free(ctx)
    }

    /// Assign `name` to `array`.
    public func setName(_ name: String, for array: MLXArray) {
        mlx_node_namer_set_name(ctx, array.ctx, name)
    }

    /// The name of `array`.
    ///
    /// Arrays without an assigned name are given a generated one, which is recorded so that
    /// subsequent calls return the same value.
    public func name(for array: MLXArray) -> String? {
        var name: UnsafePointer<CChar>?
        mlx_node_namer_get_name(&name, ctx, array.ctx)
        guard let name else { return nil }
        return String(cString: name, encoding: .utf8)
    }
}

/// Capture everything written to a `FILE *` as a `String`.
private func captureFileOutput(_ body: (UnsafeMutablePointer<FILE>) -> Void) -> String {
    var buffer: UnsafeMutablePointer<CChar>?
    var size = 0

    guard let file = open_memstream(&buffer, &size) else { return "" }
    body(file)

    // closing flushes and finalizes buffer/size
    fclose(file)

    guard let buffer else { return "" }
    defer { free(buffer) }

    return String(cString: buffer, encoding: .utf8) ?? ""
}

/// A [GraphViz](https://graphviz.org) `DOT` representation of the compute graph that produces
/// `outputs`.
///
/// The arrays are not evaluated -- this describes the graph as it currently stands.
///
/// ```swift
/// let x = MLXArray([1, 2, 3])
/// let dot = exportToDot([x * 2 + 1])
/// try dot.write(to: url, atomically: true, encoding: .utf8)
/// ```
///
/// - Parameters:
///   - outputs: the outputs of the graph to describe
///   - namer: optional namer providing readable names for arrays
/// - Returns: the graph in `DOT` format
///
/// ### See Also
/// - ``graphDescription(_:namer:)``
/// - ``NodeNamer``
public func exportToDot(_ outputs: [MLXArray], namer: NodeNamer? = nil) -> String {
    let namer = namer ?? NodeNamer()
    let vector = new_mlx_vector_array(outputs)
    defer { mlx_vector_array_free(vector) }

    return captureFileOutput { file in
        mlx_export_to_dot(file, namer.ctx, vector)
    }
}

/// A textual description of the compute graph that produces `outputs`.
///
/// The arrays are not evaluated -- this describes the graph as it currently stands.
///
/// - Parameters:
///   - outputs: the outputs of the graph to describe
///   - namer: optional namer providing readable names for arrays
/// - Returns: the description of the graph
///
/// ### See Also
/// - ``exportToDot(_:namer:)``
/// - ``NodeNamer``
public func graphDescription(_ outputs: [MLXArray], namer: NodeNamer? = nil) -> String {
    let namer = namer ?? NodeNamer()
    let vector = new_mlx_vector_array(outputs)
    defer { mlx_vector_array_free(vector) }

    return captureFileOutput { file in
        mlx_print_graph(file, namer.ctx, vector)
    }
}
