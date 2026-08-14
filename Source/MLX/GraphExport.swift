import Cmlx

/// Export the computation graph of an array to a DOT file for visualization.
public func exportGraphToDot(path: String, output: MLXArray) {
    path.withCString { cPath in
        mlx_export_to_dot_file(cPath, output.ctx)
    }
}
