// Copyright © 2024 Apple Inc.

import Foundation
import PackagePlugin

/// Prepare the metal shaders source for compilation.
///
/// The metal shaders (`mlx/backend/metal/kernels`) include headers as:
///
/// ```
/// #include "mlx/backend/metal/kernels/...h"
/// ```
///
/// but this doesn't work with the swiftpm build -- there is no way to set a header search path and this
/// absolute path won't work.
///
/// This plugin makes a copy of the shaders and modifies them to include the files using a relative path.
/// This code is specialized to the mlx/metal code but could be adapted elsewhere with changes.
@main
struct PrepareMetalShaders: BuildToolPlugin {

    /// pattern to rewrite
    private let include = try! Regex("#include \"mlx/backend/metal/kernels/([^\"]*)\"")

    func transformIncludes(url: URL) throws {
        let contents = try String(contentsOf: url, encoding: .utf8)

        let new: String

        // need to transform
        // #include "mlx/backend/metal/kernels/steel/gemm/transforms.h"
        //
        // into
        // #include "../../steel/gemm/transforms.h"

        let pathUnderKernels = url.pathComponents.drop { $0 != "output" }.dropLast()

        let rootPath =
            Array(repeating: "..", count: pathUnderKernels.count - 1).joined(separator: "/")
            + ((pathUnderKernels.count - 1 == 0) ? "" : "/")

        new =
            contents
            .replacing(include, with: { "#include \"\(rootPath)\($0[1].substring ?? "")\"" })

        try new.write(to: url, atomically: true, encoding: .utf8)
    }

    func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
        var commands = [Command]()

        let sourcePath = target.directory.appending(["mlx", "mlx", "backend", "metal", "kernels"])
        let source = URL(fileURLWithPath: sourcePath.string)

        let destinationPath = context.pluginWorkDirectory.appending(["output"])
        let destination = URL(fileURLWithPath: destinationPath.string)

        // only do the work if the directory doesn't exist
        if !FileManager.default.fileExists(atPath: destination.path(percentEncoded: false)) {
            // copy the files from the source area
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: source, to: destination)

            // the builder won't find metal kernels in subdirectories, so move them to the top
            if let enumerator = FileManager.default.enumerator(
                at: destination, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants])
            {
                for case let url as URL in enumerator {
                    let isRegularFile =
                        try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile ?? false
                    guard isRegularFile else {
                        continue
                    }

                    if url.deletingLastPathComponent().lastPathComponent == "output" {
                        // still in the top directory
                        continue
                    }

                    if url.pathExtension == "metal" {
                        try FileManager.default.moveItem(
                            at: url, to: destination.appending(component: url.lastPathComponent))
                    }
                }
            }

            // foreach file, transform the #includes
            if let enumerator = FileManager.default.enumerator(
                at: destination, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants])
            {
                for case let url as URL in enumerator {
                    let isRegularFile =
                        try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile ?? false
                    guard isRegularFile else {
                        continue
                    }

                    if url.lastPathComponent == "CMakeLists.txt" {
                        try FileManager.default.removeItem(at: url)
                        continue
                    }

                    try transformIncludes(url: url)
                }
            }
        }

        // a prebuild command to inject the output directory so swiftpm knows to pick it up
        commands.append(
            ._prebuildCommand(
                displayName: "Install Headers",
                executable: .init("/bin/echo"),
                arguments: [],
                outputFilesDirectory: Path(destination.path(percentEncoded: false))))

        return commands
    }
}
