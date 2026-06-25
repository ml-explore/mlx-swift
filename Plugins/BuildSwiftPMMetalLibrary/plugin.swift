import Foundation
import PackagePlugin

@main
struct BuildSwiftPMMetalLibrary: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: any Target) async throws -> [Command] {
        #if os(Linux)
            return []
        #else
            let packageRoot = context.package.directoryURL
            let script = packageRoot.appendingPathComponent("tools/build-swiftpm-metallib.sh")
            let output = context.pluginWorkDirectoryURL.appendingPathComponent("default.metallib")

            return [
                .buildCommand(
                    displayName: "Build SwiftPM default.metallib",
                    executable: URL(fileURLWithPath: "/bin/bash"),
                    arguments: [script.path, output.path],
                    inputFiles: inputFiles(packageRoot: packageRoot, script: script),
                    outputFiles: [output]
                )
            ]
        #endif
    }

    #if !os(Linux)
        private func inputFiles(packageRoot: URL, script: URL) -> [URL] {
            let kernelsDirectory = packageRoot.appendingPathComponent(
                "Source/Cmlx/mlx/mlx/backend/metal/kernels")
            var files = [script]
            files.append(contentsOf: recursivelyCollectedMetalInputs(in: kernelsDirectory))
            return files
        }

        private func recursivelyCollectedMetalInputs(in directory: URL) -> [URL] {
            let fileManager = FileManager.default
            guard
                let enumerator = fileManager.enumerator(
                    at: directory, includingPropertiesForKeys: nil)
            else {
                return []
            }

            return enumerator.compactMap { entry -> URL? in
                guard let url = entry as? URL else { return nil }
                guard url.pathExtension == "metal" || url.pathExtension == "h" else {
                    return nil
                }
                return url
            }.sorted { $0.path < $1.path }
        }
    #endif
}
