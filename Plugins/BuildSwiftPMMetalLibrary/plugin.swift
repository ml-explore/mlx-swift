import Foundation
import PackagePlugin

@main
struct BuildSwiftPMMetalLibrary: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: any Target) async throws -> [Command] {
        #if os(Linux)
            return []
        #else
            let packageRoot = context.package.directory
            let script = packageRoot.appending("tools", "build-swiftpm-metallib.sh")
            let output = context.pluginWorkDirectory.appending("default.metallib")

            return [
                .buildCommand(
                    displayName: "Build SwiftPM default.metallib",
                    executable: Path("/bin/bash"),
                    arguments: [script, output],
                    inputFiles: inputFiles(packageRoot: packageRoot, script: script),
                    outputFiles: [output]
                )
            ]
        #endif
    }

    #if !os(Linux)
        private func inputFiles(packageRoot: Path, script: Path) -> [Path] {
            let kernelsDirectory = packageRoot.appending(
                "Source",
                "Cmlx",
                "mlx",
                "mlx",
                "backend",
                "metal",
                "kernels"
            )
            var files = [script]
            files.append(contentsOf: recursivelyCollectedMetalInputs(in: kernelsDirectory))
            return files
        }

        private func recursivelyCollectedMetalInputs(in directory: Path) -> [Path] {
            let fileManager = FileManager.default
            guard let enumerator = fileManager.enumerator(atPath: directory.string) else {
                return []
            }

            return enumerator.compactMap { entry -> Path? in
                guard let entry = entry as? String else { return nil }
                guard entry.hasSuffix(".metal") || entry.hasSuffix(".h") else { return nil }
                return directory.appending(subpath: entry)
            }.sorted { $0.string < $1.string }
        }
    #endif
}
