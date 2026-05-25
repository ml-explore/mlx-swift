import ArgumentParser
import Foundation

extension Encuda {
    struct Compile: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "compile"
        )

        @Option(name: .customLong("nvcc"), help: "Path to nvcc")
        var nvccPath: String? = nil

        @Option(name: .customLong("clangpp"), help: "Path to clang++")
        var clangppPath: String? = nil

        @Option(
            name: .customShort("I"), parsing: .unconditionalSingleValue, help: "Include directories"
        )
        var includeDirs: [String] = []

        @Option(name: .customShort("o"), help: "Output file path")
        var output: String

        @Argument(help: "Input .cu files to compile")
        var inputFiles: [String]

        @Option(name: .customLong("std"), help: "C++ standard to use (e.g., c++17, c++20)")
        var std: String? = nil

        @Flag(name: .customShort("v"), help: "Enable verbose output")
        var verbose: Bool = false

        mutating func run() throws {

            if verbose {
                print("Running encuda compile")
            }

            let resolvedNvcc: String
            if let path = nvccPath {
                resolvedNvcc = path
            } else if let found = searchForCommand("nvcc") {
                resolvedNvcc = found.path
            } else {
                fatalError("nvcc not found")
            }

            let includeArgs = includeDirs.flatMap { ["-I", $0] }
            let ccbinArgs = clangppPath.map { ["-ccbin=\($0)"] } ?? []
            let stdArgs = std.map { ["-std=\($0)"] } ?? []
            let archArgs =
                ProcessInfo.processInfo.environment["CUDA_ARCH"].map { ["-arch", $0] } ?? []

            let process = Process()
            process.executableURL = URL(fileURLWithPath: resolvedNvcc)
            process.arguments =
                ["-cuda", "-rdc=true", "--expt-relaxed-constexpr"] + stdArgs + ccbinArgs + archArgs
                + (verbose ? ["-v"] : []) + includeArgs + inputFiles + ["-o", output]
            try process.run()
            process.waitUntilExitWorkaround()
            guard process.terminationStatus == 0 else {
                throw EncudaError.nvccFailed(process.terminationStatus)
            }
        }
    }
}
