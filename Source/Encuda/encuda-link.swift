import ArgumentParser
import Foundation

extension Encuda {
    struct Link: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "link"
        )

        @Option(name: .customLong("nvcc"), help: "Path to nvcc")
        var nvccPath: String? = nil

        @Option(name: .customLong("clangpp"), help: "Path to clang++")
        var clangppPath: String? = nil

        @Argument(help: "Input .cpp files to link")
        var inputFiles: [String]

        @Option(name: .customShort("o"), help: "Output file path")
        var output: String

        @Option(name: .customLong("std"), help: "C++ standard to use (e.g., c++17, c++20)")
        var std: String? = nil

        @Flag(name: .customShort("v"), help: "Enable verbose output")
        var verbose: Bool = false

        @Flag(name: .customLong("incremental"), help: "Skip link if output is up to date")
        var incremental: Bool = false

        mutating func run() throws {
            if verbose {
                print("CUDA Link")
                print("Input files: \(inputFiles)")
                print("Output file: \(output)")
            }

            if incremental && isUpToDate() {
                if verbose {
                    print("Output is up to date, skipping link")
                }
                return
            }

            let stdArgs = std.map { ["-std=\($0)"] } ?? []
            let archArgs =
                ProcessInfo.processInfo.environment["CUDA_ARCH"].map { ["-arch", $0] } ?? []

            for input in inputFiles {
                try clang(args: stdArgs + ["-c", input, "-o", input + ".cudalink.o"])
            }

            try nvcc(
                args: ["--device-link"] + stdArgs + archArgs + inputFiles.map { $0 + ".cudalink.o" }
                    + ["-o", output, "-Xcompiler", "-E"])

            for input in inputFiles {
                try? FileManager.default.removeItem(atPath: input + ".cudalink.o")
            }
        }

        var resolvedNvcc: String {
            if let path = nvccPath {
                return path
            } else if let found = searchForCommand("nvcc") {
                return found.path
            } else {
                fatalError("nvcc not found")
            }
        }

        var resolvedClangpp: String {
            if let path = clangppPath {
                return path
            } else if let found = searchForCommand("clang++") {
                return found.path
            } else {
                fatalError("clang++ not found")
            }
        }

        func nvcc(args: [String]) throws {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: resolvedNvcc)
            process.arguments = ["-ccbin=\(resolvedClangpp)"] + (verbose ? ["-v"] : []) + args
            try process.run()
            process.waitUntilExitWorkaround()
            guard process.terminationStatus == 0 else {
                throw EncudaError.nvccFailed(process.terminationStatus)
            }
        }

        func clang(args: [String]) throws {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: resolvedClangpp)
            process.arguments = (verbose ? ["-v"] : []) + args
            try process.run()
            process.waitUntilExitWorkaround()
            guard process.terminationStatus == 0 else {
                throw EncudaError.clangFailed(process.terminationStatus)
            }
        }

        private func isUpToDate() -> Bool {
            let fm = FileManager.default
            let outputURL = URL(fileURLWithPath: output)
            guard fm.fileExists(atPath: output),
                let outputMod =
                    (try? outputURL.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate
            else { return false }
            for input in inputFiles {
                guard fm.fileExists(atPath: input),
                    let inputMod =
                        (try? URL(fileURLWithPath: input).resourceValues(forKeys: [
                            .contentModificationDateKey
                        ]))?.contentModificationDate
                else { return false }
                if inputMod >= outputMod { return false }
            }
            return true
        }
    }
}
