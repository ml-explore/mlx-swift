import Foundation
import PackagePlugin

@main
struct CudaBuild: BuildToolPlugin {

    struct CodeGeneration: Codable {
        var tool: String
        var inputs: [String] = []
        var outputs: [String] = []
    }

    struct Settings: Codable {
        var headerSearchPaths: [String] = []
        var exclude: [String] = []
        var cppLanguageStandard: String? = nil
        var verbose: Bool = false
        var codeGeneration: [CodeGeneration] = []

        init() {}

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            headerSearchPaths =
                try c.decodeIfPresent([String].self, forKey: .headerSearchPaths) ?? []
            exclude = try c.decodeIfPresent([String].self, forKey: .exclude) ?? []
            cppLanguageStandard = try c.decodeIfPresent(String.self, forKey: .cppLanguageStandard)
            verbose = try c.decodeIfPresent(Bool.self, forKey: .verbose) ?? false
            codeGeneration =
                try c.decodeIfPresent([CodeGeneration].self, forKey: .codeGeneration) ?? []
        }
    }

    func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {

        print("CUDA Build Plugin")

        guard isCudaEnabled() else {
            print("CUDA is disabled")
            return []
        }

        guard let clangUrl = try? context.tool(named: "clang++") else {
            fatalError("clang++ not found")
        }

        print("Use clang++ at: \(clangUrl.url.path)")

        let sourceDir = target.directoryURL

        print("Source directory: \(sourceDir.path)")

        // Read settings from CudaBuild.json if it exists

        let settingsFile = sourceDir.appendingPathComponent("CudaBuild.json")
        let settings: Settings
        if let data = try? Data(contentsOf: settingsFile) {
            print("Found settings file: \(settingsFile.path)")
            settings = try JSONDecoder().decode(Settings.self, from: data)
        } else {
            settings = Settings()
        }

        // Scan source directory for .cu files

        let sourceDirPath = sourceDir.path.hasSuffix("/") ? sourceDir.path : sourceDir.path + "/"
        var sourceCuFiles: [URL] = []
        if let enumerator = FileManager.default.enumerator(
            at: sourceDir, includingPropertiesForKeys: nil)
        {
            while let inputUrl = enumerator.nextObject() as? URL {
                if inputUrl.pathExtension == "cu" {
                    guard inputUrl.path.hasPrefix(sourceDirPath) else {
                        fatalError(
                            "Input file \(inputUrl.path) is not under source directory \(sourceDirPath)"
                        )
                    }
                    let relateivePath = String(inputUrl.path.dropFirst(sourceDirPath.count))
                    if isExcluded(settings: settings, relativePath: relateivePath) {
                        print("Excluding \(relateivePath)")
                        continue
                    }
                    sourceCuFiles.append(URL(string: relateivePath, relativeTo: sourceDir)!)
                }
            }
        }

        print("Source files: \(sourceCuFiles.map { $0.relativePath })")

        let outputDir = context.pluginWorkDirectoryURL

        var commands: [Command] = []

        // Invoke code generation tools

        var generatedCuFiles: [URL] = []

        for (index, codeGen) in settings.codeGeneration.enumerated() {
            let tool = try context.tool(named: codeGen.tool)
            let codeGenOutputDir = outputDir.appendingPathComponent("gen\(index)-\(codeGen.tool)")
            try FileManager.default.createDirectory(
                at: codeGenOutputDir, withIntermediateDirectories: true)
            let inputUrls = codeGen.inputs.map { sourceDir.appendingPathComponent($0) }
            let outputUrls = codeGen.outputs.map { codeGenOutputDir.appendingPathComponent($0) }
            commands.append(
                .buildCommand(
                    displayName: "Running \(codeGen.tool)",
                    executable: tool.url,
                    arguments: inputUrls.map { $0.path } + outputUrls.flatMap { ["-o", $0.path] },
                    inputFiles: inputUrls,
                    outputFiles: outputUrls
                )
            )
            generatedCuFiles += outputUrls.filter { $0.pathExtension == "cu" }
        }

        print("Generated source files: \(generatedCuFiles.map { $0.relativePath })")

        // Invoke `encuda compile` to compile each .cu file to .cpp

        let encuda = try context.tool(named: "encuda")
        let verboseFlag = settings.verbose ? ["-v"] : []
        let headerSearchPathArgs = settings.headerSearchPaths.flatMap {
            ["-I", sourceDirPath + $0]
        }
        let stdArgs = settings.cppLanguageStandard.map { ["--std", $0] } ?? []

        for inputFile in sourceCuFiles + generatedCuFiles {
            let outputCpp = URL(string: inputFile.relativePath, relativeTo: outputDir)!
                .deletingPathExtension().appendingPathExtension("cpp")
            commands.append(
                .buildCommand(
                    displayName:
                        "Compiling \(inputFile.lastPathComponent) to \(outputCpp.lastPathComponent)",
                    executable: encuda.url,
                    arguments: ["compile"] + verboseFlag + stdArgs + [
                        "--clangpp", clangUrl.url.path,
                        "-I", sourceDir.path,
                    ] + headerSearchPathArgs + [
                        inputFile.path,
                        "-o", outputCpp.path,
                    ],
                    inputFiles: [inputFile],
                    outputFiles: [outputCpp]
                )
            )
        }

        // Invoke `encuda link` with all .cpp files

        let outputCpps = (sourceCuFiles + generatedCuFiles).map { inputFile in
            URL(string: inputFile.relativePath, relativeTo: outputDir)!
                .deletingPathExtension()
                .appendingPathExtension("cpp")
        }

        let linkOutput = outputDir.appendingPathComponent("__cuda_link.cpp")

        commands.append(
            .buildCommand(
                displayName: "Linking CUDA objects",
                executable: encuda.url,
                arguments: ["link"] + verboseFlag + stdArgs + [
                    "--clangpp", clangUrl.url.path,
                ] + outputCpps.map { $0.path } + ["-o", linkOutput.path],
                inputFiles: outputCpps,
                outputFiles: [linkOutput]
            )
        )

        return commands
    }

    func isExcluded(settings: Settings, relativePath: String) -> Bool {
        func isInFolder(path: String, folderPath: String) -> Bool {
            let prefix = folderPath.hasSuffix("/") ? folderPath : (folderPath + "/")
            return path.hasPrefix(prefix)
        }
        return settings.exclude.contains(relativePath)
            || settings.exclude.contains { isInFolder(path: relativePath, folderPath: $0) }
    }

    func isCudaEnabled() -> Bool {
        #if os(Linux)
            return ProcessInfo.processInfo.environment["SPM_CUDA"] != "0"
        #else
            return false
        #endif
    }
}
