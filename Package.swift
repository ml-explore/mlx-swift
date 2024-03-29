// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Copyright © 2024 Apple Inc.

import PackageDescription

let package = Package(
    name: "mlx-swift",

    platforms: [
        .macOS("13.3"),
        .iOS(.v16),
        .visionOS(.v1),
    ],

    products: [
        // main targets
        .library(name: "MLX", targets: ["MLX"]),
        .library(name: "MLXRandom", targets: ["MLXRandom"]),
        .library(name: "MLXNN", targets: ["MLXNN"]),
        .library(name: "MLXOptimizers", targets: ["MLXOptimizers"]),
        .library(name: "MLXFFT", targets: ["MLXFFT"]),
        .library(name: "MLXLinalg", targets: ["MLXLinalg"]),
        .library(name: "MLXFast", targets: ["MLXFast"]),

        // build support & back end
        .plugin(
            name: "PrepareMetalShaders",
            targets: ["PrepareMetalShaders"]
        ),
    ],
    dependencies: [
        // for Complex type
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")
    ],
    targets: [
        // plugin to help build the metal shaders
        .plugin(
            name: "PrepareMetalShaders",
            capability: .buildTool(),
            path: "Plugins/PrepareMetalShaders"
        ),
        .target(
            name: "Cmlx",
            exclude: [
                // exclude here -- it is part of the include directory (public api)
                "mlx-c",

                // vendored library, include header only
                "json",

                // vendored library, do not include driver
                "gguf-tools/gguf-tools.c",

                // these are selected conditionally
                // via mlx-conditional/compiled_conditional.cpp
                "mlx/mlx/backend/common/compiled_nocpu.cpp",
                "mlx/mlx/backend/common/compiled_cpu.cpp",

                // mlx files that are not part of the build
                "mlx/ACKNOWLEDGMENTS.md",
                "mlx/CMakeLists.txt",
                "mlx/CODE_OF_CONDUCT.md",
                "mlx/CONTRIBUTING.md",
                "mlx/LICENSE",
                "mlx/MANIFEST.in",
                "mlx/README.md",
                "mlx/benchmarks",
                "mlx/cmake",
                "mlx/docs",
                "mlx/examples",
                "mlx/mlx.pc.in",
                "mlx/pyproject.toml",
                "mlx/python",
                "mlx/setup.py",
                "mlx/tests",

                // opt-out of these backends (using metal)
                "mlx/mlx/backend/no_metal",
                "mlx/mlx/backend/accelerate",

                // see PrepareMetalShaders -- don't build the kernels in place
                "mlx/mlx/backend/metal/kernels",
            ],

            cSettings: [
                .headerSearchPath("mlx"),
                .headerSearchPath("include/mlx-c"),
            ],

            cxxSettings: [
                .headerSearchPath("mlx"),
                .headerSearchPath("include/mlx-c"),
                .headerSearchPath("metal-cpp"),
                .headerSearchPath("json/single_include/nlohmann"),
                .headerSearchPath("gguf-tools"),

                .define("ACCELERATE_NEW_LAPACK"),
                .define("_METAL_"),
                .define("SWIFTPM_BUNDLE", to: "\"mlx-swift_Cmlx\""),
                .define("METAL_PATH", to: "\"default.metallib\""),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
            ],

            // run the plugin to build the metal shaders
            plugins: [.plugin(name: "PrepareMetalShaders")]
        ),
        .testTarget(
            name: "CmlxTests",
            dependencies: ["Cmlx"]
        ),

        .target(
            name: "MLX",
            dependencies: [
                "Cmlx",
                .product(name: "Numerics", package: "swift-numerics"),
            ]
        ),
        .target(
            name: "MLXRandom",
            dependencies: ["MLX"]
        ),
        .target(
            name: "MLXFast",
            dependencies: ["MLX", "Cmlx"]
        ),
        .target(
            name: "MLXNN",
            dependencies: ["MLX", "MLXRandom", "MLXFast"]
        ),
        .target(
            name: "MLXOptimizers",
            dependencies: ["MLX", "MLXNN"]
        ),
        .target(
            name: "MLXFFT",
            dependencies: ["MLX"]
        ),
        .target(
            name: "MLXLinalg",
            dependencies: ["MLX"]
        ),

        .testTarget(
            name: "MLXTests",
            dependencies: [
                "MLX", "MLXRandom", "MLXNN", "MLXOptimizers", "MLXFFT", "MLXLinalg", "MLXFast",
            ]
        ),

        // ------
        // Example programs

        .executableTarget(
            name: "Example1",
            dependencies: ["MLX"],
            path: "Source/Examples",
            sources: ["Example1.swift"]
        ),
        .executableTarget(
            name: "Tutorial",
            dependencies: ["MLX"],
            path: "Source/Examples",
            sources: ["Tutorial.swift"]
        ),

        // ------
        // Internal Tools

        .executableTarget(
            name: "GenerateGrad",
            path: "Source/Tools",
            sources: ["GenerateGrad.swift"]
        ),

    ],
    cxxLanguageStandard: .gnucxx17
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    // docc builder
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
