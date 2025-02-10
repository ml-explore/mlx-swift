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
    ],
    dependencies: [
        // for Complex type
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "Cmlx",
            exclude: [
                // vendor docs
                "metal-cpp.patch",
                "vendor-README.md",

                // example code + mlx-c distributed
                "mlx-c/examples",
                "mlx-c/mlx/c/distributed.cpp",
                "mlx-c/mlx/c/distributed_group.cpp",

                // vendored library, include header only
                "json",

                // vendored library
                "fmt/test",
                "fmt/doc",
                "fmt/support",
                "fmt/src/os.cc",
                "fmt/src/fmt.cc",

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
                "mlx/mlx/backend/no_cpu",

                "mlx/mlx/backend/common/default_primitives.cpp",

                // this uses neon code and will not build on x86 (e.g. via Release).
                // see mlx-conditional/accelerate-softmax.cpp
                "mlx/mlx/backend/accelerate/softmax.cpp",

                // build variants (we are opting _out_ of these)
                "mlx/mlx/io/no_safetensors.cpp",
                "mlx/mlx/io/gguf.cpp",
                "mlx/mlx/io/gguf_quants.cpp",

                // see PrepareMetalShaders -- don't build the kernels in place
                "mlx/mlx/backend/metal/kernels",
                "mlx/mlx/backend/metal/nojit_kernels.cpp",

                // do not build distributed support (yet)
                "mlx/mlx/distributed/mpi",
                "mlx/mlx/distributed/ops.cpp",
                "mlx/mlx/distributed/primitives.cpp",

                // the mlx-c side of distributed
                "include/mlx/c/distributed.cpp",
                "include/mlx/c/distributed_group.cpp",
            ],

            cSettings: [
                .headerSearchPath("mlx"),
                .headerSearchPath("mlx-c"),
            ],

            cxxSettings: [
                .headerSearchPath("mlx"),
                .headerSearchPath("mlx-c"),
                .headerSearchPath("metal-cpp"),
                .headerSearchPath("json/single_include/nlohmann"),
                .headerSearchPath("fmt/include"),

                .define("ACCELERATE_NEW_LAPACK"),
                .define("_METAL_"),
                .define("SWIFTPM_BUNDLE", to: "\"mlx-swift_Cmlx\""),
                .define("METAL_PATH", to: "\"default.metallib\""),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
            ]
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
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXRandom",
            dependencies: ["MLX"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXFast",
            dependencies: ["MLX", "Cmlx"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXNN",
            dependencies: ["MLX", "MLXRandom", "MLXFast"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXOptimizers",
            dependencies: ["MLX", "MLXNN"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXFFT",
            dependencies: ["MLX"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXLinalg",
            dependencies: ["MLX"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
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
