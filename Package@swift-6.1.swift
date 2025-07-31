// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Copyright © 2024 Apple Inc.

import Foundation
import PackageDescription

// Function to get exclude list based on whether CUDA trait is enabled
func getExcludeList(forCUDA: Bool) -> [String] {
    var excludes = [
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
        "mlx/mlx/backend/no_gpu",

        // build variants (we are opting _out_ of these)
        "mlx/mlx/io/no_safetensors.cpp",
        "mlx/mlx/io/gguf.cpp",
        "mlx/mlx/io/gguf_quants.cpp",

        // see PrepareMetalShaders -- don't build the kernels in place
        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",
        "mlx/mlx/backend/metal/no_metal.cpp",

        // do not build distributed support (yet)
        "mlx/mlx/distributed/mpi/mpi.cpp",
        "mlx/mlx/distributed/ring/ring.cpp",

        // bnns instead of simd (accelerate)
        "mlx/mlx/backend/cpu/gemms/simd_fp16.cpp",
        "mlx/mlx/backend/cpu/gemms/simd_bf16.cpp",

        // Always exclude the individual backend compiled files
        // We use backend_compiled.cpp to conditionally include them
        "mlx/mlx/backend/cpu/compiled.cpp",
        "mlx/mlx/backend/cuda/compiled.cpp",
        "mlx/mlx/backend/no_cpu/compiled.cpp",
        "mlx-conditional/compiled_conditional.cpp",
    ]

    if forCUDA {
        // When building with CUDA, exclude CPU backend
        // (CUDA backend will be used)
    } else {
        // When building without CUDA, exclude CUDA backend directory
        excludes.append("mlx/mlx/backend/cuda")
        excludes.append("mlx/mlx/backend/no_cpu")
    }

    return excludes
}

let package = Package(
    name: "mlx-swift",

    platforms: [
        .macOS("13.3"),
        .iOS(.v16),
        .tvOS(.v16),
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

    traits: [
        .trait(name: "CUDA")
    ],

    dependencies: [
        // for Complex type
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0")
    ],

    targets: [
        .target(
            name: "Cmlx",
            exclude: getExcludeList(forCUDA: false),  // Default to CPU backend

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

                .define("MLX_USE_ACCELERATE"),
                .define("ACCELERATE_NEW_LAPACK"),
                .define("_METAL_"),
                .define("SWIFTPM_BUNDLE", to: "\"mlx-swift_Cmlx\""),
                .define("METAL_PATH", to: "\"default.metallib\""),
                .define("MLX_VERSION", to: "\"0.27.1\""),
                .define("MLX_BUILD_CUDA", .when(traits: ["CUDA"])),
            ],

            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedLibrary("cudart", .when(traits: ["CUDA"])),
                .linkedLibrary("cublas", .when(traits: ["CUDA"])),
                .linkedLibrary("cufft", .when(traits: ["CUDA"])),
                .linkedLibrary("cudnn", .when(traits: ["CUDA"])),
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
                .enableExperimentalFeature("StrictConcurrency"),
                .define("CUDA_AVAILABLE", .when(traits: ["CUDA"])),
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
