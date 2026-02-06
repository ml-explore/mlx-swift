// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Copyright © 2024 Apple Inc.

import PackageDescription

#if os(Linux)
    let platformExcludes: [String] = [
        // Linux specific excludes
        "framework",
        "include-framework",
        "metal-cpp",
        // Exclude Metal backend files on Linux, but keep no_metal.cpp for stubs
        "mlx/mlx/backend/metal/allocator.cpp",
        "mlx/mlx/backend/metal/binary.cpp",
        "mlx/mlx/backend/metal/compiled.cpp",
        "mlx/mlx/backend/metal/conv.cpp",
        "mlx/mlx/backend/metal/copy.cpp",
        "mlx/mlx/backend/metal/custom_kernel.cpp",
        "mlx/mlx/backend/metal/device.cpp",
        "mlx/mlx/backend/metal/device_info.cpp",
        "mlx/mlx/backend/metal/distributed.cpp",
        "mlx/mlx/backend/metal/eval.cpp",
        "mlx/mlx/backend/metal/event.cpp",
        "mlx/mlx/backend/metal/fence.cpp",
        "mlx/mlx/backend/metal/fft.cpp",
        "mlx/mlx/backend/metal/hadamard.cpp",
        "mlx/mlx/backend/metal/indexing.cpp",
        "mlx/mlx/backend/metal/jit_kernels.cpp",
        "mlx/mlx/backend/metal/logsumexp.cpp",
        "mlx/mlx/backend/metal/matmul.cpp",
        "mlx/mlx/backend/metal/metal.cpp",
        "mlx/mlx/backend/metal/normalization.cpp",
        "mlx/mlx/backend/metal/primitives.cpp",
        "mlx/mlx/backend/metal/quantized.cpp",
        "mlx/mlx/backend/metal/reduce.cpp",
        "mlx/mlx/backend/metal/resident.cpp",
        "mlx/mlx/backend/metal/rope.cpp",
        "mlx/mlx/backend/metal/scaled_dot_product_attention.cpp",
        "mlx/mlx/backend/metal/scan.cpp",
        "mlx/mlx/backend/metal/slicing.cpp",
        "mlx/mlx/backend/metal/softmax.cpp",
        "mlx/mlx/backend/metal/sort.cpp",
        "mlx/mlx/backend/metal/ternary.cpp",
        "mlx/mlx/backend/metal/unary.cpp",
        "mlx/mlx/backend/metal/utils.cpp",
        "mlx/mlx/backend/metal/kernels",  // Exclude kernels directory
        "mlx/mlx/backend/metal/jit",  // Exclude jit directory

        "mlx/mlx/backend/gpu",  // Exclude GPU backend on Linux, use no_gpu instead
        "mlx/mlx/backend/no_cpu",  // Exclude no_cpu backend on Linux, use cpu instead
        "mlx/mlx/backend/cpu/gemms/bnns.cpp",  // macOS Accelerate version
        "mlx-conditional",
        "mlx-c/mlx/c/metal.cpp",

        "mlx-c/mlx/c/fast.cpp",  // Exclude on Linux - calls metal_kernel unconditionally
    ]

    let cxxSettings: [CXXSetting] = []

    let linkerSettings: [LinkerSetting] = [
        .linkedLibrary("gfortran", .when(platforms: [.linux])),
        .linkedLibrary("blas", .when(platforms: [.linux])),
        .linkedLibrary("lapack", .when(platforms: [.linux])),
        .linkedLibrary("openblas", .when(platforms: [.linux])),
    ]

    let mlxSwiftExcludes: [String] = [
        "GPU+Metal.swift",
        "MLXArray+Metal.swift",
        "MLXFast.swift",
        "MLXFastKernel.swift",
    ]
#else
    let platformExcludes: [String] = [
        "mlx/mlx/backend/cpu/compiled.cpp",

        // opt-out of these backends (using metal)
        "mlx/mlx/backend/no_gpu",
        "mlx/mlx/backend/no_cpu",
        "mlx/mlx/backend/metal/no_metal.cpp",

        // bnns instead of simd (accelerate)
        "mlx/mlx/backend/cpu/gemms/simd_fp16.cpp",
        "mlx/mlx/backend/cpu/gemms/simd_bf16.cpp",
    ]

    let cxxSettings: [CXXSetting] = [
        .headerSearchPath("metal-cpp"),

        .define("MLX_USE_ACCELERATE"),
        .define("ACCELERATE_NEW_LAPACK"),
        .define("_METAL_"),
        .define("SWIFTPM_BUNDLE", to: "\"mlx-swift_Cmlx\""),
        .define("METAL_PATH", to: "\"default.metallib\""),
    ]

    let linkerSettings: [LinkerSetting] = [
        .linkedFramework("Foundation"),
        .linkedFramework("Metal"),
        .linkedFramework("Accelerate"),
    ]

    let mlxSwiftExcludes: [String] = []
#endif

let cmlx = Target.target(
    name: "Cmlx",
    path: "Source/Cmlx",
    exclude: platformExcludes + [
        // vendor docs
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
        "mlx/mlx/backend/no_cpu/compiled.cpp",

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

        // special handling for cuda -- we need to keep one file:
        // mlx/mlx/backend/cuda/no_cuda.cpp

        "mlx/mlx/backend/cuda/allocator.cpp",
        "mlx/mlx/backend/cuda/compiled.cpp",
        "mlx/mlx/backend/cuda/conv.cpp",
        "mlx/mlx/backend/cuda/cublas_utils.cpp",
        "mlx/mlx/backend/cuda/cuda.cpp",
        "mlx/mlx/backend/cuda/cudnn_utils.cpp",
        "mlx/mlx/backend/cuda/custom_kernel.cpp",
        "mlx/mlx/backend/cuda/delayload.cpp",
        "mlx/mlx/backend/cuda/device.cpp",
        "mlx/mlx/backend/cuda/device_info.cpp",
        "mlx/mlx/backend/cuda/eval.cpp",
        "mlx/mlx/backend/cuda/fence.cpp",
        "mlx/mlx/backend/cuda/indexing.cpp",
        "mlx/mlx/backend/cuda/jit_module.cpp",
        "mlx/mlx/backend/cuda/load.cpp",
        "mlx/mlx/backend/cuda/matmul.cpp",
        "mlx/mlx/backend/cuda/primitives.cpp",
        "mlx/mlx/backend/cuda/scaled_dot_product_attention.cpp",
        "mlx/mlx/backend/cuda/slicing.cpp",
        "mlx/mlx/backend/cuda/utils.cpp",
        "mlx/mlx/backend/cuda/worker.cpp",

        "mlx/mlx/backend/cuda/binary",
        "mlx/mlx/backend/cuda/conv",
        "mlx/mlx/backend/cuda/copy",
        "mlx/mlx/backend/cuda/device",
        "mlx/mlx/backend/cuda/gemms",
        "mlx/mlx/backend/cuda/quantized",
        "mlx/mlx/backend/cuda/reduce",
        "mlx/mlx/backend/cuda/steel",
        "mlx/mlx/backend/cuda/unary",

        // build variants (we are opting _out_ of these)
        "mlx/mlx/io/no_safetensors.cpp",
        "mlx/mlx/io/gguf.cpp",
        "mlx/mlx/io/gguf_quants.cpp",

        // see PrepareMetalShaders -- don't build the kernels in place
        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",

        // do not build distributed support (yet)
        "mlx/mlx/distributed/mpi/mpi.cpp",
        "mlx/mlx/distributed/ring/ring.cpp",
        "mlx/mlx/distributed/nccl/nccl.cpp",
        "mlx/mlx/distributed/nccl/nccl_stub",
        "mlx/mlx/distributed/jaccl/jaccl.cpp",
        "mlx/mlx/distributed/jaccl/mesh.cpp",
        "mlx/mlx/distributed/jaccl/ring.cpp",
        "mlx/mlx/distributed/jaccl/utils.cpp",
    ],
    cSettings: [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
    ],
    cxxSettings: cxxSettings + [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
        .headerSearchPath("json/single_include/nlohmann"),
        .headerSearchPath("fmt/include"),
        .define("MLX_VERSION", to: "\"0.24.2\""),
        .define("MLX_ENABLE_NAX", to: "1"),
    ],
    linkerSettings: linkerSettings
)

let package = Package(
    name: "mlx-swift",

    platforms: [
        .macOS("14.0"),
        .iOS(.v17),
        .tvOS(.v17),
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
        cmlx,
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
            exclude: mlxSwiftExcludes,
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
            dependencies: ["MLX"],
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
                "MLX", "MLXNN", "MLXOptimizers",
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
        .executableTarget(
            name: "CustomFunctionExample",
            dependencies: ["MLX"],
            path: "Source/Examples",
            sources: ["CustomFunctionExample.swift"]
        ),
        .executableTarget(
            name: "CustomFunctionExampleSimple",
            dependencies: ["MLX"],
            path: "Source/Examples",
            sources: ["CustomFunctionExampleSimple.swift"]
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
