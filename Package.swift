// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-swift",
    
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],

    products: [
        .library(name: "Cmlx", targets: ["Cmlx"]),
        .library(name: "Mlx", targets: ["Mlx"]),
        .plugin(
          name: "PrepareMetalShaders",
          targets: ["PrepareMetalShaders"]
        ),
        .executable(name: "Example1", targets: ["Example1"]),
    ],
    dependencies: [
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
          name: "Mlx",
          dependencies: ["Cmlx"]
        ),
        .testTarget(
            name: "MlxTests",
            dependencies: ["Mlx"]
        ),
        
        .executableTarget(
            name: "Example1",
            dependencies: ["Mlx"],
            path: "Source/Examples",
            sources: ["Example1.swift"]
        ),
    ],
    cxxLanguageStandard: .gnucxx17
)
