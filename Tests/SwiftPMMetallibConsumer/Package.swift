// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "SwiftPMMetallibConsumer",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(name: "mlx-swift", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "SwiftPMMetallibConsumer",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ]
        )
    ]
)
