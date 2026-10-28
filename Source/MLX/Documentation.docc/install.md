#  Installation

How to install and use.

``MLX`` is meant to be built and run from Xcode or SwiftPM.

## Xcode

In Xcode you can add `https://github.com/ml-explore/mlx-swift` as a package
dependency and link `MLX`, `MLXNN` and `MLXRandom` as needed.

## SwiftPM

To use ``MLX`` with SwiftPm you can add this to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0")
]
```

and add the libraries (as needed) as dependencies:

```swift
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXRandom", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimizers", package: "mlx-swift"),
               .product(name: "MLXFFT", package: "mlx-swift")]
```

> Note that SwiftPM cannot build the metal shaders so the ultimate build has to be done via
Xcode or `xcodebuild`.

## Package Traits

`JACCL` builds the JACCL distributed backend (RDMA over Thunderbolt 5) on
macOS. It is enabled by default. iOS, tvOS and visionOS never build it.

An app that does not run distributed MLX can leave the backend out of its
binary by disabling the default traits:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.33.0", traits: [])
```

SwiftPM enables the union of the traits that all dependents request. If another
package in the graph depends on `mlx-swift` with the default traits, `JACCL`
stays enabled.

## Command Line Tools

See <doc:troubleshooting> (Running From Command Line) for information about running command line tools from the shell.

