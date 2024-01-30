#  Installation

How to install and use.

``MLX`` is meant to be built and run from Xcode or SwiftPM.

## Xcode

In Xcode you can add `https://github.com/ml-explore/mlx-swift` as a package
dependency and link `MLX`, `MLXNN` and `MLXRandom` as needed.

## SwiftPM

To use ``MLX`` with swiftpm you can add this to your `Package.swift`:

```
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0")
]
```

and add the libraries as dependencies:

```
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXRandom", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift")]
```

> Note that SwiftPM cannot build the metal shaders so the ultimate build has to be done via
Xcode or `xcodebuild`.

## Command Line Tools

> MLX requires metal shaders from the `Cmlx` framework -- these not not usable
from command line tools unless `DYLD_FRAMEWORK_PATH` makes them visible.

``MLX`` is built on top of the `mlx` C++ library packaged in the `Cmlx` swift package.  `Cmlx`
produces a bundle called `mlx-swift_Cmlx.bundle` which contains the compiled metal shaders.
If you build a command line tool and run it from Xcode, the `DYLD_FRAMEWORK_PATH` is set 
so that this bundle is found.

If you want to run the same command from the shell you must manually set the `DYLD_FRAMEWORK_PATH`
to the build directory.

## Applications

Applications should be configured to copy `mlx-swift_Cmlx.bundle` -- this will happen automatically
if you link the `MLX` library.

