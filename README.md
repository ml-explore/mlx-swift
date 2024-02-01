# MLX Swift

[**Installation**](#installation) | [**MLX**](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/) | [**MLXRandom**](https://ml-explore.github.io/mlx-swift/MLXRandom/documentation/mlxrandom/) | [**MLXNN**](https://ml-explore.github.io/mlx-swift/MLXNN/documentation/mlxnn/) | [**Examples**](#examples) 

MLX Swift is a Swift API for [MLX](https://ml-explore.github.io/mlx/build/html/index.html).

MLX is an array framework for machine learning research on Apple silicon. MLX
Swift expands MLX to the Swift language, making experimentation on Apple
silicon easier for ML researchers.

## Examples

Coming soon.

## Installation

``MLX`` is meant to be built and run from XCode or SwiftPM.  A CMake install is also provided. 

More details in the [documentation.](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/install)

### XCode

In XCode you can add `https://github.com/ml-explore/mlx-swift` as a package
dependency and link `MLX`, `MLXNN` and `MLXRandom` as needed.

### SwiftPM

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
XCode.


### CMake

With CMake:
```
mkdir build
cd build
cmake .. -G Ninja
ninja
./example
```

## Limitations

`mlx-swift` currently has a subset of the `mlx` functionality.
Everything for doing inference is available and the rest are coming soon.

Here is what is missing:

- [fft](https://ml-explore.github.io/mlx/build/html/python/fft.html) package
- [transforms](https://ml-explore.github.io/mlx/build/html/python/fft.html) package -- `eval()` and `grad()` **are** available
- [linalg](https://ml-explore.github.io/mlx/build/html/python/linalg.html) package
- [some NN](https://ml-explore.github.io/mlx/build/html/python/nn.html) layers
- [optimizers](https://ml-explore.github.io/mlx/build/html/python/optimizers.html) package

Please check out the documentation!

[**MLX**](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/) | [**MLXRandom**](https://ml-explore.github.io/mlx-swift/MLXRandom/documentation/mlxrandom/) | [**MLXNN**](https://ml-explore.github.io/mlx-swift/MLXNN/documentation/mlxnn/)
