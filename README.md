# MLX Swift

[**Installation**](#installation) | [**MLX**](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/) | [**MLXRandom**](https://ml-explore.github.io/mlx-swift/MLXRandom/documentation/mlxrandom/) | [**MLXNN**](https://ml-explore.github.io/mlx-swift/MLXNN/documentation/mlxnn/) | [**Examples**](#examples) 

MLX Swift is a Swift API for [MLX](https://ml-explore.github.io/mlx/build/html/index.html).

MLX is an array framework for machine learning research on Apple silicon. MLX
Swift expands MLX to the Swift language, making experimentation on Apple
silicon easier for ML researchers.

## Examples

Coming soon.

## Installation

The ``MLX`` Swift package can be built and run from Xcode or SwiftPM. A CMake install is also provided. 

More details are in the [documentation](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/install).

### Xcode

In Xcode you can add `https://github.com/ml-explore/mlx-swift` as a package
dependency and link `MLX`, `MLXNN` and `MLXRandom` as needed.

### SwiftPM

To use ``MLX`` with SwiftPM you can add this to your `Package.swift`:

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

> [!Note] 
> SwiftPM cannot build the Metal shaders so the ultimate build has to be done
> via Xcode.


### CMake

Building with CMake requires both CMake and Ninja to be installed. You can do
this with [Homebrew](https://brew.sh/):

```shell
brew install cmake
brew install ninja
```

With CMake:

```shell
mkdir build
cd build
cmake .. -G Ninja
ninja
./example
```

## Limitations

MLX Swift currently supports a subset of MLX. Most of what you need for
training and inference with typical neural network models is available. The
rest is coming soon.

Here is what is missing:

- [fft](https://ml-explore.github.io/mlx/build/html/python/fft.html) package
- [linalg](https://ml-explore.github.io/mlx/build/html/python/linalg.html) package

For more details on the API see the [documentation](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/).

## Contributing 

Check out the [contribution guidelines](CONTRIBUTING.md) for more information
on contributing to MLX. See the
[docs](https://ml-explore.github.io/mlx/build/html/install.html) for more
information on building from source, and running tests.

We are grateful for all of [our
contributors](ACKNOWLEDGMENTS.md#Individual-Contributors). If you contribute
to MLX Swift and wish to be acknowledged, please add your name to the list in your
pull request.

MLX Swift was initially developed by David Koski and Ronan Collobert, and is
now maintained by David Koski. MLX Swift is built on top of
[MLX](https://github.com/ml-explore/mlx), which was initially developed with
equal contribution by Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and
Ronan Collobert.
