# MLX Swift

[**Installation**](#installation) | [**Documentation**](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx) | [**Examples**](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/examples)

MLX Swift is a Swift API for [MLX](https://ml-explore.github.io/mlx/build/html/index.html).

MLX is an array framework for machine learning research on Apple
silicon. MLX Swift expands MLX to the Swift language, making research and
experimentation easier on Apple silicon. MLX is intended for research and
not for production deployment of models in apps.

## Examples

MLX Swift has a [few
examples](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/examples),
including:

- Large-scale text generation with Mistral 7B
- Training a simple LeNet on MNIST
- Examples that run on macOS or iOS

The [MLX Swift Examples repo](https://github.com/ml-explore/mlx-swift-examples)
contains the complete code and documentation for these examples.

## Installation

The ``MLX`` Swift package can be built and run from Xcode or SwiftPM. A CMake install is also provided. 

More details are in the [documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/install).

### Xcode

In Xcode you can add `https://github.com/ml-explore/mlx-swift` as a package
dependency and link `MLX`, `MLXNN`, `MLXOptimizers` and `MLXRandom` as needed.

### SwiftPM

To use ``MLX`` with SwiftPM you can add this to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0")
]
```

and add the libraries as dependencies:

```swift
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXRandom", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimizers", package: "mlx-swift"),
               .product(name: "MLXFFT", package: "mlx-swift")]
```

> [!Note] 
> SwiftPM (command line) cannot build the Metal shaders so the ultimate build has to be done
> via Xcode.

### xcodebuild

Although `SwiftPM` (command line) cannot build the Metal shaders, `xcodebuild` can and
it can be used to do command line builds:

```
# build and run tests
xcodebuild test -scheme mlx-swift-Package -destination 'platform=OS X'

# build Tutorial
xcodebuild build -scheme Tutorial -destination 'platform=OS X'
```

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

## Contributing 

Check out the [contribution guidelines](CONTRIBUTING.md) for more information
on contributing to MLX. See the
[docs](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/install) for more
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

## Versions

The ``MLX`` array functions should match MLX as of tag 
[v0.3.0](https://github.com/ml-explore/mlx/releases/tag/v0.3.0).  The `MLXNN`
package should match MLX (`mlx.nn`) as of tag
[v0.0.10](https://github.com/ml-explore/mlx/releases/tag/v0.0.10).
