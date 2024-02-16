# MLX Swift

[**Installation**](#installation) | [**Documentation**](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/) | [**Examples**](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/examples)

MLX Swift is a Swift API for [MLX](https://ml-explore.github.io/mlx/build/html/index.html).

MLX is an array framework for machine learning research on Apple
silicon. MLX Swift expands MLX to the Swift language, making research and
experimentation easier on Apple silicon. MLX is intended for research and
not for production deployment of models in apps.

## Examples

MLX Swift has a [few examples]((https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/examples), including:

- Large-scale text generation with Mistral 7B
- Training a simple multt-layer perceoptron on MNSIT

## Installation

The ``MLX`` Swift package can be built and run from Xcode or SwiftPM. A CMake install is also provided. 

More details are in the [documentation](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/install).

### Xcode

In Xcode you can add `https://github.com/ml-explore/mlx-swift` as a package
dependency and link `MLX`, `MLXNN`, `MLXOptimizers` and `MLXRandom` as needed.

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
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimziers", package: "mlx-swift"),
               .product(name: "MLXFFT", package: "mlx-swift")]
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

## Versions

The software generally matches the API and implementation of MLX as of hash
[f6e911ced01cb1f1d7f0843620412f002d525e37](https://github.com/ml-explore/mlx/tree/f6e911ced01cb1f1d7f0843620412f002d525e37).

There may be some implementations that match newer code/api, especially where
there is a python equivalent but that hash was the reference implementation.
