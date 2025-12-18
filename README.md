# MLX Swift

[**Installation**](#installation) | [**Documentation**](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx) | [**Examples**](#examples)

MLX Swift is a Swift API for [MLX](https://ml-explore.github.io/mlx/build/html/index.html).

MLX is an array framework for machine learning on Apple silicon. MLX Swift
expands MLX to the Swift language, making research and experimentation easier
on Apple silicon.

## Language Models

LLM and VLM implementations are available in [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

## Examples

MLX Swift has [many
examples](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/examples),
including:

- [MNISTTrainer](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/MNISTTrainer/README.md): An example that runs on
  both iOS and macOS that downloads MNIST training data and trains a
  [LeNet](https://en.wikipedia.org/wiki/LeNet).

- [MLXChatExample](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/MLXChatExample/README.md): An example chat app that runs on both iOS and macOS that supports LLMs and VLMs.

- [LLMEval](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/LLMEval/README.md): A simple example that runs on both iOS
  and macOS that downloads an LLM and tokenizer from Hugging Face and
  generates text from a given prompt.

- [StableDiffusionExample](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/StableDiffusionExample/README.md): An
  example that runs on both iOS and macOS that downloads a stable diffusion model
  from Hugging Face and generates an image from a given prompt.

- [llm-tool](https://github.com/ml-explore/mlx-swift-examples/blob/main/Tools/llm-tool/README.md): A command line tool for generating text
  using a variety of LLMs available on the Hugging Face hub.

The [MLX Swift Examples repo](https://github.com/ml-explore/mlx-swift-examples)
contains the complete code and documentation for these examples, including
[guidelines on porting models](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxlmcommon/porting)
from MLX Python.

## Installation

The ``MLX`` Swift package can be built and run from Xcode or SwiftPM. A CMake installation is also provided, featuring a native Linux build option.

More details are in the [documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/install).

### Xcode

In Xcode you can add `https://github.com/ml-explore/mlx-swift.git` as a package
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
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimizers", package: "mlx-swift")]
```

> [!Note]
> SwiftPM (command line) cannot build the Metal shaders so the ultimate build has to be done
> via Xcode.

### xcodebuild

Although `SwiftPM` (command line) cannot build the Metal shaders, `xcodebuild` can and
it can be used to do command line builds:

```shell
# build and run tests
xcodebuild test -scheme mlx-swift-Package -destination 'platform=OS X'

# build Tutorial
xcodebuild build -scheme Tutorial -destination 'platform=OS X'
```

### CMake

#### (1) macOS

**Install Dependencies**

Building with CMake requires both CMake and Ninja to be installed. You can do
this with [Homebrew](https://brew.sh/):

```shell
brew install cmake
brew install ninja
```

**Build + Run Examples**

The examples use the Metal GPU backend by default on macOS. Note that the CUDA GPU backend is exclusive to Linux.

```shell
mkdir -p build
cd build
cmake .. -G Ninja
ninja
./example1
./tutorial
```

#### (2) Linux

**Install Dependencies**

RHEL/Fedora:
```shell
sudo dnf install -y blas-devel lapack-devel openblas-devel clang llvm cmake make ninja
# Then install Swift by following the instructions at https://swift.org
```

Ubuntu/Debian:
```shell
sudo apt-get update
sudo apt-get install -y libblas-dev liblapack-dev liblapacke-dev libopenblas-dev clang llvm cmake make ninja-build
# Then install Swift by following the instructions at https://swift.org
```

Refer to [swift.org](https://www.swift.org/install/linux/) for installation options and instructions specific to your Linux distribution.

**Build + Run Examples (CPU backend)**

On Linux, the examples use the CPU backend by default.

Note: SwiftPM builds are not currently supported for native Linux targets.

```shell
mkdir -p build
pushd build
cmake -DMLX_BUILD_METAL=OFF .. -G Ninja
ninja
./example1
./tutorial
popd
```

**Build + Run Examples (GPU CUDA backend)**

To find a complete reference implementation, consult the [CI setup](.github/scripts/setup+build-linux-cuda-cmake.sh) which provides a working example for Ubuntu 24.04 x86_64.

```shell
mkdir -p build
pushd build
cmake -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=ON -DMLX_C_BUILD_EXAMPLES=OFF .. -G Ninja
ninja
./example1 --device gpu
./tutorial --device gpu
popd
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

See [Releases](https://github.com/ml-explore/mlx-swift/releases).  Generally the MLX Swift version number corresponds to the same version number in [MLX](https://github.com/ml-explore/mlx).  Release notes indicate specifics.

All capabilities in MLX (Python) should be available in MLX Swift.  If you encounter any that are missing please file an issue or feel free to submit a PR.
