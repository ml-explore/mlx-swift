# mlx-swift

[**Installation**](#installation) | [**MLX**](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/) | [**MLXRandom**](https://ml-explore.github.io/mlx-swift/MLXRandom/documentation/mlxrandom/) | [**MLXNN**](https://ml-explore.github.io/mlx-swift/MLXNN/documentation/mlxnn/) | [**Examples**](#examples) 

`mlx-swift` is a Swift API for the [mlx package](https://ml-explore.github.io/mlx/build/html/index.html).

MLX is an array framework for machine learning on Apple silicon, brought to you
by Apple machine learning research.

Some key features of MLX include:

 - **Composable function transformations**: MLX supports composable function
   transformations for automatic differentiation, automatic vectorization,
   and computation graph optimization.

 - **Lazy computation**: Computations in MLX are lazy. Arrays are only
   materialized when needed.

 - **Dynamic graph construction**: Computation graphs in MLX are constructed
   dynamically. Changing the shapes of function arguments does not trigger
   slow compilations, and debugging is simple and intuitive.

 - **Multi-device**: Operations can run on any of the supported devices
   (currently the CPU and the GPU).

 - **Unified memory**: A notable difference from MLX and other frameworks
   is the *unified memory model*. Arrays in MLX live in shared memory.
   Operations on MLX arrays can be performed on any of the supported
   device types without transferring data.
   
Read more about these [here](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/).

MLX is designed by machine learning researchers for machine learning
researchers. The framework is intended to be user-friendly, but still efficient
to train and deploy models. The design of the framework itself is also
conceptually simple. We intend to make it easy for researchers to extend and
improve MLX with the goal of quickly exploring new ideas. 

The design of MLX is inspired by frameworks like
[NumPy](https://numpy.org/doc/stable/index.html),
[PyTorch](https://pytorch.org/), [Jax](https://github.com/google/jax), and
[ArrayFire](https://arrayfire.org/).

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
