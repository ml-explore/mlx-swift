#  ``MLX``

MLX is a NumPy-like array framework designed for efficient and flexible machine
learning on Apple silicon, brought to you by Apple machine learning research.

The Swift API closely follows the 
[C++ and Python MLX APIs](https://ml-explore.github.io/mlx/build/html/index.html), which in turn closely follow
NumPy with a few exceptions. Here are some useful pages showing how MLX Swift works and is different
from python:

- <doc:converting-python> -- information about converting Python code and differences between Python and Swift
- <doc:indexing> -- information about array indexing
- <doc:arithmetic> -- information about array arithmetic

The main differences between MLX and NumPy are:

 - **Composable function transformations**: MLX has composable function
   transformations for automatic differentiation, automatic vectorization,
   and computation graph optimization.
 - **<doc:lazy-evaluation>**: Computations in MLX are lazy. Arrays are only
   materialized when needed.
 - **Multi-device**: Operations can run on any of the supported devices (CPU,
   GPU, ...)

The design of MLX is inspired by frameworks like 
[PyTorch](https://pytorch.org/), [Jax](https://github.com/google/jax), and
[ArrayFire](https://arrayfire.org/). A notable difference from these
frameworks and MLX is the <doc:unified-memory>. Arrays in MLX live in shared
memory. Operations on MLX arrays can be performed on any of the supported
device types without performing data copies. Currently supported device types
are the CPU and GPU.

## Other MLX Packages

- [MLXRandom](https://ml-explore.github.io/mlx-swift/MLXRandom/documentation/mlxrandom/)
- [MLXNN](https://ml-explore.github.io/mlx-swift/MLXNN/documentation/mlxnn/)
- [MLXOptimizers](https://ml-explore.github.io/mlx-swift/MLXOptimizers/documentation/mlxoptimizers/)
- [MLXFFT](https://ml-explore.github.io/mlx-swift/MLXFFT/documentation/mlxfft/)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)


## Topics

### MLX

- <doc:install>
- <doc:troubleshooting>
- <doc:examples>
- <doc:converting-python>
- <doc:broadcasting>
- <doc:lazy-evaluation>
- <doc:unified-memory>
- <doc:compilation>
- <doc:using-streams>

### MLXArray

- ``MLXArray``

### Free Functions

- <doc:free-functions>

### Data Types

- ``DType``
- ``HasDType``
- ``ScalarOrArray``

### Nested Data

- ``NestedDictionary``
- ``NestedItem``
- ``IndentedDescription``

### Streams and Devices

- ``StreamOrDevice``
- ``Device``
- ``DeviceType``
- ``Stream``
