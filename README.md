# mlx-swift
Swift example for MLX


``MLX`` is meant to be built and run from XCode or SwiftPM.  A CMake install is also provided. 

## XCode

In XCode you can add `https://github.com/ml-explore/mlx-swift` as a package
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
XCode.


## CMake

With CMake:
```
mkdir build
cd build
cmake .. -G Ninja
ninja
./example
```
