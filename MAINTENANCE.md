# Maintenance Tasks

## Updating Documentation

1. `git checkout gh-pages`
2. `git rebase main`
3. update `Source/MLX/Documentation.docc/Resources/mlx-examples-swift.zip` as needed
4. `./tools/build-documentation.sh`
5. `git add docs`
6. `git commit docs`
7. `git push -f`

## Adding a New Package

Here is adding `MLXFFT`:

1. `Package.swift` add a new product (for anything that should be exported) and target:

```
products: [
    ...
    .library(name: "MLXFFT", targets: ["MLXFFT"]),
```

```
targets: [
    ...
    .target(
        name: "MLXFFT",
        dependencies: ["MLX"]
    ),
```

add to MLXTests:

```
        .testTarget(
            name: "MLXTests",
            dependencies: ["MLX", "MLXRandom", "MLXNN", "MLXOptimizers", "MLXFFT"]
        ),
```

    
2. Update `CMakeLists`
    
```
# MLXFFT package
file(GLOB MLXFFT-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLXFFT/*.swift)
add_library(MLXFFT STATIC ${MLXFFT-src})
target_link_libraries(MLXFFT PRIVATE MLX)
```

3. Create directory in `Source`

4. Add a _Documentation Catalog_

5. Add source files and documentation

6. Add linkage to the other documentation, e.g. in `MLXFFT.md`

```
## Other MLX Packages

- [MLX](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/)
- [MLXRandom](https://ml-explore.github.io/mlx-swift/MLXRandom/documentation/mlxrandom/)
- [MLXNN](https://ml-explore.github.io/mlx-swift/MLXNN/documentation/mlxnn/)
- [MLXOptimizers](https://ml-explore.github.io/mlx-swift/MLXOptimizers/documentation/mlxoptimizers/)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)
```

7. Add linkage to new package in other documentation, e.g. `Documentation/MLX.md`, etc.

```
## Other MLX Packages

...
- [MLXFFT](https://ml-explore.github.io/mlx-swift/MLXFFT/documentation/mlxfft/)
```

8. Update README.md

```
[**Installation**](#installation) | [**MLX**](https://ml-explore.github.io/mlx-swift/MLX/documentation/mlx/) | ... | [**MLXFFT**](https://ml-explore.github.io/mlx-swift/MLXFFT/documentation/mlxfft/) | [**Examples**](#examples) 
```

```
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXRandom", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimziers", package: "mlx-swift"),
               .product(name: "MLXFFT", package: "mlx-swift")]
```

9. Update install.md

```
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXRandom", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimziers", package: "mlx-swift"),
               .product(name: "MLXFFT", package: "mlx-swift")]
```

10. Update `tools/generate_integration_tests.py` as needed

```
import MLXNN
@testable import MLXOptimizers
import MLXFFT
```

11. Update tests as needed

12. Update `tools/build-documentation.sh`

```
for x in MLX MLXRandom MLXNN MLXOptimizers MLXFFT; do
```

13. Run `pre-commit`

```
pre-commit run --all-files
```

14. Make a PR
