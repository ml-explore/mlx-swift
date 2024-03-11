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

- [MLX](mlx)
- [MLXRandom](mlxrandom)
- [MLXNN](mlxnn)
- [MLXOptimizers](mlxoptimizers)
- [MLXFFT](mlxfft)
- [MLXLinalg](mlxlinalg)
- [MLXFast](mlxfast)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)
```

7. Add linkage to new package in other documentation, e.g. `Documentation/MLX.md`, etc.

```
## Other MLX Packages

...
- [MLXFFT](../mlxfft/)
```

8. Update README.md

```
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXRandom", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimizers", package: "mlx-swift"),
               .product(name: "MLXFFT", package: "mlx-swift")]
```

9. Update install.md

```
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXRandom", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimizers", package: "mlx-swift"),
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

13. Add to `.spi.yml` for swift package index

14. Run `pre-commit`

```
pre-commit run --all-files
```

14. Make a PR

## Updating `mlx` and `mlx-c`

SwiftPM is able to fetch repositories from github and build them _if_ they have
a `Package.swift` at the top level.  It is unable to do this for repositories
that do not have a `Package.swift`.  For this reason `mlx-swift` uses
git submodules to include the `mlx` and `mlx-c` repositories.

When a new version of `mlx` and its equivalent `mlx-c` are to be used, there is a
process to go through to update `mlx-swift`.

Additionally, SwiftPM supports plugins that can produce derived source for
building, but this can only produce new swift source.  It is possible to use
plugins to generate new source `.cpp` files and even compile them, but at
best the `.o` is copied into the output as a resource, not linked.
This is important because `mlx` has some build-time source generation
(e.g. `make_compiled_preamble.sh`).  This is handled in `mlx-swift` by
pre-generating the source when updating the `mlx` version.

1. Update the `mlx` and `mlx-c` submodules via `git pull` or `git checkout ...`
    - `Source/Cmlx/mlx`
    - `Source/Cmlx/mlx-c`
    
2. Add any vendored dependencies as needed in `/vendor`

3. Regenerate any build-time source: `./tools/update-mlx.sh`

4. Fix any build issues

5. Wrap any new API with swift, update documentation, etc.

6. Run `pre-commit run --all-files`

7. Make a PR
