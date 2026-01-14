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

Here is adding `MLXOptimizers`:

1. `Package.swift` add a new product (for anything that should be exported) and target:

```
products: [
    ...
    .library(name: "MLXOptimizers", targets: ["MLXOptimizers"]),
```

```
targets: [
    ...
    .target(
        name: "MLXOptimizers",
        dependencies: ["MLX"]
    ),
```

add to MLXTests:

```
        .testTarget(
            name: "MLXTests",
            dependencies: ["MLX", "MLXNN", "MLXOptimizers"]
        ),
```

2. Update `CMakeLists`

```
# MLXOptimizers package
file(GLOB MLXOptimizers-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLXOptimizers/*.swift)
add_library(MLXOptimizers STATIC ${MLXOptimizers-src})
target_link_libraries(MLXOptimizers PRIVATE MLX)
```

3. Create directory in `Source`

4. Add a _Documentation Catalog_

5. Add source files and documentation

6. Add linkage to the other documentation, e.g. in `MLXOptimizers.md`

```
## Other MLX Packages

- [MLX](mlx)
- [MLXNN](mlxnn)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)
```

7. Add linkage to new package in other documentation, e.g. `Documentation/MLX.md`, etc.

```
## Other MLX Packages

...
- [MLXOptimizers](../mlxoptimizers/)
```

8. Update README.md

```
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimizers", package: "mlx-swift")]
```

9. Update install.md

```
dependencies: [.product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimizers", package: "mlx-swift")]
```

10. Update `tools/generate_integration_tests.py` as needed

```
import MLXNN
@testable import MLXOptimizers
```

11. Update tests as needed

12. Update `tools/build-documentation.sh`

```
for x in MLX MLXNN MLXOptimizers; do
```

13. Add to `.spi.yml` for swift package index

14. Run `pre-commit`

```
pre-commit run --all-files
```

14. Make a PR

## Updating `mlx` and `mlx-c`

SwiftPM is able to fetch repositories from github and build them _if_ they have
a `Package.swift` at the top level. It is unable to do this for repositories
that do not have a `Package.swift`. For this reason `mlx-swift` uses
git submodules to include the `mlx` and `mlx-c` repositories.

When a new version of `mlx` and its equivalent `mlx-c` are to be used, there is a
process to go through to update `mlx-swift`.

Additionally, SwiftPM supports plugins that can produce derived source for
building, but this can only produce new swift source. It is possible to use
plugins to generate new source `.cpp` files and even compile them, but at
best the `.o` is copied into the output as a resource, not linked.
This is important because `mlx` has some build-time source generation
(e.g. `make_compiled_preamble.sh`). This is handled in `mlx-swift` by
pre-generating the source when updating the `mlx` version.

1. Update the `mlx` and `mlx-c` submodules via `git pull` or `git checkout ...`
   - `Source/Cmlx/mlx`
   - `Source/Cmlx/mlx-c`
2. Add any vendored dependencies as needed in `/vendor`

3. Regenerate any build-time source: `./tools/update-mlx.sh`
    - this updates headers in Source/Cmlx/include
    - this updates headers in Source/Cmlx/include-framework
    - this generates various files in Source/Cmlx/mlx-generated

4. Fix any build issues with SwiftPM build (opening Package.swift)
5. Fix any build issues with xcodeproj build (opening xcode/MLX.codeproj), see also [README.xcodeproj.md]

6. Wrap any new API with swift, update documentation, etc.

7. Run `pre-commit run --all-files`

8. Make a PR

## Updating `xcode/MLX.xcodeproj`

### Updating

After updating the mlx/mlx-c version the xcodeproj needs to be brought up to date.  

- the headers in Cmlx/include-framework must all be public
- no other headers in the project should be included as resources (public/private/project)
    - the easiest way to adjust is look at Project -> Cmlx -> Build Phases and then look at the Headers task
- similarly there should be _no_ Copy Bundle Resources from the same section
- compilation issues in .metal files typically mean they are new to the project and need to be removed from Cmlx target membership

### Cmlx

This is set up to build roughly how Package.swift builds.

- Look at Project -> Cmlx -> Build Phases
- remove all Project headers
- remove all Copy Bundle Resources
- remove any files that should not be built from the Target membership, e.g the items in `exclude`

Public headers are in `include-framework` and this is managed by tools/update-mlx

Settings, including header search paths are in xcode/xcconfig.

### MLX, etc.

These are just normal frameworks that link to Cmlx and others as needed.  The source files are all swift and there are no special settings needed.

