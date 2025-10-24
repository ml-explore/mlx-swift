See `xcode/MLX.xcodeproj` and [MAINTENANCE.md].

# Cmlx

This is set up to build roughly how Package.swift builds.

- Look at Project -> Cmlx -> Build Phases
- remove all Project headers
- remove all Copy Bundle Resources
- remove any files that should not be built from the Target membership, e.g the items in `exclude`

Public headers are in `include-framework` and this is managed by tools/update-mlx

Settings, including header search paths are in xcode/xcconfig.

## Updating

After updating the mlx/mlx-c version the xcodeproj needs to be brought up to date.  

- the headers in Cmlx/include-framework must all be public
- no other headers in the project should be included as resources (public/private/project)
    - the easiest way to adjust is look at Project -> Cmlx -> Build Phases and then look at the Headers task
- similarly there should be _no_ Copy Bundle Resources from the same section
- compilation issues in .metal files typically mean they are new to the project and need to be removed from Cmlx target membership

# MLX, etc.

These are just normal frameworks that link to Cmlx and others as needed.  The source files are all swift and there are no special settings needed.
