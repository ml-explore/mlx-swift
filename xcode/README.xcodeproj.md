# Alternate Build Method - xcodeproj

Note that the SwiftPM and XCode (1) methods build `MLX` as a Library, not as a framework.
It is possible to construct a situation where YourApp -> MLX, YourApp -> YourFramework
and YourFramework -> MLX.  This would give two copies of MLX in the resulting process
and it may not work as expected.

If this cannot be avoided, either by making YourFramework a Library or having YourApp
_not_ link MLX, you can use the `xcode/MLX.xcodeproj` to build MLX as a _Framework_.
This will require `mlx-swift` to be checked out adjacent or inside your project,
possibly using git submodules, and dragging the `mlx-swift/xcode/MLX.xcodeproj` into
your project.  Once that is done your application can build and link MLX and related
as Frameworks.


## Maintenance

See [MAINTENANCE.md](../MAINTENANCE.md) for information about how to
update the xcodeproj when incorporating a new version of mlx/mlx-c.
