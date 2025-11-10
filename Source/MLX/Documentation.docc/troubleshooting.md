# Troubleshooting

Help with problems you might run into.

## Installation

See <doc:install> for more information.  If you use Xcode or SwiftPM to reference
the package on github.com there should be no manual steps.

If you clone the `mlx-swift` repository yourself, e.g. in order to
work on the implementation please be aware that it uses git submodules.

When you clone the repository you must give an argument to check out the submodules at the same time:

```
git clone --recurse-submodules https://github.com/ml-explore/mlx-swift.git
```

If you have already cloned the repository you can use this command to force the submodules to update:

```
git submodule update --init --recursive
```

## Running From Command Line

> MLX requires metal shaders from the `Cmlx` framework -- these are not usable
from command line tools unless `DYLD_FRAMEWORK_PATH` makes them visible.

``MLX`` is built on top of the `mlx` C++ library packaged in the `Cmlx` swift package.  `Cmlx`
produces a bundle called `mlx-swift_Cmlx.bundle` which contains the compiled metal shaders.
If you build a command line tool and run it from Xcode, the `DYLD_FRAMEWORK_PATH` is set 
so that this bundle is found.

If you want to run the same command from the shell you must manually set the `DYLD_FRAMEWORK_PATH`
to the build directory.

Note: applications will link the `MLX` and `Cmlx` libraries which will automatically
provide access to the metal libraries as a resource of your application.

[mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) contains a wrapper script, `mlx-run`
that can be used to run the example command line tools:

```
./mlx-run llm-tool --help
```

## Building

Specific build issues you may encounter.

### The file "kernels" couldn't be opened ...

This message comes from part of the build process that prepares the metal kernels.
This may mean that the `kernels` directory in `Source/Cmlx/mlx/mlx/backend/metal/kernels`
does not exist.  In particular it may mean that the git submodules were not checked out -- see "Installation" above.

### received multiple target ended messages for target ID ...

If you receive a message like this:

```
error: Internal inconsistency error: received multiple target ended messages for target ID '5' or received target ended message but did not receive corresponding target started message, while retrieving parent activity in taskStarted message.
```

There are a few approaches that have been observed to work around the issue:

- wait a few seconds and try building again
- quit and restart xcode
- clean the build folder and rebuild
- use Xcode 15.3 beta 2 or later

## Porting Python Code

See <doc:converting-python> for examples and information about
symbol changes from python to swift.  [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
contains information about porting models from MLX (Python) to Swift.

## Running on iOS Simulator

It isn't possible to use the iOS simulator for developing MLX applications, since MLX requires
a modern [Metal MTLGPUFamily](https://developer.apple.com/documentation/metal/mtlgpufamily)
and the simulator does not provide that.

See <doc:running-on-ios>.
