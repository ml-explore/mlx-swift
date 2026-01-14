#!/bin/sh

BUILD_DIR=`xcodebuild -configuration Debug -showBuildSettings -scheme MLX | grep 'BUILT_PRODUCTS_DIR = /' | sed -e 's/^[^=]*= //g' | head -1`

# rpath points to PackageFrameworks so link it to the built products
(cd $BUILD_DIR/PackageFrameworks; ln -s ../*.framework .)

xcrun xctest "$BUILD_DIR/MLXTests.xctest"
