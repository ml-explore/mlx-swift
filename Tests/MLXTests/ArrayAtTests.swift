// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import XCTest
import Cmlx

class ArrayAtTests: XCTestCase {
    override class func setUp() {
        super.setUp()

        // Cmlx is not an automatically loaded bundle. 
        // Let's load Cmlx manually, so that the mlx-lib can find the default.metallib.
        // This requires patch #2885 in ml-explore/mlx be applied.

        // Find the test bundle (.xctest)
        let testBundle = Bundle(for: self)

        // Go to its parent directory: …/debug
        let buildDir = testBundle.bundleURL.deletingLastPathComponent()

        // Append the actual SwiftPM bundle name for Cmlx
        let cmlxURL = buildDir.appendingPathComponent("mlx-swift_Cmlx.bundle")

        if let cmlxBundle = Bundle(url: cmlxURL) {
            print("Loaded Cmlx bundle at:", cmlxURL.path)
            _ = cmlxBundle.load()  // registers it globally
        } else {
            print("Failed to load Cmlx bundle at:", cmlxURL.path)
        }

    }

    func testArrayAt() {
        // from example at https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.at.html#mlx.core.array.at

        // this references each index twice
        let idx = MLXArray([0, 1, 0, 1])

        // assign through index -- we can only observe the last assignment to a location
        let a0 = MLXArray([0, 0])
        a0[idx] = MLXArray(2)
        assertEqual(a0, MLXArray([2, 2]))

        // similar to above -- we can only observe one assignment, so we just get a +1
        // note: there was a bug in the += operator where the lhs was not inout and
        // this was producing [0, 0]
        let a1 = MLXArray([0, 0])
        a1[idx] += 1
        assertEqual(a1, MLXArray([1, 1]))

        // the bare add produces a value for each index including the duplicates
        let a2 = MLXArray([0, 0])
        assertEqual(a2[idx] + 1, MLXArray([1, 1, 1, 1]))

        // but the assign back through the index will collapse the values down
        // into the same location -- this is the same as a2[idx] += 1
        a2[idx] = a2[idx] + 1
        assertEqual(a2, MLXArray([1, 1]))

        // this will update 0 and 1 twice
        let a3 = MLXArray([0, 0])
        assertEqual(a3.at[idx].add(1), MLXArray([2, 2]))
    }
}
