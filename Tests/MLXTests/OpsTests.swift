// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX

class OpsTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testAsStridedReshape() {
        // just changing the shape and using the default strides is the same as reshape
        let a = MLXArray(0 ..< 12, [4, 3])

        // this uses [4, 1] as the strides
        let b = asStrided(a, [3, 4])
        assertEqual(b, a.reshaped([3, 4]))

        let c = asStrided(a, [3, 4], strides: [4, 1])
        assertEqual(b, c)
    }

    func testAsStridedTranspose() {
        // strides in the reverse order is a transpose
        let a = MLXArray(0 ..< 12, [4, 3])

        let b = asStrided(a, [3, 4], strides: [1, 3])
        assertEqual(b, a.transposed())
    }

    func testAsStridedOffset() {
        let a = MLXArray(0 ..< 16, [4, 4])

        let b = asStrided(a, [3, 4], offset: 1)
        assertEqual(b, MLXArray(1 ..< 13, [3, 4]))
    }

    func testAsStridedReverse() {
        let a = MLXArray(0 ..< 16, [4, 4])
        let expected = MLXArray((0 ..< 16).reversed(), [4, 4])

        let b = asStrided(a, [4, 4], strides: [-4, -1], offset: 15)
        assertEqual(b, expected)
    }

}
