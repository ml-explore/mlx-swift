// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX

class MLXArrayTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testArrayProperties() {
        let a = MLXArray(converting: [3.5, 4.5, 5.5, 7.0, 9.4, 10.0], [2, 3, 1])

        XCTAssertEqual(a.itemSize, 4)
        XCTAssertEqual(a.size, 6)
        XCTAssertEqual(a.count, 2)
        XCTAssertEqual(a.nbytes, 6 * 4)
        XCTAssertEqual(a.ndim, 3)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [2, 3, 1])
    }

    func testArrayRead() {
        let a = MLXArray(0 ..< 12, [4, 3])

        XCTAssertEqual(a.asArray(Int32.self), Array(0 ..< 12))
        XCTAssertEqual(a[1][2].item(Int.self), 5)
    }

}
