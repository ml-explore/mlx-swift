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

    func testAsArrayContiguous() {
        // read array from contiguous memory
        let a = MLXArray(0 ..< 12, [4, 3])
        let b = a.asArray(Int.self)
        XCTAssertEqual(b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    }

    func testAsArrayNonContiguous1() {
        // skipping elements via slicing
        let a = MLXArray(0 ..< 9, [3, 3])

        let s = a[0 ..< 2, 1 ..< 3]
        assertEqual(s, MLXArray([1, 2, 4, 5], [2, 2]))

        XCTAssertEqual(s.shape, [2, 2])

        // size and nbytes are the logical size
        XCTAssertEqual(s.size, 2 * 2)
        XCTAssertEqual(s.nbytes, 2 * 2 * s.itemSize)

        // internal property for counting the physical size of the backing.
        // note that the physical size doesn't include the row that is
        // sliced out
        XCTAssertEqual(s.physicalSize, 3 * 2)

        // evaluating s (the comparison above) will realize the strides.
        // if we eamine these before they might be [2, 1] which are the
        // "logical" strides
        XCTAssertEqual(s.strides, [3, 1])

        let s_arr = s.asArray(Int32.self)
        XCTAssertEqual(s_arr, [1, 2, 4, 5])
    }

    func testAsArrayNonContiguous2() {
        // a transpose via strides
        let a = MLXArray(0 ..< 12, [4, 3])

        let s = asStrided(a, [3, 4], strides: [1, 3])

        let expected: [Int32] = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        assertEqual(s, MLXArray(expected, [3, 4]))

        // Note: be careful to use the matching type -- if we transcode
        // to a different type it will be converted to contiguous
        let s_arr = s.asArray(Int32.self)
        XCTAssertEqual(s_arr, expected)
    }

    func testAsArrayNonContiguous3() {
        // reversed via strides -- note that the base pointer for the
        // storage has an offset applied to it
        let a = MLXArray(0 ..< 9, [3, 3])

        let s = asStrided(a, [3, 3], strides: [-3, -1], offset: 8)

        let expected: [Int32] = [8, 7, 6, 5, 4, 3, 2, 1, 0]
        assertEqual(s, MLXArray(expected, [3, 3]))

        let s_arr = s.asArray(Int32.self)
        XCTAssertEqual(s_arr, expected)
    }

}
