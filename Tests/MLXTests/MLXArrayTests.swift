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
        // if we examine these before they might be [2, 1] which are the
        // "logical" strides
        XCTAssertEqual(s.internalStrides, [3, 1])

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

    func testAsArrayNonContiguous4() {
        // buffer with holes (last dimension has stride of 2 and
        // thus larger storage than it physically needs)
        let a = MLXArray(0 ..< 16, [4, 4])
        let s = a[0..., .stride(by: 2)]

        let expected: [Int32] = [0, 2, 4, 6, 8, 10, 12, 14]
        assertEqual(s, MLXArray(expected, [4, 2]))

        XCTAssertEqual(s.internalStrides, [4, 2])

        let s_arr = s.asArray(Int32.self)
        XCTAssertEqual(s_arr, expected)
    }

    func testContiguousStrides() {
        XCTAssertEqual(contiguousStrides(shape: [1, 1, 1]), [1, 1, 1])
        XCTAssertEqual(contiguousStrides(shape: [4, 4]), [4, 1])
        XCTAssertEqual(contiguousStrides(shape: [4, 2]), [2, 1])
        XCTAssertEqual(contiguousStrides(shape: [3, 2, 1, 5]), [10, 5, 5, 1])
    }

    func testAsDataContiguous() {
        // contiguous source
        let a = MLXArray(0 ..< 16, [4, 4])

        let result = a.asData()
        XCTAssertEqual(result.shape, [4, 4])
        XCTAssertEqual(result.strides, [4, 1])
        XCTAssertEqual(result.dType, .int32)
    }

    func testAsDataContiguousNoCopy() {
        // contiguous source
        let a = MLXArray(0 ..< 16, [4, 4])

        do {
            let result = a.asData(access: .noCopy)
            XCTAssertEqual(result.shape, [4, 4])
            XCTAssertEqual(result.strides, [4, 1])
            XCTAssertEqual(result.dType, .int32)
        }
        do {
            let result = a.asData(access: .noCopyIfContiguous)
            XCTAssertEqual(result.shape, [4, 4])
            XCTAssertEqual(result.strides, [4, 1])
            XCTAssertEqual(result.dType, .int32)
        }
    }

    func testAsDataRoundTrip() {
        let a = MLXArray(0 ..< 16, [4, 4])
        let arrayData = a.asData(access: .copy)
        let result = MLXArray(arrayData.data, arrayData.shape, dtype: arrayData.dType)
        assertEqual(a, result)
    }

    func testAsDataNonContiguous() {
        // buffer with holes (last dimension has stride of 2 and
        // thus larger storage than it physically needs)
        let a = MLXArray(0 ..< 16, [4, 4])
        let s = a[0..., .stride(by: 2)]

        let result = s.asData()
        XCTAssertEqual(result.shape, [4, 2])
        XCTAssertEqual(result.strides, [2, 1])
        XCTAssertEqual(result.dType, .int32)
    }

    func testAsDataNonContiguousNoCopy() {
        // buffer with holes (last dimension has stride of 2 and
        // thus larger storage than it physically needs)
        let a = MLXArray(0 ..< 16, [4, 4])
        let s = a[0..., .stride(by: 2)]

        do {
            // the strides will match the strided array
            let result = s.asData(access: .noCopy)
            XCTAssertEqual(result.shape, [4, 2])
            XCTAssertEqual(result.strides, [4, 2])
            XCTAssertEqual(result.dType, .int32)
        }
        do {
            // it isn't contiguous so we will get a contiguous copy
            let result = s.asData(access: .noCopyIfContiguous)
            XCTAssertEqual(result.shape, [4, 2])
            XCTAssertEqual(result.strides, [2, 1])
            XCTAssertEqual(result.dType, .int32)
        }
    }

}
