// Copyright © 2024 Apple Inc.

import Foundation
import XCTest

@testable import MLX

class MLXArrayIndexingTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Subscript (get)

    func testArraySubscriptInt() {
        let a = MLXArray(0 ..< 512, [8, 8, 8])
        let s = a[1]
        XCTAssertEqual(s.ndim, 2)
        XCTAssertEqual(s.shape, [8, 8])

        assertEqual(s, MLXArray(64 ..< 128, [8, 8]))
    }

    func testArraySubscriptIntAxis() {
        let a = MLXArray(0 ..< 512, [8, 8, 8])
        let s = a[1, axis: -1]
        XCTAssertEqual(s.ndim, 2)
        XCTAssertEqual(s.shape, [8, 8])

        // array([[1, 9, 17, ..., 41, 49, 57],
        //        [65, 73, 81, ..., 105, 113, 121],
        //        [129, 137, 145, ..., 169, 177, 185],
        //        ...,
        //        [321, 329, 337, ..., 361, 369, 377],
        //        [385, 393, 401, ..., 425, 433, 441],
        //        [449, 457, 465, ..., 489, 497, 505]], dtype=int64)

        XCTAssertEqual(s[0, 0].item(Int.self), 1)
        XCTAssertEqual(s[0, 1].item(Int.self), 9)
        XCTAssertEqual(s[0, 2].item(Int.self), 17)
    }

    func testArraySubscriptIntArray() {
        // squeeze output dimensions as needed
        let a = MLXArray(0 ..< 512, [8, 8, 8])
        let s1 = a[1, 2]
        XCTAssertEqual(s1.ndim, 1)
        XCTAssertEqual(s1.shape, [8])
        assertEqual(s1, MLXArray(80 ..< 88))

        let s2 = a[1, 2, 3]
        XCTAssertEqual(s2.ndim, 0)
        XCTAssertEqual(s2.shape, [])
        XCTAssertEqual(s2.item(), 64 + 2 * 8 + 3)
    }

    func testArraySubscriptIntArray2() {
        // last dimension should not be squeezed
        let a = MLXArray(0 ..< 512, [8, 8, 8, 1])

        let s = a[1]
        XCTAssertEqual(s.ndim, 3)
        XCTAssertEqual(s.shape, [8, 8, 1])

        let s1 = a[1, 2]
        XCTAssertEqual(s1.ndim, 2)
        XCTAssertEqual(s1.shape, [8, 1])

        let s2 = a[1, 2, 3]
        XCTAssertEqual(s2.ndim, 1)
        XCTAssertEqual(s2.shape, [1])
    }

    func testArraySubscriptFromEnd() {
        // allow negative indicies to indicate distance from end (only for int indicies)
        let a = MLXArray(0 ..< 12, [3, 4])
        let s = a[-1, -2]
        XCTAssertEqual(s.ndim, 0)
        XCTAssertEqual(s.item(), 10)
    }

    func testArraySubscriptRange() {
        let a = MLXArray(0 ..< 512, [8, 8, 8])

        let s1 = a[1 ..< 3]
        XCTAssertEqual(s1.ndim, 3)
        XCTAssertEqual(s1.shape, [2, 8, 8])
        assertEqual(s1, MLXArray(64 ..< 192, [2, 8, 8]))

        // even though the first dimension is 1 we do not squeeze it
        let s2 = a[1 ... 1]
        XCTAssertEqual(s2.ndim, 3)
        XCTAssertEqual(s2.shape, [1, 8, 8])
        assertEqual(s2, MLXArray(64 ..< 128, [1, 8, 8]))

        // multiple ranges, resolving RangeExpressions vs the dimensions
        let s3 = a[1 ..< 2, ..<3, 3...]
        XCTAssertEqual(s3.ndim, 3)
        XCTAssertEqual(s3.shape, [1, 3, 5])
        assertEqual(
            s3, MLXArray([67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 83, 84, 85, 86, 87], [1, 3, 5]))

        let s4 = a[-2 ..< -1, ..<(-3), (-3)...]
        XCTAssertEqual(s4.ndim, 3)
        XCTAssertEqual(s4.shape, [1, 5, 3])
        assertEqual(
            s4,
            MLXArray(
                [389, 390, 391, 397, 398, 399, 405, 406, 407, 413, 414, 415, 421, 422, 423],
                [1, 5, 3]))
    }

    func testArraySubscriptAdvanced() {
        // advanced subscript examples taken from
        // https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

        let a = MLXArray(0 ..< 35, [5, 7]).asType(Int32.self)

        let i1 = MLXArray([0, 2, 4])
        let i2 = MLXArray([0, 1, 2])

        let s1 = a[i1, i2]

        XCTAssertEqual(s1.ndim, 1)
        XCTAssertEqual(s1.shape, [3])

        let expected = MLXArray([0, 15, 30].asInt32)
        assertEqual(s1, expected)
    }

    func testArraySubscriptAdvanced2() {
        let a = MLXArray(0 ..< 12, [6, 2]).asType(Int32.self)

        let i1 = MLXArray([0, 2, 4])
        let s2 = a[i1]

        let expected = MLXArray([0, 1, 4, 5, 8, 9].asInt32, [3, 2])
        assertEqual(s2, expected)
    }

    func testArraySubscriptAdvanced2d() {
        let a = MLXArray(0 ..< 12, [4, 3]).asType(Int32.self)

        let rows = MLXArray([0, 0, 3, 3], [2, 2])
        let cols = MLXArray([0, 2, 0, 2], [2, 2])

        let s = a[rows, cols]

        let expected = MLXArray([0, 2, 9, 11].asInt32, [2, 2])
        assertEqual(s, expected)
    }

    func testArraySubscriptAdvanced2d2() {
        let a = MLXArray(0 ..< 12, [4, 3]).asType(Int32.self)

        let rows = MLXArray([0, 3], [2, 1])
        let cols = MLXArray([0, 2])

        let s = a[rows, cols]

        let expected = MLXArray([0, 2, 9, 11].asInt32, [2, 2])
        assertEqual(s, expected)
    }

    // MARK: - Subscript (set)

    func testArrayMutateSingleIndex() {
        let a = MLXArray((0 ..< 12).asInt32, [3, 4])
        a[1] = [77]

        let expected = MLXArray([0, 1, 2, 3, 77, 77, 77, 77, 8, 9, 10, 11].asInt32, [3, 4])
        assertEqual(a, expected)
    }

    func testArrayMutateBroadcastMultiIndex() {
        let a = MLXArray((0 ..< 20).asInt32, [2, 2, 5])

        // broadcast to a row
        a[1, 0] = MLXArray(77, dtype: .int32)

        // assign to a row
        a[0, 0] = MLXArray([55, 66, 77, 88, 99].asInt32)

        // single element
        a[0, 1, 3] = MLXArray(123, dtype: .int32)

        let expected = MLXArray(
            [55, 66, 77, 88, 99, 5, 6, 7, 123, 9, 77, 77, 77, 77, 77, 15, 16, 17, 18, 19].asInt32,
            [2, 2, 5])
        assertEqual(a, expected)
    }

    func testArrayMutateBroadcastSlice() {
        let a = MLXArray((0 ..< 20).asInt32, [2, 2, 5])

        // write using slices -- this ends up covering two elements
        a[0 ..< 1, 1 ..< 2, 2 ..< 4] = [88]

        let expected = MLXArray(
            [0, 1, 2, 3, 4, 5, 6, 88, 88, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19].asInt32,
            [2, 2, 5])
        assertEqual(a, expected)
    }

    func testArrayMutateAdvanced() {
        let a = MLXArray(0 ..< 35, [5, 7]).asType(Int32.self)

        let i1 = MLXArray([0, 2, 4])
        let i2 = MLXArray([0, 1, 2])

        a[i1, i2] = MLXArray([100, 200, 300].asInt32)

        XCTAssertEqual(a[0, 0].item(), 100.int32)
        XCTAssertEqual(a[2, 1].item(), 200.int32)
        XCTAssertEqual(a[4, 2].item(), 300.int32)
    }

    func testCollection() {
        let a = MLXArray((0 ..< 20).asInt32, [2, 2, 5])

        // enumerate "rows"
        for (i, r) in a.enumerated() {
            let expected = MLXArray((i * 10 ..< i * 10 + 10).asInt32, [2, 5])
            assertEqual(r, expected)
        }
    }

    public func testReversed() {
        // tests for [::-1]
        let a = MLXArray(0 ..< 4)

        // reverse last dimension (only in this case)
        assertEqual(a[stride: -1], MLXArray([3, 2, 1, 0]))

        // reverse first dimension
        assertEqual(a.reshaped(1, 1, 4)[stride: -1, axis: 0], MLXArray(0 ..< 4).reshaped(1, 1, 4))

    }

    public func testStridedBy2() {
        let a = MLXArray(0 ..< (2 * 3 * 4), [2, 3, 4])

        let r = a[stride: 2, axis: -1]
        let expected = MLXArray(Array(stride(from: 0, to: 2 * 3 * 4, by: 2)), [2, 3, 2])
        assertEqual(r, expected)
    }

    public func testStridedBy2Offset() {
        let a = MLXArray(0 ..< (2 * 3 * 5), [2, 3, 5])

        let r = a[from: 1, stride: 2, axis: -1]
        let expected = MLXArray([1, 3, 6, 8, 11, 13, 16, 18, 21, 23, 26, 28], [2, 3, 2])
        assertEqual(r, expected)
    }

    public func testStridedByNegative1Last() {
        let a = MLXArray(0 ..< 6, [2, 3])

        let r = a[stride: -1, axis: -1]
        let expected = MLXArray([2, 1, 0, 5, 4, 3], [2, 3])
        assertEqual(r, expected)
    }

    public func testStridedByNegative1First() {
        let a = MLXArray(0 ..< 6, [2, 3])

        let r = a[stride: -1, axis: 0]
        let expected = MLXArray([3, 4, 5, 0, 1, 2], [2, 3])
        assertEqual(r, expected)
    }

    public func testStridedByNegative1SecondOffset() {
        let a = MLXArray(0 ..< (2 * 3 * 5), [2, 3, 5])

        let r = a[from: 1, stride: -1, axis: 1]
        let expected = MLXArray(
            [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 15, 16, 17, 18, 19], [2, 2, 5])
        assertEqual(r, expected)
    }

    public func testStridedByNegative1LastOffset() {
        let a = MLXArray(0 ..< (2 * 3 * 5), [2, 3, 5])

        let r = a[from: 1, stride: -1, axis: -1]
        let expected = MLXArray([1, 0, 6, 5, 11, 10, 16, 15, 21, 20, 26, 25], [2, 3, 2])
        assertEqual(r, expected)
    }

    public func testStridedByNegative2SecondOffset() {
        let a = MLXArray(0 ..< (2 * 5 * 3), [2, 5, 3])

        let r = a[from: 1, stride: -2, axis: 1]
        let expected = MLXArray([3, 4, 5, 18, 19, 20], [2, 1, 3])
        assertEqual(r, expected)
    }

    public func testStridedByNegative2Last() {
        let a = MLXArray(0 ..< (2 * 3 * 4), [2, 3, 4])

        let r = a[stride: -2, axis: -1]

        // reverse order, stride by 2
        //
        // array([[[3, 1],
        //         [7, 5],
        //         [11, 9]],
        //        [[15, 13],
        //         [19, 17],
        //         [23, 21]]], dtype=int64)

        let expected = MLXArray([3, 1, 7, 5, 11, 9, 15, 13, 19, 17, 23, 21], [2, 3, 2])
        assertEqual(r, expected)
    }

    public func testStridedByNegative2First() {
        let a = MLXArray(0 ..< (2 * 3 * 4), [2, 3, 4])

        let r = a[stride: -2, axis: 0]

        // last row
        //
        // array([[[12, 13, 14, 15],
        //         [16, 17, 18, 19],
        //         [20, 21, 22, 23]]], dtype=int64)

        let expected = MLXArray([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [1, 3, 4])
        assertEqual(r, expected)
    }

    public func testStridedByNegative2SecondSet() {
        let a = MLXArray(Int32(0) ..< (2 * 3 * 4), [2, 3, 4])

        a[stride: -2, axis: 1] = [99, 88, 77, 66]

        // array([[[99, 88, 77, 66],
        //         [4, 5, 6, 7],
        //         [99, 88, 77, 66]],
        //        [[99, 88, 77, 66],
        //         [16, 17, 18, 19],
        //         [99, 88, 77, 66]]], dtype=int64)

        let expected = MLXArray(
            [
                99, 88, 77, 66, 4, 5, 6, 7, 99, 88, 77, 66, 99, 88, 77, 66, 16, 17, 18, 19, 99, 88,
                77, 66,
            ], [2, 3, 4])
        assertEqual(a, expected)
    }

}
