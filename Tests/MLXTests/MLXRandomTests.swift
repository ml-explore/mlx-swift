// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXRandom

class MLXRandomTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testSplit() {
        let key = MLXRandom.key(0)
        let keys = split(key: key, into: 4)

        XCTAssertEqual(keys.count, 4)
    }

    func testUniformSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.uniform(0 ..< 10, key: key).item(Float.self)
        XCTAssertEqual(value, 4.18, accuracy: 0.01)
    }

    func testUniformMultiple() {
        let key = MLXRandom.key(0)

        // specify shape to broadcast
        let value = MLXRandom.uniform(0.0 ..< 10, [3], key: key)
        let expected = MLXArray(converting: [9.65, 3.14, 6.33])
        assertEqual(value, expected, atol: 0.01)
    }

    func testUniformMultipleArray() {
        let key = MLXRandom.key(0)

        // give an array of bounds
        let value = MLXRandom.uniform(low: [0, 10], high: [10, 100], key: key)

        let expected = MLXArray(converting: [2.16, 82.37])
        assertEqual(value, expected, atol: 0.01)
    }

    func testNormal() {
        let key = MLXRandom.key(0)

        let value = MLXRandom.normal(key: key).item(Float.self)
        XCTAssertEqual(value, -0.20, accuracy: 0.01)
    }

    func testRandIntSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.randInt(0 ..< 100, key: key).item(Int.self)
        XCTAssertEqual(value, 41)
    }

    func testRandIntMultiple() {
        let key = MLXRandom.key(0)

        let value = MLXRandom.randInt(low: [0, 10], high: [10, 100], key: key)

        let expected: [Int32] = [2, 82]
        assertEqual(value, MLXArray(expected))
    }

    func testRandIntMultipleType() {
        let key = MLXRandom.key(0)

        let value = MLXRandom.randInt(low: [0, 10], high: [10, 100], type: Int8.self, key: key)

        let expected: [Int8] = [2, 82]
        assertEqual(value, MLXArray(expected))
    }

    func testBernoulliSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.bernoulli(key: key).item(Bool.self)
        XCTAssertEqual(value, true)
    }

    func testBernoulliMultiple() {
        let key = MLXRandom.key(0)
        let array = MLXRandom.bernoulli([4], key: key)
        let expected = MLXArray([false, true, false, true])
        assertEqual(array, expected)
    }

    func testBernoulliP() {
        let key = MLXRandom.key(0)
        let array = MLXRandom.bernoulli(0.8, [4], key: key)
        let expected = MLXArray([false, true, true, true])
        assertEqual(array, expected)
    }

    func testBernoulliPArray() {
        let key = MLXRandom.key(0)
        let array = MLXRandom.bernoulli(MLXArray(converting: [0.1, 0.5, 0.8]), key: key)
        let expected = MLXArray([false, true, true])
        assertEqual(array, expected)
    }

    func testTruncatedNormalSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.truncatedNormal(0 ..< 10, key: key).item(Float.self)
        XCTAssertEqual(value, 0.55, accuracy: 0.01)
    }

    func testTruncatedNormalMultiple() {
        let key = MLXRandom.key(0)

        // specify shape to broadcast
        let value = MLXRandom.truncatedNormal(0 ..< 0.5, [3], key: key)
        let expected = MLXArray(converting: [0.48, 0.15, 0.30])
        assertEqual(value, expected, atol: 0.01)
    }

    func testTruncatedNormalMultipleArray() {
        let key = MLXRandom.key(0)

        // give an array of bounds
        let value = MLXRandom.truncatedNormal(
            low: MLXArray(converting: [0, 0.5]), high: MLXArray(converting: [0.5, 1]), key: key)

        let expected = MLXArray(converting: [0.10, 0.88])
        assertEqual(value, expected, atol: 0.01)
    }

    func testGumbel() {
        let key = MLXRandom.key(0)

        let value = MLXRandom.gumbel(key: key).item(Float.self)
        XCTAssertEqual(value, 0.13, accuracy: 0.01)
    }

    func testLogits() {
        let key = MLXRandom.key(0)

        let logits = MLXArray.zeros([5, 20])
        let result = MLXRandom.categorical(logits, key: key)

        XCTAssertEqual(result.shape, [5])
        XCTAssertEqual(result.dtype, .uint32)

        let expected = MLXArray([1, 1, 17, 17, 17])
        assertEqual(result, expected)
    }

    func testLogitsCount() {
        let key = MLXRandom.key(0)

        let logits = MLXArray.zeros([5, 20])
        let result = MLXRandom.categorical(logits, count: 2, key: key)

        XCTAssertEqual(result.shape, [5, 2])
        XCTAssertEqual(result.dtype, .uint32)

        let expected = MLXArray([16, 3, 14, 10, 17, 7, 6, 8, 12, 8], [5, 2])
        assertEqual(result, expected)
    }

}
