// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

class MLXRandomTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testSplit() {
        let key = MLXRandom.key(0)
        let keys = MLXRandom.split(key: key, into: 4)

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

    func testRandomStateOrKeySame() {
        // these should all produce the same value since they
        // all resolve to the same key

        let key = MLXRandom.key(0)
        let (_, k1) = split(key: key)

        let state = MLXRandom.RandomState(seed: 0)
        MLXRandom.seed(0)

        // global state
        let v0 = uniform(0 ..< 1, [5])

        // explicit key
        let v1 = uniform(0 ..< 1, [5], key: k1)

        // local RandomState
        let v2 = uniform(0 ..< 1, [5], key: state)

        assertEqual(v0, v1)
        assertEqual(v1, v2)
    }

    func testRandomStateOrKeyDifferent() {
        // these should all produce different values as they
        // use different keys -- note this is otherwise identical
        // to testRandomStateOrKeySame

        let key = MLXRandom.key(7)
        let (_, k1) = split(key: key)

        let state = MLXRandom.RandomState(seed: 11)
        MLXRandom.seed(31)

        // global state
        let v0 = uniform(0 ..< 1, [5])

        // explicit key
        let v1 = uniform(0 ..< 1, [5], key: k1)

        // local RandomState
        let v2 = uniform(0 ..< 1, [5], key: state)

        assertNotEqual(v0, v1)
        assertNotEqual(v1, v2)
        assertNotEqual(v0, v2)
    }

    func testRandomThreadsSame() async {
        // several threads using task local random state with a constant
        // seed will produce the same value
        await withTaskGroup(of: Float.self) { group in
            for _ in 0 ..< 10 {
                group.addTask {
                    let state = MLXRandom.RandomState(seed: 23)
                    return withRandomState(state) {
                        var t: Float = 0.0
                        for _ in 0 ..< 100 {
                            t += uniform(0 ..< 1, [10, 10]).sum().item(Float.self)
                        }
                        return t
                    }
                }
            }

            var result = [Float]()
            for await v in group {
                result.append(v)
            }

            let unique = Set(result)
            XCTAssertEqual(unique.count, 1, "Different values: \(result)")
        }
    }

    func testRandomThreadsDifferent() async {

        // several threads using task local random state with different
        // seeds will produce different values
        await withTaskGroup(of: Float.self) { group in
            for i in 0 ..< 10 {
                group.addTask {
                    let state = MLXRandom.RandomState(seed: UInt64(i))
                    return withRandomState(state) {
                        var t: Float = 0.0
                        for _ in 0 ..< 100 {
                            let x = uniform(0 ..< 1, [10, 10]).sum()
                            asyncEval(x)
                            t += x.item(Float.self)
                        }
                        return t
                    }
                }
            }

            var result = [Float]()
            for await v in group {
                result.append(v)
            }

            let unique = Set(result)
            XCTAssertEqual(unique.count, 10, "Same values: \(result)")
        }
    }

}
