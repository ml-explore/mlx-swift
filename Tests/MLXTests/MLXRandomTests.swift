// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct MLXRandomTests {

    @Test func splitKey() {
        let key = MLXRandom.key(0)
        let keys = MLXRandom.split(key: key, into: 4)

        #expect(keys.count == 4)
    }

    @Test func uniformSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.uniform(0 ..< 10, key: key).item(Float.self)
        expectEqual(value, 4.18, accuracy: 0.01)
    }

    @Test func uniformMultiple() {
        let key = MLXRandom.key(0)

        // specify shape to broadcast
        let value = MLXRandom.uniform(0.0 ..< 10, [3], key: key)
        let expected = MLXArray(converting: [9.65, 3.14, 6.33])
        expectEqual(value, expected, atol: 0.01)
    }

    @Test func uniformMultipleArray() {
        let key = MLXRandom.key(0)

        // give an array of bounds
        let value = MLXRandom.uniform(low: [0, 10], high: [10, 100], key: key)

        let expected = MLXArray(converting: [2.16, 82.37])
        expectEqual(value, expected, atol: 0.01)
    }

    @Test func normalSingle() {
        let key = MLXRandom.key(0)

        let value = MLXRandom.normal(key: key).item(Float.self)
        expectEqual(value, -0.20, accuracy: 0.01)
    }

    @Test func randIntSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.randInt(0 ..< 100, key: key).item(Int.self)
        #expect(value == 41)
    }

    @Test func randIntMultiple() {
        let key = MLXRandom.key(0)

        let value = MLXRandom.randInt(low: [0, 10], high: [10, 100], key: key)

        let expected: [Int32] = [2, 82]
        expectEqual(value, MLXArray(expected))
    }

    @Test func randIntMultipleType() {
        let key = MLXRandom.key(0)

        let value = MLXRandom.randInt(low: [0, 10], high: [10, 100], type: Int8.self, key: key)

        let expected: [Int8] = [2, 82]
        expectEqual(value, MLXArray(expected))
    }

    @Test func bernoulliSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.bernoulli(key: key).item(Bool.self)
        #expect(value == true)
    }

    @Test func bernoulliMultiple() {
        let key = MLXRandom.key(0)
        let array = MLXRandom.bernoulli([4], key: key)
        let expected = MLXArray([false, true, false, true])
        expectEqual(array, expected)
    }

    @Test func bernoulliP() {
        let key = MLXRandom.key(0)
        let array = MLXRandom.bernoulli(0.8, [4], key: key)
        let expected = MLXArray([false, true, true, true])
        expectEqual(array, expected)
    }

    @Test func bernoulliPArray() {
        let key = MLXRandom.key(0)
        let array = MLXRandom.bernoulli(MLXArray(converting: [0.1, 0.5, 0.8]), key: key)
        let expected = MLXArray([false, true, true])
        expectEqual(array, expected)
    }

    @Test func truncatedNormalSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.truncatedNormal(0 ..< 10, key: key).item(Float.self)
        expectEqual(value, 0.55, accuracy: 0.01)
    }

    @Test func truncatedNormalMultiple() {
        let key = MLXRandom.key(0)

        // specify shape to broadcast
        let value = MLXRandom.truncatedNormal(0 ..< 0.5, [3], key: key)
        let expected = MLXArray(converting: [0.48, 0.15, 0.30])
        expectEqual(value, expected, atol: 0.01)
    }

    @Test func truncatedNormalMultipleArray() {
        let key = MLXRandom.key(0)

        // give an array of bounds
        let value = MLXRandom.truncatedNormal(
            low: MLXArray(converting: [0, 0.5]), high: MLXArray(converting: [0.5, 1]), key: key)

        let expected = MLXArray(converting: [0.10, 0.88])
        expectEqual(value, expected, atol: 0.01)
    }

    @Test func gumbelSingle() {
        let key = MLXRandom.key(0)

        let value = MLXRandom.gumbel(key: key).item(Float.self)
        expectEqual(value, 0.13, accuracy: 0.01)
    }

    @Test func categoricalLogits() {
        let key = MLXRandom.key(0)

        let logits = MLXArray.zeros([5, 20])
        let result = MLXRandom.categorical(logits, key: key)

        #expect(result.shape == [5])
        #expect(result.dtype == .uint32)

        let expected = MLXArray([1, 1, 17, 17, 17])
        expectEqual(result, expected)
    }

    @Test func categoricalLogitsCount() {
        let key = MLXRandom.key(0)

        let logits = MLXArray.zeros([5, 20])
        let result = MLXRandom.categorical(logits, count: 2, key: key)

        #expect(result.shape == [5, 2])
        #expect(result.dtype == .uint32)

        let expected = MLXArray([16, 3, 14, 10, 17, 7, 6, 8, 12, 8], [5, 2])
        expectEqual(result, expected)
    }

    @Test func randomStateOrKeySame() {
        // these should all produce the same value since they
        // all resolve to the same key
        //
        // note: this does not use the global state (MLXRandom.seed()) -- it is
        // shared by every test in the process and other tests running in parallel
        // may draw from it.  MLXRandom.seed(n) is MLXRandom.globalState.seed(n),
        // which is covered by v3 below.

        let key = MLXRandom.key(0)
        let (_, k1) = split(key: key)

        let state = MLXRandom.RandomState(seed: 0)

        // task local RandomState (the fallback when no key is given)
        let v0 = withRandomState(MLXRandom.RandomState(seed: 0)) {
            uniform(0 ..< 1, [5])
        }

        // explicit key
        let v1 = uniform(0 ..< 1, [5], key: k1)

        // local RandomState
        let v2 = uniform(0 ..< 1, [5], key: state)

        // a RandomState that is re-seeded, the same as MLXRandom.seed(0) does
        // to the global state
        let reseeded = MLXRandom.RandomState(seed: 99)
        _ = uniform(0 ..< 1, [5], key: reseeded)
        reseeded.seed(0)
        let v3 = uniform(0 ..< 1, [5], key: reseeded)

        expectEqual(v0, v1)
        expectEqual(v1, v2)
        expectEqual(v2, v3)
    }

    @Test func randomStateOrKeyDifferent() {
        // these should all produce different values as they
        // use different keys -- note this is otherwise identical
        // to randomStateOrKeySame

        let key = MLXRandom.key(7)
        let (_, k1) = split(key: key)

        let state = MLXRandom.RandomState(seed: 11)

        // task local RandomState
        let v0 = withRandomState(MLXRandom.RandomState(seed: 31)) {
            uniform(0 ..< 1, [5])
        }

        // explicit key
        let v1 = uniform(0 ..< 1, [5], key: k1)

        // local RandomState
        let v2 = uniform(0 ..< 1, [5], key: state)

        expectNotEqual(v0, v1)
        expectNotEqual(v1, v2)
        expectNotEqual(v0, v2)
    }

    @Test func randomThreadsSame() async {
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
            #expect(unique.count == 1, "Different values: \(result)")
        }
    }

    @Test func randomThreadsDifferent() async {

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
            #expect(unique.count == 10, "Same values: \(result)")
        }
    }

}
