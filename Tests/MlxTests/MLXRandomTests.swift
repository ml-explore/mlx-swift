import Foundation
import XCTest
@testable import Mlx

class MLXRandomTests : XCTestCase {
    
    func testUniformSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.uniform(Float(0.0) ..< 10.0, key: key).item(Float.self)
        XCTAssertEqual(value, 4.18, accuracy: 0.1)
    }
    
    func testUniformMultiple() {
        let key = MLXRandom.key(0)
        
        // specify shape to broadcast
        let value = MLXRandom.uniform(Float(0.0) ..< 10.0, [3], key: key)
        let expected: [Float] = [9.65, 3.14, 6.33]
        assertEqual(value, MLXArray(expected), atol: 0.1)
    }

    func testUniformMultipleArray() {
        let key = MLXRandom.key(0)
        
        // give an array of bounds
        let value = MLXRandom.uniform([0, 10], [10, 100], type: Float.self, key: key)
        
        let expected: [Float] = [2.16, 82.37]
        assertEqual(value, MLXArray(expected), atol: 0.1)
    }

    func testNormal() {
        let key = MLXRandom.key(0)
        
        let value = MLXRandom.normal(key: key).item(Float.self)
        XCTAssertEqual(value, -0.20, accuracy: 0.1)
    }

    func testRandIntSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.randInt(0 ..< 100, key: key).item(Int.self)
        XCTAssertEqual(value, 41)
    }

    func testBernoulliSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.bernoulli(key: key).item(Bool.self)
        XCTAssertEqual(value, true)
    }

    func testTruncatedNormalSingle() {
        let key = MLXRandom.key(0)
        let value = MLXRandom.truncatedNormal(Float(0.0) ..< 10.0, key: key).item(Float.self)
        XCTAssertEqual(value, 0.55, accuracy: 0.1)
    }

}
