// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

class LinalgTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testNormNoAxes() {
        let a = MLXArray(0 ..< 9) - 4
        let b = a.reshaped(3, 3)

        XCTAssertEqual(MLXLinalg.norm(a).item(Float.self), 7.74597, accuracy: 0.001)
        XCTAssertEqual(MLXLinalg.norm(b).item(Float.self), 7.74597, accuracy: 0.001)

        XCTAssertEqual(MLXLinalg.norm(b, ord: .fro).item(Float.self), 7.74597, accuracy: 0.001)

        // Double.infinity
        XCTAssertEqual(MLXLinalg.norm(a, ord: .infinity).item(Float.self), 4, accuracy: 0.001)
        XCTAssertEqual(MLXLinalg.norm(b, ord: .infinity).item(Float.self), 9, accuracy: 0.001)

        XCTAssertEqual(MLXLinalg.norm(a, ord: -.infinity).item(Float.self), 0, accuracy: 0.001)
        XCTAssertEqual(MLXLinalg.norm(b, ord: -.infinity).item(Float.self), 2, accuracy: 0.001)

        XCTAssertEqual(MLXLinalg.norm(a, ord: 1).item(Float.self), 20, accuracy: 0.001)
        XCTAssertEqual(MLXLinalg.norm(b, ord: 1).item(Float.self), 7, accuracy: 0.001)

        XCTAssertEqual(MLXLinalg.norm(a, ord: -1).item(Float.self), 0, accuracy: 0.001)
        XCTAssertEqual(MLXLinalg.norm(b, ord: -1).item(Float.self), 6, accuracy: 0.001)
    }

    func testNormAxis() {
        let c = MLXArray([1, 2, 3, -1, 1, 4], [2, 3])

        assertEqual(MLXLinalg.norm(c, axis: 0), MLXArray(converting: [1.41421, 2.23607, 5]))
        assertEqual(MLXLinalg.norm(c, ord: 1, axis: 1), MLXArray(converting: [6, 6]))
    }

    func testNormAxes() {
        let m = MLXArray(0 ..< 8, [2, 2, 2])

        assertEqual(MLXLinalg.norm(m, axes: [1, 2]), MLXArray(converting: [3.74166, 11.225]))
    }

    func testQR() {
        let a = MLXArray(converting: [2, 3, 1, 2], [2, 2])
        let (q, r) = MLXLinalg.qr(a, stream: .cpu)

        assertEqual(q, MLXArray(converting: [-0.894427, -0.447214, -0.447214, 0.894427], [2, 2]))
        assertEqual(r, MLXArray(converting: [-2.23607, -3.57771, 0, 0.447214], [2, 2]))
    }

    func testSVDOverload() {
        let a = MLXRandom.uniform(0 ..< 1, [10, 10])

        Stream.withNewDefaultStream(device: .cpu) {
            let (_, s, _) = MLXLinalg.svd(a)
            let s2 = MLXLinalg.svd(a)

            XCTAssertEqual(s.shape, s2.shape)
        }
    }
}
