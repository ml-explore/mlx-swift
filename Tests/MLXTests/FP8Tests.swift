// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

class FP8Tests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testFromFP8DecodesE4M3Bytes() {
        let allBytes = (0 ... 255).map { UInt8($0) }
        let bytes = MLXArray(allBytes)
        let decoded = fromFP8(bytes, dtype: .float32)

        XCTAssertEqual(decoded.dtype, .float32)
        XCTAssertEqual(decoded.shape, [256])

        let expected = MLXArray(allBytes.map { e4m3Value($0) })
        assertEqual(decoded, expected)
    }

    func testToFP8EncodesE4M3Bytes() {
        let values = MLXArray(
            [
                0.0, -0.0, 1.0, -2.0, 0.125, 0.015625, 448.0, -448.0,
            ] as [Float])

        let encoded = toFP8(values)

        XCTAssertEqual(encoded.dtype, .uint8)
        XCTAssertEqual(encoded.shape, values.shape)
        XCTAssertEqual(
            encoded.asArray(UInt8.self),
            [0x00, 0x80, 0x38, 0xC0, 0x20, 0x08, 0x7E, 0xFE])
    }
}

private func e4m3Value(_ value: UInt8) -> Float {
    let sign: Float = (value & 0x80) == 0 ? 1 : -1
    let exponent = Int((value >> 3) & 0x0F)
    let mantissa = Int(value & 0x07)

    if exponent == 0 {
        if mantissa == 0 {
            return sign < 0 ? -0.0 : 0.0
        }
        return sign * Float(mantissa) / 512
    }

    return sign * Float(8 + mantissa) * powf(2, Float(exponent - 10))
}
