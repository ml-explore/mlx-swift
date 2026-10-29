// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct FP8Tests {

    @Test func fromFP8DecodesE4M3BytesPositive() {
        // 127 and 255 are NaN and don't compare equal, so skip them
        let allBytes = (0 ... 126).map { UInt8($0) }
        let bytes = MLXArray(allBytes)
        let decoded = fromFP8(bytes, dtype: .float32)

        #expect(decoded.dtype == .float32)
        #expect(decoded.shape == [127])

        let expected = MLXArray(allBytes.map { e4m3Value($0) })
        expectEqual(decoded, expected)
    }

    @Test func fromFP8DecodesE4M3BytesNegative() {
        let allBytes = (128 ... 254).map { UInt8($0) }
        let bytes = MLXArray(allBytes)
        let decoded = fromFP8(bytes, dtype: .float32)

        #expect(decoded.dtype == .float32)
        #expect(decoded.shape == [127])

        let expected = MLXArray(allBytes.map { e4m3Value($0) })
        expectEqual(decoded, expected)
    }

    @Test func toFP8EncodesE4M3Bytes() {
        let values = MLXArray(
            [
                0.0, -0.0, 1.0, -2.0, 0.125, 0.015625, 448.0, -448.0,
            ] as [Float])

        let encoded = toFP8(values)

        #expect(encoded.dtype == .uint8)
        #expect(encoded.shape == values.shape)
        #expect(
            encoded.asArray(UInt8.self) == [0x00, 0x80, 0x38, 0xC0, 0x20, 0x08, 0x7E, 0xFE])
    }
}

private func e4m3Value(_ value: UInt8) -> Float {
    if (value & 0x7F) == 0x7F { return .nan }
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
