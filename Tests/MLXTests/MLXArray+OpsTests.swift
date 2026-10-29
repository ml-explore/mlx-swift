// Copyright © 2024 Apple Inc.

import Foundation
import Numerics
import Testing

@testable import MLX

@Suite
struct MLXArrayOpsTests {

    // MARK: - Operators

    @Test func arithmeticSimple() {
        var a = MLXArray([1, 2, 3])
        let b = MLXArray(converting: [-5.0, 37.5, 4])

        // example of an expression -- the - 1 is using the 1 as ExpressibleByIntegerLiteral
        let r = a + b - 1

        // make sure everything got hooked up
        #expect(r.shape == [3])
        #expect(r.dtype == .float32)
        expectEqual(r, MLXArray(converting: [-5, 38.5, 6]))

        a += b
        #expect(a.shape == [3])
        #expect(a.dtype == .float32)
        expectEqual(a, MLXArray(converting: [-4, 39.5, 7]))

        a += 1
        #expect(a.shape == [3])
        #expect(a.dtype == .float32)
        expectEqual(a, MLXArray(converting: [-3, 40.5, 8]))
    }

    @Test func arithmeticMatrix() {
        let a = MLXArray([1, 2, 3, 4], [2, 2])
        let b = MLXArray(converting: [-5.0, 37.5, 4, 7, 1, 0], [2, 3])

        // (a * 5)^2 @ b
        let r = ((a * 5) ** 2).matmul(b)

        // make sure everything got hooked up
        #expect(r.shape == [2, 3])
        #expect(r.dtype == .float32)
        expectEqual(r, MLXArray(converting: [575, 1037.5, 100, 1675, 8837.5, 900], [2, 3]))
    }

    @Test func arithmeticDivide() {
        let a = MLXArray([1, 2, 3])

        // in python: (a // 2) / 0.5
        let r = a.floorDivide(2) / Float(0.5)

        // make sure everything got hooked up
        #expect(r.shape == [3])
        #expect(r.dtype == .float32)
        expectEqual(r, MLXArray(converting: [0, 2, 2]))
    }

    @Test func opsLogical() {
        let a = MLXArray([1, 2, 3])

        let r = (a .< (a + 1))

        // make sure everything got hooked up
        #expect(r.shape == [3])
        #expect(r.dtype == .bool)
        #expect(r.all().item(Bool.self) == true)

        // generic item() inferred from context
        let allTrue: Bool = r.all().item()
        #expect(allTrue)
    }

    @Test func opsLogicalBoolContext() {
        let a = MLXArray([1, 2, 3])

        if (a .< (a + 1)).all().item() {
            // expected
        } else {
            Issue.record("should be true")
        }
    }

    @Test func functions() {
        let a = MLXArray(0 ..< 12, [4, 3])

        let r = a.square().sqrt().transposed().T
        expectEqual(a, r)
    }

    @Test func splitEqual() {
        let a = MLXArray(0 ..< 12, [4, 3])

        let s = a.split(parts: 2)
        #expect(s[0].shape == [2, 3])
        #expect(s[1].shape == [2, 3])
    }

    @Test func splitIndices() {
        let a = MLXArray(0 ..< 12, [4, 3])

        let s = a.split(indices: [0, 3])
        #expect(s[0].shape == [0, 3])
        #expect(s[1].shape == [3, 3])
        #expect(s[2].shape == [1, 3])
    }

    @Test func flatten() {
        let a = MLXArray(0 ..< (8 * 4 * 3), [8, 4, 3])

        let f1 = a.flattened()
        #expect(f1.shape == [8 * 4 * 3])

        let f2 = a.flattened(start: 1)
        #expect(f2.shape == [8, 4 * 3])
    }

    @Test func moveAxis() {
        let array = MLXArray(0 ..< 16, [2, 2, 2, 2])

        let r = array.movedAxis(source: 0, destination: 3)

        let expected = MLXArray(
            [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15], [2, 2, 2, 2])
        expectEqual(r, expected)
    }

    @Test func swapAxes() {
        let array = MLXArray(0 ..< 16, [2, 2, 2, 2])

        let r = array.swappedAxes(0, 3)

        let expected = MLXArray(
            [0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15], [2, 2, 2, 2])
        expectEqual(r, expected)
    }

    @Test func promotionTests() {
        // expect that adding a scalar will not promote to a larger type
        let x = MLXArray.ones([10], type: Float16.self)

        #expect((x * 15.5).dtype == .float16)
        #expect((15.5 * x).dtype == .float16)
    }

    @Test func bfloat16() {
        let x = MLXArray([1, 5, 9]).asType(.bfloat16)
        let y = x * x
        #expect(y.dtype == .bfloat16)

        // this will convert the 3 to bfloat16 via init<T: HasDType>(_ value: T, dtype: DType)
        let z = (y + 3).sum()
        #expect(z.dtype == .bfloat16)

        // read out the value as a usable type
        let r1 = z.item(Float32.self)
        #expect(r1 == 116)

        let r2 = z.asArray(Float.self)
        #expect(r2 == [116])
    }

    @Test func complexPromote() {
        let x = MLXArray(Complex(3, 4))
        let y = x * x
        #expect(y.dtype == .complex64)

        // this will convert the 3 to (3, 0i) via init<T: HasDType>(_ value: T, dtype: DType)
        let z = y + 3
        #expect(z.dtype == .complex64)

        // read out as complex
        let r1 = z.item(Complex.self)
        #expect(r1 == Complex(-4, 24))

        // read out the value as a usable type
        let r2 = z.asArray(Complex.self)
        #expect(r2 == [Complex(-4, 24)])
    }

    @Test func countNonzero() {
        let a = MLXArray([0, 1, 2, 0, 3, 0], [2, 3])
        let resAxes0 = a.countNonzero(axes: [0])
        let resAxes1 = a.countNonzero(axes: [1])
        let resAxis0 = a.countNonzero(axis: 0)
        let resAxis1 = a.countNonzero(axis: 1)
        let resAll = a.countNonzero()

        expectEqual(resAxes0, [0, 2, 1])
        expectEqual(resAxes0, resAxis0)
        expectEqual(resAxes1, [2, 1])
        expectEqual(resAxes1, resAxis1)
        expectEqual(resAll, 3.asMLXArray(dtype: .int32))
    }

}
