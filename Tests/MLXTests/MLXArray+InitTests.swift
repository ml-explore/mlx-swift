// Copyright © 2024 Apple Inc.

import Foundation
import Numerics
import XCTest

@testable import MLX

#if canImport(IOSurface)
    import IOSurface
#endif

class MLXArrayInitTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    // MARK: - Dtype
    func testDtypeSize() {
        // Checking that the size of the dtype matches the array's itemsize
        for dtype in DType.allCases {
            XCTAssertEqual(MLXArray(Data(), dtype: dtype).itemSize, dtype.size)
        }
    }

    func testDtypeCodable() {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        // Test encoding / decoding round trip
        for dtype in DType.allCases {
            do {
                let json: Data = try encoder.encode(dtype)
                let decoded = try decoder.decode(DType.self, from: json)
                XCTAssertEqual(decoded, dtype)
            } catch {
                XCTFail("Encoding / decoding failed")
            }
        }
    }

    // MARK: - Creation
    func testInt() {
        // array creation with Int -- we want it to produce .int32
        let a1 = MLXArray(500)
        XCTAssertEqual(a1.dtype, .int32)

        // eplicit int64
        let a2 = MLXArray(int64: 500)
        XCTAssertEqual(a2.dtype, .int64)

        let a3 = MLXArray([1, 2, 3])
        XCTAssertEqual(a3.dtype, .int32)

        let a4 = MLXArray(int64: [1, 2, 3])
        XCTAssertEqual(a4.dtype, .int64)

        let a5 = MLXArray(0 ..< 12)
        XCTAssertEqual(a5.dtype, .int32)

        let a6 = MLXArray(int64: 0 ..< 12)
        XCTAssertEqual(a6.dtype, .int64)
    }

    func testArrayCreationLiteralArray() {
        let a: MLXArray = [20, 30, 40]
        assertEqual(a, MLXArray([20, 30, 40].asInt32))
    }

    func testArrayCreationDoubleArray() {
        // this transforms the array to [Float] and constructs (as a convenience)
        let a = MLXArray(converting: [0.1, 0.5])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a[0].item(Float.self), 0.1, accuracy: 0.01)
    }

    func testArrayCreationArray1D() {
        let a = MLXArray([1, 2, 3])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.count, 3)
        XCTAssertEqual(a.ndim, 1)
        XCTAssertEqual(a.dim(0), 3)
    }

    func testArrayCreationRange() {
        let a = MLXArray(0 ..< 12, [3, 4])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.size, 12)
        XCTAssertEqual(a.ndim, 2)
    }

    func testArrayCreationClosedRange() {
        let a = MLXArray(Int16(3) ... Int16(6))
        XCTAssertEqual(a.dtype, .int16)
        XCTAssertEqual(a.count, 4)
        XCTAssertEqual(a.ndim, 1)
    }

    func testArrayCreationStride() {
        let a = MLXArray(stride(from: Float(0.5), to: Float(1.5), by: Float(0.1)))
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.count, 10)
        XCTAssertEqual(a.ndim, 1)
    }

    func testArrayCreationZeros() {
        let a = MLXArray.zeros([2, 4], type: Int.self)
        XCTAssertEqual(a.dtype, .int64)
        XCTAssertEqual(a.size, 8)
        XCTAssertEqual(a.ndim, 2)
    }

    // MARK: - Arange

    func testArangeIntStop() {
        // arange(10) -> [0, 1, 2, ..., 9]
        let a = arange(10)
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.shape, [10])
        assertEqual(a, MLXArray(0 ..< 10))
    }

    func testArangeIntStartStop() {
        // arange(2, 10) -> [2, 3, 4, ..., 9]
        let a = arange(2, 10)
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.shape, [8])
        assertEqual(a, MLXArray(2 ..< 10))
    }

    func testArangeIntStep() {
        // arange(2, 10, step: 2) -> [2, 4, 6, 8]
        let a = arange(2, 10, step: 2)
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.shape, [4])
        assertEqual(a, MLXArray([2, 4, 6, 8].asInt32))
    }

    func testArangeIntDtype() {
        // arange(10, dtype: .float32) -> [0.0, 1.0, ..., 9.0]
        let a = arange(10, dtype: .float32)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [10])
        assertEqual(a, MLXArray((0 ..< 10).map { Float($0) }))
    }

    func testArangeIntStepDtype() {
        // arange(2, 10, step: 2, dtype: .float32) -> [2.0, 4.0, 6.0, 8.0]
        let a = arange(2, 10, step: 2, dtype: .float32)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [4])
        assertEqual(a, MLXArray([2.0, 4.0, 6.0, 8.0] as [Float]))
    }

    func testArangeDoubleStop() {
        // arange(5.0) -> [0.0, 1.0, 2.0, 3.0, 4.0]
        let a = arange(5.0)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [5])
        assertEqual(a, MLXArray([0.0, 1.0, 2.0, 3.0, 4.0] as [Float]))
    }

    func testArangeDoubleStartStop() {
        // arange(1.0, 5.0) -> [1.0, 2.0, 3.0, 4.0]
        let a = arange(1.0, 5.0)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [4])
        assertEqual(a, MLXArray([1.0, 2.0, 3.0, 4.0] as [Float]))
    }

    func testArangeDoubleStep() {
        // arange(0.0, 2.0, step: 0.5) -> [0.0, 0.5, 1.0, 1.5]
        let a = arange(0.0, 2.0, step: 0.5)
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [4])
        assertEqual(a, MLXArray([0.0, 0.5, 1.0, 1.5] as [Float]))
    }

    func testArangeStaticMethod() {
        // Test static method versions
        let a = MLXArray.arange(10)
        XCTAssertEqual(a.dtype, .int32)
        assertEqual(a, MLXArray(0 ..< 10))

        let b = MLXArray.arange(0.0, 3.0, step: 0.5)
        XCTAssertEqual(b.dtype, .float32)
        assertEqual(b, MLXArray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5] as [Float]))
    }

    func testArangeEmpty() {
        // arange(0) -> empty array
        let a = arange(0)
        XCTAssertEqual(a.shape, [0])

        // arange(5, 5) -> empty array
        let b = arange(5, 5)
        XCTAssertEqual(b.shape, [0])

        // arange(10, 5) -> empty array (start > stop with positive step)
        let c = arange(10, 5)
        XCTAssertEqual(c.shape, [0])
    }

    func testData() {
        let data = Data([1, 2, 3, 4])
        let a = MLXArray(data, [2, 2], type: UInt8.self)
        let b = MLXArray(data, [2, 2], dtype: DType.uint8)
        let expected = MLXArray(UInt8(1) ... 4, [2, 2])
        assertEqual(a, expected)
        assertEqual(b, expected)
    }

    func testUnsafeRawPointer() {
        let data = Data([1, 2, 3, 4])
        let a = data.withUnsafeBytes { ptr in
            MLXArray(ptr, [2, 2], type: UInt8.self)
        }
        let expected = MLXArray(UInt8(1) ... 4, [2, 2])
        assertEqual(a, expected)
    }

    func testUnsafeBufferPointer() {
        let values: [UInt16] = [1, 2, 3, 4]
        let a = values.withUnsafeBufferPointer { ptr in
            MLXArray(ptr, [2, 2])
        }
        let expected = MLXArray(UInt16(1) ... 4, [2, 2])
        assertEqual(a, expected)
    }

    func testComplexScalar() {
        let c1 = MLXArray(real: 3, imaginary: 4)
        XCTAssertEqual(c1.realPart().item(), 3)
        XCTAssertEqual(c1.imaginaryPart().item(), 4)

        let c2 = MLXArray(Complex(3, 4))
        assertEqual(c1, c2)
    }

    func testComplexArray() {
        let r1 = MLXArray(converting: [2, 3, 4])
        let i1 = MLXArray(converting: [7, 8, 9])
        let c1 = r1 + i1.asImaginary()

        assertEqual(c1.realPart(), r1)
        assertEqual(c1.imaginaryPart(), i1)

        let a1: [Complex<Float>] = [Complex(2, 7), Complex(3, 8), Complex(4, 9)]
        let c2 = MLXArray(a1)

        assertEqual(c1, c2)
    }

    func testFloat64Array() {
        let d: [Double] = [1.0, 2.0, 3.0]
        let a = MLXArray(d)
        XCTAssertEqual(a.dtype, .float64)

        let b = MLXArray(0.5)
        XCTAssertEqual(b.dtype, .float32)

        let c = MLXArray(1.1e40)

        XCTAssertEqual(c.dtype, .float64)

        let e = MLXArray(float64: 0.5)
        XCTAssertEqual(e.dtype, .float64)
    }

    // TODO: disabled until next release
    //    #if canImport(IOSurface)
    //        func testIOSurface() {
    //            let height = 100
    //            let width = 128
    //            let pixelFormat = kCVPixelFormatType_32BGRA
    //
    //            let properties: [IOSurfacePropertyKey: any Sendable] = [
    //                .width: width,
    //                .height: height,
    //                .pixelFormat: pixelFormat,
    //                .bytesPerElement: 4,
    //            ]
    //
    //            guard let ioSurface = IOSurface(properties: properties) else {
    //                XCTFail("unable to allocate IOSurface")
    //                return
    //            }
    //
    //            let array = MLXArray(
    //                rawPointer: ioSurface.baseAddress, [height, width, 4], dtype: .uint8
    //            ) {
    //                [ioSurface] in
    //                // this holds reference to the ioSurface and implicitly releases it when it returns
    //                _ = ioSurface
    //                print("release IOSurface")
    //            }
    //            print(mean(array))
    //        }
    //    #endif
}
