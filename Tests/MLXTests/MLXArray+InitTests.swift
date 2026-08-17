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

    // MARK: - Nested (multi dimensional) arrays

    func testArrayCreationNested2D() {
        let a = MLXArray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.shape, [2, 3])
        XCTAssertEqual(a.ndim, 2)
        XCTAssertEqual(a.size, 6)

        // count is the size of dimension 0
        XCTAssertEqual(a.count, 2)

        assertEqual(a, MLXArray(1 ... 6, [2, 3]))
    }

    func testArrayCreationNested3D() {
        // literal nesting and the flat + shape form must agree
        let a = MLXArray([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ])
        XCTAssertEqual(a.dtype, .int32)
        XCTAssertEqual(a.shape, [2, 3, 2])
        assertEqual(a, MLXArray(0 ..< 12, [2, 3, 2]))

        // also works with a computed (non literal) nested array
        let nested = (0 ..< 2).map { i in
            (0 ..< 3).map { j in
                (0 ..< 4).map { k in i * 12 + j * 4 + k }
            }
        }
        let b = MLXArray(nested)
        XCTAssertEqual(b.shape, [2, 3, 4])
        assertEqual(b, MLXArray(0 ..< 24, [2, 3, 4]))
    }

    func testArrayCreationNested4D() {
        let a = MLXArray([
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
        ])
        XCTAssertEqual(a.shape, [2, 2, 2, 2])
        XCTAssertEqual(a.ndim, 4)
        assertEqual(a, MLXArray(0 ..< 16, [2, 2, 2, 2]))
    }

    func testArrayCreationNestedDTypes() {
        // the dtype of a nested array matches the 1d initializers for the same leaf type
        XCTAssertEqual(MLXArray([[1, 2], [3, 4]]).dtype, .int32)
        XCTAssertEqual(MLXArray([[Int32(1), 2], [3, 4]]).dtype, .int32)
        XCTAssertEqual(MLXArray([[Int16(1), 2], [3, 4]]).dtype, .int16)
        XCTAssertEqual(MLXArray([[UInt8(1), 2], [3, 4]]).dtype, .uint8)
        XCTAssertEqual(MLXArray([[Float(1), 2], [3, 4]]).dtype, .float32)
        XCTAssertEqual(MLXArray([[true, false], [false, true]]).dtype, .bool)

        // Int64 is a distinct type from Int and must not take the Int -> .int32 path
        XCTAssertEqual(MLXArray([[Int64(1), 2], [3, 4]]).dtype, .int64)

        // casting the literal as a whole is a different inference path than casting the
        // elements and must stay unambiguous
        XCTAssertEqual(
            MLXArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] as [[Float32]]).dtype, .float32)
        XCTAssertEqual(MLXArray([[[1.0, 2.0]]] as [[[Float32]]]).dtype, .float32)

        // as with `MLXArray([Double])` this produces .float64 rather than promoting
        let doubles: [[Double]] = [[1, 2], [3, 4]]
        XCTAssertEqual(MLXArray(doubles).dtype, .float64)

        let complex: [[Complex<Float>]] = [[Complex(2, 7)], [Complex(3, 8)]]
        let c = MLXArray(complex)
        XCTAssertEqual(c.dtype, .complex64)
        XCTAssertEqual(c.shape, [2, 1])
        assertEqual(c, MLXArray([Complex<Float>(2, 7), Complex<Float>(3, 8)], [2, 1]))
    }

    func testArrayCreationNestedInt64() {
        // a value that does not fit in an Int32 -- the whole point of the int64: variant
        let big = Int(Int32.max) + 10
        let a = MLXArray(int64: [[1, 2, 3], [4, 5, big]])
        XCTAssertEqual(a.dtype, .int64)
        XCTAssertEqual(a.shape, [2, 3])
        assertEqual(a, MLXArray(int64: [1, 2, 3, 4, 5, big], [2, 3]))
    }

    func testArrayCreationNestedConverting() {
        // non square so a transposed flattening could not pass
        let a = MLXArray(converting: [[0.1, 0.5, 0.9], [1.3, 1.7, 2.1]])
        XCTAssertEqual(a.dtype, .float32)
        XCTAssertEqual(a.shape, [2, 3])
        assertEqual(a, MLXArray(converting: [0.1, 0.5, 0.9, 1.3, 1.7, 2.1], [2, 3]))
    }

    func testArrayCreationNestedEmpty() {
        let a = MLXArray([[Int](), [Int]()])
        XCTAssertEqual(a.shape, [2, 0])
        XCTAssertEqual(a.size, 0)

        let b = MLXArray([[Int]]())
        XCTAssertEqual(b.shape, [0, 0])
        XCTAssertEqual(b.size, 0)
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

    /// The finalizer's captures must be released once mlx has called it.
    ///
    /// `init(rawPointer:_:dtype:finalizer:)` retains a box holding the closure and
    /// hands it to mlx as an opaque payload; `finalizerTrampoline` is the only
    /// thing that can release it. If the trampoline reads the box without
    /// consuming the reference, the closure — and everything it captured — is
    /// pinned for the lifetime of the process, once per adopted buffer.
    ///
    /// `testIOSurface` below documents the intended behaviour in a comment
    /// ("implicitly releases it when it returns") but never asserts it, which is
    /// what let this go unnoticed.
    func testFinalizerCapturesAreReleasedAfterFinalizerRuns() {
        // mlx may invoke the dtor from its own thread, so the counters cannot be
        // plain locals read across that boundary.
        final class Counter: @unchecked Sendable {
            private let lock = NSLock()
            private var value = 0
            func increment() {
                lock.lock()
                value += 1
                lock.unlock()
            }
            var current: Int {
                lock.lock()
                defer { lock.unlock() }
                return value
            }
        }
        final class Witness {
            let onDeinit: () -> Void
            init(_ onDeinit: @escaping () -> Void) { self.onDeinit = onDeinit }
            deinit { onDeinit() }
        }

        let finalizerRuns = Counter()
        let witnessReleases = Counter()

        do {
            let buffer = UnsafeMutableRawPointer.allocate(
                byteCount: 4 * MemoryLayout<Float>.stride, alignment: 16)
            buffer.initializeMemory(as: Float.self, repeating: 1, count: 4)

            let witness = Witness { witnessReleases.increment() }
            let array = MLXArray(rawPointer: buffer, [4], dtype: .float32) {
                [witness] in
                // Holds the witness exactly the way the IOSurface example holds
                // its surface.
                _ = witness
                finalizerRuns.increment()
                buffer.deallocate()
            }
            XCTAssertEqual(array.sum().item(Float.self), 4)
        }

        // mlx keeps freed buffers in its allocator cache, so the dtor has not
        // necessarily run at scope exit.
        Memory.clearCache()
        let deadline = Date().addingTimeInterval(5)
        while finalizerRuns.current == 0, Date() < deadline { usleep(1_000) }

        XCTAssertEqual(finalizerRuns.current, 1, "mlx must call the finalizer exactly once")
        XCTAssertEqual(
            witnessReleases.current, 1,
            "the finalizer's captures must be released along with it — otherwise every adopted "
                + "buffer permanently pins whatever the closure held")
    }

    #if canImport(IOSurface)
        func testIOSurface() {
            let height = 100
            let width = 128
            let pixelFormat = kCVPixelFormatType_32BGRA

            let properties: [IOSurfacePropertyKey: any Sendable] = [
                .width: width,
                .height: height,
                .pixelFormat: pixelFormat,
                .bytesPerElement: 4,
            ]

            guard let ioSurface = IOSurface(properties: properties) else {
                XCTFail("unable to allocate IOSurface")
                return
            }

            let array = MLXArray(
                rawPointer: ioSurface.baseAddress, [height, width, 4], dtype: .uint8
            ) {
                [ioSurface] in
                // this holds reference to the ioSurface and implicitly releases it when it returns
                _ = ioSurface
                print("release IOSurface")
            }
            print(mean(array))
        }
    #endif
}
