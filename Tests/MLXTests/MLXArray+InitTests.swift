// Copyright © 2024 Apple Inc.

import Foundation
import Numerics
import Testing

@testable import MLX

#if canImport(IOSurface)
    import IOSurface
    import CoreVideo
#endif

@Suite
struct MLXArrayInitTests {

    // MARK: - Dtype
    @Test func dtypeSize() {
        // Checking that the size of the dtype matches the array's itemsize
        for dtype in DType.allCases {
            #expect(MLXArray(Data(), dtype: dtype).itemSize == dtype.size)
        }
    }

    @Test func dtypeCodable() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        // Test encoding / decoding round trip
        for dtype in DType.allCases {
            let json: Data = try encoder.encode(dtype)
            let decoded = try decoder.decode(DType.self, from: json)
            #expect(decoded == dtype)
        }
    }

    // MARK: - Creation
    @Test func int() {
        // array creation with Int -- we want it to produce .int32
        let a1 = MLXArray(500)
        #expect(a1.dtype == .int32)

        // eplicit int64
        let a2 = MLXArray(int64: 500)
        #expect(a2.dtype == .int64)

        let a3 = MLXArray([1, 2, 3])
        #expect(a3.dtype == .int32)

        let a4 = MLXArray(int64: [1, 2, 3])
        #expect(a4.dtype == .int64)

        let a5 = MLXArray(0 ..< 12)
        #expect(a5.dtype == .int32)

        let a6 = MLXArray(int64: 0 ..< 12)
        #expect(a6.dtype == .int64)
    }

    @Test func arrayCreationLiteralArray() {
        let a: MLXArray = [20, 30, 40]
        expectEqual(a, MLXArray([20, 30, 40].asInt32))
    }

    @Test func arrayCreationDoubleArray() {
        // this transforms the array to [Float] and constructs (as a convenience)
        let a = MLXArray(converting: [0.1, 0.5])
        #expect(a.dtype == .float32)
        expectEqual(a[0].item(Float.self), 0.1, accuracy: 0.01)
    }

    @Test func arrayCreationArray1D() {
        let a = MLXArray([1, 2, 3])
        #expect(a.dtype == .int32)
        #expect(a.count == 3)
        #expect(a.ndim == 1)
        #expect(a.dim(0) == 3)
    }

    @Test func arrayCreationRange() {
        let a = MLXArray(0 ..< 12, [3, 4])
        #expect(a.dtype == .int32)
        #expect(a.size == 12)
        #expect(a.ndim == 2)
    }

    // MARK: - Nested (multi dimensional) arrays

    @Test func arrayCreationNested2D() {
        let a = MLXArray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        #expect(a.dtype == .int32)
        #expect(a.shape == [2, 3])
        #expect(a.ndim == 2)
        #expect(a.size == 6)

        // count is the size of dimension 0
        #expect(a.count == 2)

        expectEqual(a, MLXArray(1 ... 6, [2, 3]))
    }

    @Test func arrayCreationNested3D() {
        // literal nesting and the flat + shape form must agree
        let a = MLXArray([
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ])
        #expect(a.dtype == .int32)
        #expect(a.shape == [2, 3, 2])
        expectEqual(a, MLXArray(0 ..< 12, [2, 3, 2]))

        // also works with a computed (non literal) nested array
        let nested = (0 ..< 2).map { i in
            (0 ..< 3).map { j in
                (0 ..< 4).map { k in i * 12 + j * 4 + k }
            }
        }
        let b = MLXArray(nested)
        #expect(b.shape == [2, 3, 4])
        expectEqual(b, MLXArray(0 ..< 24, [2, 3, 4]))
    }

    @Test func arrayCreationNested4D() {
        let a = MLXArray([
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
            [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
        ])
        #expect(a.shape == [2, 2, 2, 2])
        #expect(a.ndim == 4)
        expectEqual(a, MLXArray(0 ..< 16, [2, 2, 2, 2]))
    }

    @Test func arrayCreationNestedDTypes() {
        // the dtype of a nested array matches the 1d initializers for the same leaf type
        #expect(MLXArray([[1, 2], [3, 4]]).dtype == .int32)
        #expect(MLXArray([[Int32(1), 2], [3, 4]]).dtype == .int32)
        #expect(MLXArray([[Int16(1), 2], [3, 4]]).dtype == .int16)
        #expect(MLXArray([[UInt8(1), 2], [3, 4]]).dtype == .uint8)
        #expect(MLXArray([[Float(1), 2], [3, 4]]).dtype == .float32)
        #expect(MLXArray([[true, false], [false, true]]).dtype == .bool)

        // Int64 is a distinct type from Int and must not take the Int -> .int32 path
        #expect(MLXArray([[Int64(1), 2], [3, 4]]).dtype == .int64)

        // casting the literal as a whole is a different inference path than casting the
        // elements and must stay unambiguous
        #expect(MLXArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] as [[Float32]]).dtype == .float32)
        #expect(MLXArray([[[1.0, 2.0]]] as [[[Float32]]]).dtype == .float32)

        // as with `MLXArray([Double])` this produces .float64 rather than promoting
        let doubles: [[Double]] = [[1, 2], [3, 4]]
        #expect(MLXArray(doubles).dtype == .float64)

        let complex: [[Complex<Float>]] = [[Complex(2, 7)], [Complex(3, 8)]]
        let c = MLXArray(complex)
        #expect(c.dtype == .complex64)
        #expect(c.shape == [2, 1])
        expectEqual(c, MLXArray([Complex<Float>(2, 7), Complex<Float>(3, 8)], [2, 1]))
    }

    @Test func arrayCreationNestedInt64() {
        // a value that does not fit in an Int32 -- the whole point of the int64: variant
        let big = Int(Int32.max) + 10
        let a = MLXArray(int64: [[1, 2, 3], [4, 5, big]])
        #expect(a.dtype == .int64)
        #expect(a.shape == [2, 3])
        expectEqual(a, MLXArray(int64: [1, 2, 3, 4, 5, big], [2, 3]))
    }

    @Test func arrayCreationNestedConverting() {
        // non square so a transposed flattening could not pass
        let a = MLXArray(converting: [[0.1, 0.5, 0.9], [1.3, 1.7, 2.1]])
        #expect(a.dtype == .float32)
        #expect(a.shape == [2, 3])
        expectEqual(a, MLXArray(converting: [0.1, 0.5, 0.9, 1.3, 1.7, 2.1], [2, 3]))
    }

    @Test func arrayCreationNestedEmpty() {
        let a = MLXArray([[Int](), [Int]()])
        #expect(a.shape == [2, 0])
        #expect(a.size == 0)

        let b = MLXArray([[Int]]())
        #expect(b.shape == [0, 0])
        #expect(b.size == 0)
    }

    @Test func arrayCreationClosedRange() {
        let a = MLXArray(Int16(3) ... Int16(6))
        #expect(a.dtype == .int16)
        #expect(a.count == 4)
        #expect(a.ndim == 1)
    }

    @Test func arrayCreationStride() {
        let a = MLXArray(stride(from: Float(0.5), to: Float(1.5), by: Float(0.1)))
        #expect(a.dtype == .float32)
        #expect(a.count == 10)
        #expect(a.ndim == 1)
    }

    @Test func arrayCreationZeros() {
        let a = MLXArray.zeros([2, 4], type: Int.self)
        #expect(a.dtype == .int64)
        #expect(a.size == 8)
        #expect(a.ndim == 2)
    }

    // MARK: - Arange

    @Test func arangeIntStop() {
        // arange(10) -> [0, 1, 2, ..., 9]
        let a = arange(10)
        #expect(a.dtype == .int32)
        #expect(a.shape == [10])
        expectEqual(a, MLXArray(0 ..< 10))
    }

    @Test func arangeIntStartStop() {
        // arange(2, 10) -> [2, 3, 4, ..., 9]
        let a = arange(2, 10)
        #expect(a.dtype == .int32)
        #expect(a.shape == [8])
        expectEqual(a, MLXArray(2 ..< 10))
    }

    @Test func arangeIntStep() {
        // arange(2, 10, step: 2) -> [2, 4, 6, 8]
        let a = arange(2, 10, step: 2)
        #expect(a.dtype == .int32)
        #expect(a.shape == [4])
        expectEqual(a, MLXArray([2, 4, 6, 8].asInt32))
    }

    @Test func arangeIntDtype() {
        // arange(10, dtype: .float32) -> [0.0, 1.0, ..., 9.0]
        let a = arange(10, dtype: .float32)
        #expect(a.dtype == .float32)
        #expect(a.shape == [10])
        expectEqual(a, MLXArray((0 ..< 10).map { Float($0) }))
    }

    @Test func arangeIntStepDtype() {
        // arange(2, 10, step: 2, dtype: .float32) -> [2.0, 4.0, 6.0, 8.0]
        let a = arange(2, 10, step: 2, dtype: .float32)
        #expect(a.dtype == .float32)
        #expect(a.shape == [4])
        expectEqual(a, MLXArray([2.0, 4.0, 6.0, 8.0] as [Float]))
    }

    @Test func arangeDoubleStop() {
        // arange(5.0) -> [0.0, 1.0, 2.0, 3.0, 4.0]
        let a = arange(5.0)
        #expect(a.dtype == .float32)
        #expect(a.shape == [5])
        expectEqual(a, MLXArray([0.0, 1.0, 2.0, 3.0, 4.0] as [Float]))
    }

    @Test func arangeDoubleStartStop() {
        // arange(1.0, 5.0) -> [1.0, 2.0, 3.0, 4.0]
        let a = arange(1.0, 5.0)
        #expect(a.dtype == .float32)
        #expect(a.shape == [4])
        expectEqual(a, MLXArray([1.0, 2.0, 3.0, 4.0] as [Float]))
    }

    @Test func arangeDoubleStep() {
        // arange(0.0, 2.0, step: 0.5) -> [0.0, 0.5, 1.0, 1.5]
        let a = arange(0.0, 2.0, step: 0.5)
        #expect(a.dtype == .float32)
        #expect(a.shape == [4])
        expectEqual(a, MLXArray([0.0, 0.5, 1.0, 1.5] as [Float]))
    }

    @Test func arangeStaticMethod() {
        // Test static method versions
        let a = MLXArray.arange(10)
        #expect(a.dtype == .int32)
        expectEqual(a, MLXArray(0 ..< 10))

        let b = MLXArray.arange(0.0, 3.0, step: 0.5)
        #expect(b.dtype == .float32)
        expectEqual(b, MLXArray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5] as [Float]))
    }

    @Test func arangeEmpty() {
        // arange(0) -> empty array
        let a = arange(0)
        #expect(a.shape == [0])

        // arange(5, 5) -> empty array
        let b = arange(5, 5)
        #expect(b.shape == [0])

        // arange(10, 5) -> empty array (start > stop with positive step)
        let c = arange(10, 5)
        #expect(c.shape == [0])
    }

    @Test func data() {
        let data = Data([1, 2, 3, 4])
        let a = MLXArray(data, [2, 2], type: UInt8.self)
        let b = MLXArray(data, [2, 2], dtype: DType.uint8)
        let expected = MLXArray(UInt8(1) ... 4, [2, 2])
        expectEqual(a, expected)
        expectEqual(b, expected)
    }

    @Test func unsafeRawPointer() {
        let data = Data([1, 2, 3, 4])
        let a = data.withUnsafeBytes { ptr in
            MLXArray(ptr, [2, 2], type: UInt8.self)
        }
        let expected = MLXArray(UInt8(1) ... 4, [2, 2])
        expectEqual(a, expected)
    }

    @Test func unsafeBufferPointer() {
        let values: [UInt16] = [1, 2, 3, 4]
        let a = values.withUnsafeBufferPointer { ptr in
            MLXArray(ptr, [2, 2])
        }
        let expected = MLXArray(UInt16(1) ... 4, [2, 2])
        expectEqual(a, expected)
    }

    @Test func complexScalar() {
        let c1 = MLXArray(real: 3, imaginary: 4)
        #expect(c1.realPart().item(Float.self) == 3)
        #expect(c1.imaginaryPart().item(Float.self) == 4)

        let c2 = MLXArray(Complex(3, 4))
        expectEqual(c1, c2)
    }

    @Test func complexArray() {
        let r1 = MLXArray(converting: [2, 3, 4])
        let i1 = MLXArray(converting: [7, 8, 9])
        let c1 = r1 + i1.asImaginary()

        expectEqual(c1.realPart(), r1)
        expectEqual(c1.imaginaryPart(), i1)

        let a1: [Complex<Float>] = [Complex(2, 7), Complex(3, 8), Complex(4, 9)]
        let c2 = MLXArray(a1)

        expectEqual(c1, c2)
    }

    @Test func float64Array() {
        let d: [Double] = [1.0, 2.0, 3.0]
        let a = MLXArray(d)
        #expect(a.dtype == .float64)

        let b = MLXArray(0.5)
        #expect(b.dtype == .float32)

        let c = MLXArray(1.1e40)

        #expect(c.dtype == .float64)

        let e = MLXArray(float64: 0.5)
        #expect(e.dtype == .float64)
    }

    /// The finalizer's captures must be released once mlx has called it.
    ///
    /// `init(rawPointer:_:dtype:finalizer:)` retains a box holding the closure and
    /// hands it to mlx as an opaque payload; `finalizerTrampoline` is the only
    /// thing that can release it. If the trampoline reads the box without
    /// consuming the reference, the closure — and everything it captured — is
    /// pinned for the lifetime of the process, once per adopted buffer.
    ///
    /// `ioSurface()` below documents the intended behaviour in a comment
    /// ("implicitly releases it when it returns") but never asserts it, which is
    /// what let this go unnoticed.
    @Test func finalizerCapturesAreReleasedAfterFinalizerRuns() {
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
            #expect(array.sum().item(Float.self) == 4)
        }

        // mlx keeps freed buffers in its allocator cache, so the dtor has not
        // necessarily run at scope exit.
        Memory.clearCache()
        let deadline = Date().addingTimeInterval(5)
        while finalizerRuns.current == 0, Date() < deadline { usleep(1_000) }

        #expect(finalizerRuns.current == 1, "mlx must call the finalizer exactly once")
        #expect(
            witnessReleases.current == 1,
            "the finalizer's captures must be released along with it — otherwise every adopted buffer permanently pins whatever the closure held"
        )
    }

    // MARK: - linspace

    @Test func linspaceIntegerBoundsAreFloat() {
        // matches python: mx.linspace(0, 1, 3) -> float32 [0, 0.5, 1].
        // an integer result would truncate to [0, 0, 1]
        let a = MLXArray.linspace(0, 1, count: 3)
        #expect(a.dtype == .float32)
        expectEqual(a, MLXArray(converting: [0.0, 0.5, 1.0]), atol: 1e-6)
    }

    @Test func linspaceIntegerDType() {
        // an integer (truncating) result is available, but has to be asked for
        let a = MLXArray.linspace(0, 10, count: 6, dtype: .int32)
        #expect(a.dtype == .int32)
        #expect(a.asArray(Int32.self) == [0, 2, 4, 6, 8, 10])
    }

    @Test func linspaceDoubleBoundsAreFloat32() {
        // Double.dtype is float64, but like MLXArray(1.0) we do not promote to
        // float64 -- it is not available on the GPU
        let a = MLXArray.linspace(0.0, 1.0, count: 5)
        #expect(a.dtype == .float32)
        expectEqual(a, MLXArray(converting: [0.0, 0.25, 0.5, 0.75, 1.0]), atol: 1e-6)
    }

    @Test func linspaceFloatBoundsKeepTheirDType() {
        #expect(MLXArray.linspace(Float(0), Float(1), count: 5).dtype == .float32)
        #if !arch(x86_64)
            #expect(MLXArray.linspace(Float16(0), Float16(1), count: 5).dtype == .float16)
        #endif
    }

    @Test func linspaceEndpoint() {
        let inclusive = MLX.linspace(0, 1, count: 5)
        expectEqual(inclusive, MLXArray(converting: [0, 0.25, 0.5, 0.75, 1.0]), atol: 1e-6)

        let halfOpen = MLX.linspace(0, 1, count: 5, endpoint: false)
        expectEqual(halfOpen, MLXArray(converting: [0, 0.2, 0.4, 0.6, 0.8]), atol: 1e-6)
    }

    #if canImport(IOSurface)
        @Test func ioSurface() throws {
            let height = 100
            let width = 128
            let pixelFormat = kCVPixelFormatType_32BGRA

            let properties: [IOSurfacePropertyKey: any Sendable] = [
                .width: width,
                .height: height,
                .pixelFormat: pixelFormat,
                .bytesPerElement: 4,
            ]

            let ioSurface = try #require(
                IOSurface(properties: properties), "unable to allocate IOSurface")

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
