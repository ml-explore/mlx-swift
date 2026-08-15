// Copyright © 2024 Apple Inc.

import Cmlx

extension MLXArray {

    /// A move-only, flat view of this array's contents as `Scalar`, converting the ``DType``
    /// if needed.
    ///
    /// This is shorthand for the common ``asArray(_:)`` / ``asType(_:stream:)`` + ``asArray(_:)``
    /// patterns, but lets you pull out single values or ranges (and iterate a zero-copy `Span`)
    /// without eagerly copying the whole array into a `[Scalar]`.
    ///
    /// ```swift
    /// let flat = prediction.view(Float.self)
    /// let x = flat[3]                    // one value
    /// let head = flat[0 ..< 4]           // a range, copied out
    /// flat.withSpan { span in ... }      // zero-copy bulk read
    /// ```
    ///
    /// ### See Also
    /// - ``MLXArrayOf``
    /// - ``asArray(_:)``
    public func view<Scalar: HasDType>(_ type: Scalar.Type = Scalar.self) -> MLXArrayOf<Scalar> {
        MLXArrayOf(self)
    }
}

/// A move-only, flat (1d) view over an ``MLXArray`` as a specific `Scalar` type.
///
/// Created via ``MLXArray/view(_:)``. The array is converted to `Scalar`'s ``DType`` and made
/// contiguous up front; reads then copy out of the backing directly with no further MLX calls.
///
/// ### See Also
/// - ``MLXArray/view(_:)``
public struct MLXArrayOf<Scalar: HasDType>: ~Copyable {

    /// The backing: matches `Scalar`'s ``DType``, contiguous, and evaluated.
    public let values: MLXArray

    public init(_ array: MLXArray) {
        var values = array.asType(Scalar.self)
        values.eval()
        // reads index the backing directly, so it must be contiguous
        if values.contiguousToDimension() != 0 {
            values = MLXArray(values.asArray(Scalar.self))
        }
        self.values = values
    }

    /// Number of elements in the flattened view.
    public var count: Int { values.size }

    /// A zero-copy `Span` over the contents. Prefer this over per-element subscripting in
    /// hot loops.
    public var span: Span<Scalar> {
        @_lifetime(borrow self)
        borrowing get {
            let base = unsafe UnsafeRawPointer(mlx_array_data_uint8(values.ctx)!)
                .assumingMemoryBound(to: Scalar.self)
            let buffer = unsafe UnsafeBufferPointer(start: base, count: count)
            let span = unsafe Span(_unsafeElements: buffer)
            // the backing lives as long as `self` holds `values`, not as long as the local buffer
            return unsafe _overrideLifetime(span, borrowing: self)
        }
    }

    /// Copy out a single value at `index` in the flattened contents.
    public subscript(_ index: Int) -> Scalar {
        precondition(index >= 0 && index < count, "index \(index) out of bounds 0..<\(count)")
        return span[index]
    }

    /// Copy out a contiguous `range` of the flattened contents.
    public subscript(_ range: Range<Int>) -> [Scalar] {
        precondition(
            range.lowerBound >= 0 && range.upperBound <= count,
            "range \(range) out of bounds 0..<\(count)")
        let span = self.span
        return [Scalar](unsafeUninitializedCapacity: range.count) { buffer, initialized in
            for (offset, source) in range.enumerated() {
                buffer[offset] = span[source]
            }
            initialized = range.count
        }
    }

    /// Copy out the entire flattened contents as a `[Scalar]`.
    public func asArray() -> [Scalar] { values.asArray(Scalar.self) }
}
