// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Initialization from nested Swift Arrays

extension MLXArray {

    // MARK: 2-D

    /// Create a 2-D ``MLXArray`` from a row-major ``[[T]]`` literal.
    ///
    /// Each inner array becomes a row of the resulting array. All rows must
    /// have the same length.
    ///
    /// Example:
    ///
    /// ```swift
    /// let a = MLXArray([[1.0, 2.0],
    ///                   [3.0, 4.0]])
    /// // a.shape == [2, 2], a.dtype == .float32
    /// ```
    ///
    /// Closes [#161](https://github.com/ml-explore/mlx-swift/issues/161).
    ///
    /// ### See Also
    /// - <doc:initialization>
    /// - ``asArray2(_:)``
    public convenience init<T: HasDType>(_ value: [[T]]) {
        let rows = value.count
        let cols = value.first?.count ?? 0
        precondition(
            value.allSatisfy { $0.count == cols },
            "MLXArray(nested): rows must have equal length (got \(value.map(\.count)))"
        )
        let flat = value.flatMap { $0 }
        self.init(flat, [rows, cols])
    }

    /// Create a 2-D ``MLXArray`` of dtype `.int32` from a row-major ``[[Int]]`` literal.
    ///
    /// Mirrors the 1-D behavior of ``init(_:_:)-(value:[Int],_)``, which produces
    /// an int32 array (and not int64) for ergonomic interop with Python /
    /// numpy.
    ///
    /// ### See Also
    /// - ``init(int64:)-([[Int]])``
    public convenience init(_ value: [[Int]]) {
        let rows = value.count
        let cols = value.first?.count ?? 0
        precondition(
            value.allSatisfy { $0.count == cols },
            "MLXArray(nested): rows must have equal length (got \(value.map(\.count)))"
        )
        let flat = value.flatMap { $0 }
        self.init(flat, [rows, cols])
    }

    /// Create a 2-D `.int64` ``MLXArray`` from a row-major ``[[Int]]`` literal.
    public convenience init(int64 value: [[Int]]) {
        let rows = value.count
        let cols = value.first?.count ?? 0
        precondition(
            value.allSatisfy { $0.count == cols },
            "MLXArray(nested): rows must have equal length (got \(value.map(\.count)))"
        )
        let flat = value.flatMap { $0 }
        self.init(int64: flat, [rows, cols])
    }

    /// Create a 2-D `.float32` ``MLXArray`` from a row-major ``[[Double]]`` literal.
    ///
    /// `Double` is converted to `Float` because MLX does not support `.float64`
    /// natively on all backends.
    public convenience init(converting value: [[Double]]) {
        let rows = value.count
        let cols = value.first?.count ?? 0
        precondition(
            value.allSatisfy { $0.count == cols },
            "MLXArray(nested): rows must have equal length (got \(value.map(\.count)))"
        )
        let flat = value.flatMap { $0 }
        self.init(converting: flat, [rows, cols])
    }

    // MARK: 3-D

    /// Create a 3-D ``MLXArray`` from a nested ``[[[T]]]`` literal in
    /// outer-to-inner order.
    ///
    /// All sub-arrays at the same depth must have equal length.
    ///
    /// Example:
    ///
    /// ```swift
    /// let a = MLXArray([[[1.0], [2.0]],
    ///                   [[3.0], [4.0]]])
    /// // a.shape == [2, 2, 1]
    /// ```
    public convenience init<T: HasDType>(_ value: [[[T]]]) {
        let d0 = value.count
        let d1 = value.first?.count ?? 0
        let d2 = value.first?.first?.count ?? 0
        precondition(
            value.allSatisfy { $0.count == d1 && $0.allSatisfy { $0.count == d2 } },
            "MLXArray(nested): inner arrays must have equal length at every depth"
        )
        let flat = value.flatMap { $0.flatMap { $0 } }
        self.init(flat, [d0, d1, d2])
    }

    /// Create a 3-D `.int32` ``MLXArray`` from a ``[[[Int]]]`` literal.
    public convenience init(_ value: [[[Int]]]) {
        let d0 = value.count
        let d1 = value.first?.count ?? 0
        let d2 = value.first?.first?.count ?? 0
        precondition(
            value.allSatisfy { $0.count == d1 && $0.allSatisfy { $0.count == d2 } },
            "MLXArray(nested): inner arrays must have equal length at every depth"
        )
        let flat = value.flatMap { $0.flatMap { $0 } }
        self.init(flat, [d0, d1, d2])
    }

    /// Create a 3-D `.int64` ``MLXArray`` from a ``[[[Int]]]`` literal.
    public convenience init(int64 value: [[[Int]]]) {
        let d0 = value.count
        let d1 = value.first?.count ?? 0
        let d2 = value.first?.first?.count ?? 0
        precondition(
            value.allSatisfy { $0.count == d1 && $0.allSatisfy { $0.count == d2 } },
            "MLXArray(nested): inner arrays must have equal length at every depth"
        )
        let flat = value.flatMap { $0.flatMap { $0 } }
        self.init(int64: flat, [d0, d1, d2])
    }

    /// Create a 3-D `.float32` ``MLXArray`` from a ``[[[Double]]]`` literal.
    public convenience init(converting value: [[[Double]]]) {
        let d0 = value.count
        let d1 = value.first?.count ?? 0
        let d2 = value.first?.first?.count ?? 0
        precondition(
            value.allSatisfy { $0.count == d1 && $0.allSatisfy { $0.count == d2 } },
            "MLXArray(nested): inner arrays must have equal length at every depth"
        )
        let flat = value.flatMap { $0.flatMap { $0 } }
        self.init(converting: flat, [d0, d1, d2])
    }

    // MARK: 4-D

    /// Create a 4-D ``MLXArray`` from a nested ``[[[[T]]]]`` literal.
    ///
    /// All sub-arrays at the same depth must have equal length.
    ///
    /// Useful for inputs shaped like `[batch, channels, height, width]` or
    /// `[batch, height, width, channels]`.
    public convenience init<T: HasDType>(_ value: [[[[T]]]]) {
        let d0 = value.count
        let d1 = value.first?.count ?? 0
        let d2 = value.first?.first?.count ?? 0
        let d3 = value.first?.first?.first?.count ?? 0
        precondition(
            value.allSatisfy {
                $0.count == d1
                    && $0.allSatisfy {
                        $0.count == d2 && $0.allSatisfy { $0.count == d3 }
                    }
            },
            "MLXArray(nested): inner arrays must have equal length at every depth"
        )
        let flat = value.flatMap { $0.flatMap { $0.flatMap { $0 } } }
        self.init(flat, [d0, d1, d2, d3])
    }

    /// Create a 4-D `.int32` ``MLXArray`` from a ``[[[[Int]]]]`` literal.
    public convenience init(_ value: [[[[Int]]]]) {
        let d0 = value.count
        let d1 = value.first?.count ?? 0
        let d2 = value.first?.first?.count ?? 0
        let d3 = value.first?.first?.first?.count ?? 0
        precondition(
            value.allSatisfy {
                $0.count == d1
                    && $0.allSatisfy {
                        $0.count == d2 && $0.allSatisfy { $0.count == d3 }
                    }
            },
            "MLXArray(nested): inner arrays must have equal length at every depth"
        )
        let flat = value.flatMap { $0.flatMap { $0.flatMap { $0 } } }
        self.init(flat, [d0, d1, d2, d3])
    }

    /// Create a 4-D `.int64` ``MLXArray`` from a ``[[[[Int]]]]`` literal.
    public convenience init(int64 value: [[[[Int]]]]) {
        let d0 = value.count
        let d1 = value.first?.count ?? 0
        let d2 = value.first?.first?.count ?? 0
        let d3 = value.first?.first?.first?.count ?? 0
        precondition(
            value.allSatisfy {
                $0.count == d1
                    && $0.allSatisfy {
                        $0.count == d2 && $0.allSatisfy { $0.count == d3 }
                    }
            },
            "MLXArray(nested): inner arrays must have equal length at every depth"
        )
        let flat = value.flatMap { $0.flatMap { $0.flatMap { $0 } } }
        self.init(int64: flat, [d0, d1, d2, d3])
    }

    /// Create a 4-D `.float32` ``MLXArray`` from a ``[[[[Double]]]]`` literal.
    public convenience init(converting value: [[[[Double]]]]) {
        let d0 = value.count
        let d1 = value.first?.count ?? 0
        let d2 = value.first?.first?.count ?? 0
        let d3 = value.first?.first?.first?.count ?? 0
        precondition(
            value.allSatisfy {
                $0.count == d1
                    && $0.allSatisfy {
                        $0.count == d2 && $0.allSatisfy { $0.count == d3 }
                    }
            },
            "MLXArray(nested): inner arrays must have equal length at every depth"
        )
        let flat = value.flatMap { $0.flatMap { $0.flatMap { $0 } } }
        self.init(converting: flat, [d0, d1, d2, d3])
    }
}

// MARK: - Extracting nested Swift Arrays from an MLXArray

extension MLXArray {

    /// Return the contents as a row-major 2-D Swift array.
    ///
    /// The array's ``dtype`` is preserved if it equals `T.dtype`; otherwise
    /// values are cast to `T` (matching the semantics of ``asArray(_:)``).
    /// The receiver must be a 2-D ``MLXArray``.
    ///
    /// Example:
    ///
    /// ```swift
    /// let a = MLXArray([[1.0, 2.0], [3.0, 4.0]])
    /// let rows = a.asArray2(Float.self)
    /// // rows == [[1.0, 2.0], [3.0, 4.0]]
    /// ```
    ///
    /// ### See Also
    /// - <doc:conversion>
    /// - ``asArray(_:)``
    /// - ``init(_:)-([[T]])``
    public func asArray2<T: HasDType>(_ type: T.Type) -> [[T]] {
        precondition(
            ndim == 2,
            "asArray2 requires a 2-D MLXArray (got \(ndim)-D, shape \(shape))"
        )
        let rows = dim(0)
        let cols = dim(1)
        let flat = asArray(type)
        var result = [[T]]()
        result.reserveCapacity(rows)
        for r in 0..<rows {
            let start = r * cols
            result.append(Array(flat[start..<(start + cols)]))
        }
        return result
    }

    /// Return the contents as a 3-D nested Swift array.
    ///
    /// The receiver must be a 3-D ``MLXArray``.
    public func asArray3<T: HasDType>(_ type: T.Type) -> [[[T]]] {
        precondition(
            ndim == 3,
            "asArray3 requires a 3-D MLXArray (got \(ndim)-D, shape \(shape))"
        )
        let d0 = dim(0)
        let d1 = dim(1)
        let d2 = dim(2)
        let flat = asArray(type)
        var result = [[[T]]]()
        result.reserveCapacity(d0)
        for i in 0..<d0 {
            var plane = [[T]]()
            plane.reserveCapacity(d1)
            for j in 0..<d1 {
                let start = (i * d1 + j) * d2
                plane.append(Array(flat[start..<(start + d2)]))
            }
            result.append(plane)
        }
        return result
    }

    /// Return the contents as a 4-D nested Swift array.
    ///
    /// The receiver must be a 4-D ``MLXArray``.
    public func asArray4<T: HasDType>(_ type: T.Type) -> [[[[T]]]] {
        precondition(
            ndim == 4,
            "asArray4 requires a 4-D MLXArray (got \(ndim)-D, shape \(shape))"
        )
        let d0 = dim(0)
        let d1 = dim(1)
        let d2 = dim(2)
        let d3 = dim(3)
        let flat = asArray(type)
        var result = [[[[T]]]]()
        result.reserveCapacity(d0)
        for i in 0..<d0 {
            var cube = [[[T]]]()
            cube.reserveCapacity(d1)
            for j in 0..<d1 {
                var plane = [[T]]()
                plane.reserveCapacity(d2)
                for k in 0..<d2 {
                    let start = ((i * d1 + j) * d2 + k) * d3
                    plane.append(Array(flat[start..<(start + d3)]))
                }
                cube.append(plane)
            }
            result.append(cube)
        }
        return result
    }
}
