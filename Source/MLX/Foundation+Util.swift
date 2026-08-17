// Copyright © 2024 Apple Inc.

import Foundation

/// Keep the common shape/axis case on the stack without allowing an untrusted collection size to
/// produce an arbitrarily large stack allocation.
@usableFromInline
let mlxInteropStackBufferCapacity = 64

extension Collection where Element == Int {

    /// Calls `body` with a scoped `Int32` representation suitable for C APIs.
    ///
    /// Unlike ``asInt32``, this does not allocate a temporary `Array`. The pointer must not be
    /// stored or otherwise escape `body`.
    @inlinable
    func withInt32Buffer<Result>(
        _ body: (UnsafePointer<Int32>?, Int) throws -> Result
    ) rethrows -> Result {
        guard !isEmpty else { return try body(nil, 0) }

        if count <= mlxInteropStackBufferCapacity {
            return try withUnsafeTemporaryAllocation(of: Int32.self, capacity: count) { buffer in
                var destination = buffer.startIndex
                for value in self {
                    buffer.initializeElement(at: destination, to: Int32(value))
                    buffer.formIndex(after: &destination)
                }
                return try body(buffer.baseAddress, buffer.count)
            }
        }

        let buffer = UnsafeMutableBufferPointer<Int32>.allocate(capacity: count)
        var initializedCount = 0
        defer {
            buffer.baseAddress?.deinitialize(count: initializedCount)
            buffer.deallocate()
        }
        for value in self {
            buffer.initializeElement(at: initializedCount, to: Int32(value))
            initializedCount += 1
        }
        return try body(buffer.baseAddress, buffer.count)
    }
}

extension [Int] {

    /// Convenience to coerce array of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt32: [Int32] {
        self.map { Int32($0) }
    }

    /// Convenience to coerce array of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt64: [Int64] {
        self.map { Int64($0) }
    }
}

extension Sequence<Int> {

    /// Convenience to coerce  sequence of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt32: [Int32] {
        self.map { Int32($0) }
    }

    @inlinable
    var asInt64: [Int64] {
        self.map { Int64($0) }
    }
}

extension Int {

    /// Convenience to convert `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var int32: Int32 { Int32(self) }

    @inlinable
    var int64: Int64 { Int64(self) }
}
