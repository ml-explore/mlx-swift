// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// Poison propagation for the non-throwing operator API.
///
/// Operators like `a + b` route through `operator +` and cannot `throw`. Rather
/// than return a valid-looking `MLXArray` backed by an empty `mlx_array` (the
/// current behaviour, which produces cascading secondary errors), we attach the
/// original ``MLXError`` to the result. The error then rides *inside the value*:
/// the next `try eval()`, `try item()`, or `try asArray()` that touches it
/// rethrows the original, first error — preserving attribution and eliminating
/// the zombie cascade.
///
/// Storage is a side-table keyed by object identity so `MLXArray`'s layout and
/// `Cmlx` ownership are untouched.
extension MLXArray {

    private static let poisonTable = PoisonTable()

    /// Attach an error to this array. Idempotent — the first error wins,
    /// matching the "first error" semantics of the old `ErrorBox`.
    func poison(_ error: MLXError) {
        Self.poisonTable.set(self, error)
    }

    /// The attached error, if this array (or the op that produced it) failed.
    public var poisonError: MLXError? {
        Self.poisonTable.get(self)
    }

    /// Rethrow the attached error if present. Called by the throwing sync points
    /// before handing the array to the backend.
    func throwIfPoisoned() throws {
        if let error = poisonError { throw error }
    }
}

/// Thread-safe identity-keyed side table. `NSMapTable` with weak keys releases
/// entries when the `MLXArray` is deallocated, so poison never leaks.
private final class PoisonTable: @unchecked Sendable {
    private let lock = NSLock()
    private let table = NSMapTable<MLXArray, NSError>.weakToStrongObjects()

    func set(_ key: MLXArray, _ error: MLXError) {
        lock.withLock {
            guard table.object(forKey: key) == nil else { return } // first wins
            table.setObject(error as NSError, forKey: key)
        }
    }

    func get(_ key: MLXArray) -> MLXError? {
        lock.withLock { table.object(forKey: key) as? MLXError }
    }
}
