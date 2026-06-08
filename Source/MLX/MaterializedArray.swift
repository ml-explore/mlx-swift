// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import Numerics

/// A fully-evaluated, immutable ``MLXArray`` that is safe to share across
/// concurrency domains.
///
/// `MLXArray` is normally lazy: it is a handle to a node in a computation
/// graph that is not realized until something forces it (a scalar read,
/// an ``eval(_:)-(MLXArray...)`` call, etc.).  These unrealized arrays are
/// not thread safe -- they require mutation for their evaluation.
///
/// `MaterializedArray` is a snapshot that closes that gap:
///
/// - The contents are evaluated at the moment of construction, so no further
///   graph work is pending.
/// - You can only create an instance via ``MLXArray/materialized()``
///   or ``materialize(_:)->MaterializedArray``
/// - Mutation methods are marked as unavailable and will `fatalError`
///   if you somehow manage to call them.
/// - It is declared `@unchecked Sendable` and may be passed freely between
///   tasks, actors, and other concurrency boundaries.
///
/// Construction is intentionally narrow.  Obtain one via:
///
/// ```swift
/// let m1 = a.materialized()
/// let m2 = materialize(a)
/// ```
///
/// A `MaterializedArray` is itself an ``MLXArray`` and can be used anywhere
/// an `MLXArray` is accepted.  Operations involving one still produce
/// ordinary (lazy) `MLXArray` results — only the snapshot itself is frozen.
///
/// ### See Also
/// - ``MLXArray/materialized()``
/// - ``materialize(_:)->MaterializedArray``
public final class MaterializedArray: MLXArray, @unchecked Sendable {

    init(materialized ctx: consuming mlx_array) {
        super.init(ctx)
    }

    @available(*, unavailable, message: "MaterializedArray can only be created via materialize()")
    required public convenience init(arrayLiteral elements: Int32...) {
        fatalError("unavailable")
    }

    final public override func materialized() -> MaterializedArray {
        self
    }

    // MARK: - Update sealing

    @available(*, unavailable)
    override public func _updateInternal(_ array: MLXArray) {
        // Note that this might be called via:
        //
        // a[1] = b
        // a += b
        fatalError("unavailable")
    }

    // MARK: - In place operators

    @available(*, unavailable)
    public static func += (lhs: inout MaterializedArray, rhs: MLXArray) {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public static func += (lhs: inout MaterializedArray, rhs: some ScalarOrArray) {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public static func -= (lhs: inout MaterializedArray, rhs: MLXArray) {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public static func -= (lhs: inout MaterializedArray, rhs: some ScalarOrArray) {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public static func *= (lhs: inout MaterializedArray, rhs: MLXArray) {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public static func *= (lhs: inout MaterializedArray, rhs: some ScalarOrArray) {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public static func /= (lhs: inout MaterializedArray, rhs: MLXArray) {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public static func /= (lhs: inout MaterializedArray, rhs: some ScalarOrArray) {
        fatalError("unavailable")
    }

}
