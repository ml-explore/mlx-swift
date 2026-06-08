// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import Numerics

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
