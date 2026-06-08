// Copyright © 2026 Apple Inc.

import MLX

/// A `Module` whose parameters have been materialized so that the whole
/// module is safe to share across concurrency domains.
///
/// A normal ``Module`` is not `Sendable`: its parameters are ``MLXArray``
/// instances, which are lazy and may be mutated in place during evaluation
/// or training.  `MaterializedModule` wraps a base module, evaluates every
/// parameter, replaces each with a ``MaterializedArray``, and seals the
/// mutation surface so the wrapped module cannot be modified through this
/// reference.
///
/// ## Construction and the `consuming` contract
///
/// The initializer takes its base module as `consuming`:
///
/// ```swift
/// let lm = try MaterializedModule(Linear(10, 10))
/// ```
///
/// `consuming` expresses intent — **the caller must not retain or use the
/// original `base` reference after passing it in**.  Because `Module` is a
/// reference type, Swift cannot enforce this for you: a caller who keeps a
/// reference and later mutates it will violate the `Sendable` invariant
/// that `MaterializedModule` relies on.  In other words, `Sendable` here is
/// a contract you can follow rather than a guarantee the compiler proves.
/// The recommended pattern is to construct the base module inline at the
/// `MaterializedModule(...)` call site, as shown above, so no other
/// reference exists.
///
/// ## What is sealed
///
/// The following `Module` operations are marked `@available(*, unavailable)`
/// on `MaterializedModule` and will trap if called:
///
/// - `update(parameters:...)` and `update(modules:...)`
/// - `updateModule(key:_:)`
/// - `apply(filter:map:)`
/// - `freeze(...)` / `unfreeze(...)`
/// - `train(_:)`
///
/// ## Calling the wrapped module
///
/// `MaterializedModule` does not itself know how to invoke `base`; that is
/// added per-layer-shape via an extension that constrains `LayerType`.
/// For example, every ``UnaryLayer`` already supports being called with a
/// single `MLXArray`, so the package ships:
///
/// ```swift
/// extension MaterializedModule where LayerType: UnaryLayer {
///     public func callAsFunction(_ x: MLXArray) -> MLXArray {
///         base(x)
///     }
/// }
/// ```
///
/// Layers with different call signatures can be wrapped the same way.  For a
/// transformer-style block that takes a tensor and an attention mask:
///
/// ```swift
/// extension MaterializedModule where LayerType: AttentionBlock {
///     public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
///         base(x, mask: mask)
///     }
/// }
/// ```
///
/// The pattern is always the same: constrain `LayerType` to the protocol or
/// concrete type that exposes the call you want, then forward to `base`.
open class MaterializedModule<LayerType: Module>: Module, @unchecked Sendable {

    let base: LayerType

    public init(_ base: consuming LayerType) throws {
        self.base = base
        try self.base.materialize()

        // force caching of accessors (buildCaches)
        _ = self.base.items()
        super.init()
        _ = self.items()
    }

    override func materialize() throws {
        // NOP
    }

    @available(*, unavailable)
    @discardableResult
    open override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    @discardableResult
    open override func apply(
        filter: (Module, String, ModuleItem) -> Bool = Module.filterValidParameters,
        map: @escaping (MLXArray) -> MLXArray
    ) -> Self {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    @discardableResult
    open override func update(
        modules: ModuleChildren, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    open override func updateModule(key: String, _ value: Any) throws {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        fatalError("unavailable")
    }

    @available(*, unavailable)
    public override func train(_ mode: Bool = true) {
        fatalError("unavailable")
    }

}

extension MaterializedModule where LayerType: UnaryLayer {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        base(x)
    }
}
