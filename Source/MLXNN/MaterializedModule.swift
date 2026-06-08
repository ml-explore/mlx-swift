// Copyright © 2026 Apple Inc.

import MLX

open class MaterializedModule<LayerType: Module>: Module, @unchecked Sendable {

    let base: LayerType

    public init(_ base: consuming LayerType) throws {
        self.base = base
        try self.base.materialize()

        // force caching of accessors (buildCaches)
        _ = base.items()
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
