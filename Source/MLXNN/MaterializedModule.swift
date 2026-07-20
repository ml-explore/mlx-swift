// Copyright © 2026 Apple Inc.

import MLX

/// A container for a `Module` whose parameters have been materialized so that
/// the whole module is safe to share across concurrency domains.
///
/// A normal ``Module`` is not `Sendable`: its parameters are `MLXArray`
/// instances, which are lazy and may be mutated in place during evaluation
/// or training.  `MaterializedModule` wraps a base module, evaluates every
/// parameter, replaces each with a `MaterializedArray`, and seals the
/// mutation surface so the wrapped module cannot be modified through this
/// reference.
///
/// Note: only parameters that can be be mutated, e.g. are wrapped with `@ParameterInfo`,
/// will actually be updated to the `MaterializedArray` type.  All others will
/// be evaluated and are materialized, even if they do not have the type
/// indicating that fact.
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
/// that `MaterializedModule` relies on.  As a runtime backstop, the base
/// module and all of its descendants are sealed during initialization —
/// any subsequent call to a mutating API (`update(parameters:)`,
/// `update(modules:)`, `apply(...)`, `freeze`/`unfreeze`, `train`) on the
/// retained reference will trap with `fatalError`.  In other words,
/// `Sendable` here is a contract you can follow rather than a guarantee
/// the compiler proves, but a violation fails loudly rather than silently.
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
///         _base(x)
///     }
/// }
/// ```
///
/// A pattern that is used in `mlx-swift-lm` works like this.
///
/// The model is defined in terms of a protocol.  All language models (LLMs and VLMs)
/// implement this protocol.  Note: in reality there are more methods and
/// a `BaseLanguageModel` that lifts out methods useful to Embedding models.
/// For this illustration it is simplified as those details are irrelevant to the technique.
///
/// ```swift
/// public protocol LanguageModel {
///     func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray
/// }
/// ```
///
/// Then an extension to `MaterializedModule` is created where the
/// interior layer is of the given type and `MaterializedModule` is made
/// to conform to the type:
///
/// ```swift
/// // this allows access to _base
/// @_spi(MaterializedModule) import MLXNN
///
/// extension MaterializedModule: LanguageModel where LayerType: LanguageModel {
///
///     public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
///         _base.callAsFunction(inputs, cache: cache)
///     }
/// }
/// ```
///
/// Finally there is a specific way that types can consume this.  Note the
/// use of `any` and `some`.
///
/// ```swift
/// public struct ModelContext: Sendable {
///     public var model: any LanguageModel & Sendable
///
///     public init(
///     model: some LanguageModel, ...
///     ) {
///         self.model = MaterializedModule(model)
///         ...
///     }
/// }
/// ```
///
/// ### Design Notes
///
/// Why use extensions?  Why not subclass?  Certainly it could, but it does require that the type you
/// are specializing to is a subtype of `Module`.  You can't do this:
///
/// ```swift
/// class MaterializedUnaryLayer: MaterializedModule<UnaryLayer> { ... }
/// ```
///
/// because `UnaryLayer` is not a `Module`.
///
/// The class is `open` so you could give it a try.
open class MaterializedModule<LayerType: Module>: IndentedDescription, @unchecked Sendable {

    /// Usable by extensions to implement `callAsFunction()`
    @_spi(MaterializedModule)
    public let _base: LayerType

    /// Sum of all the `nbytes` of the parameters in the encapsulated model.
    public let parameterNBytes: Int

    /// Sum of all the `parameterCount` of the modules in the encapsulated model.
    public let parameterCount: Int

    public init(_ base: consuming LayerType) {
        self._base = base
        self._base.materialize()

        // seal the consumed base so that any retained reference held by a
        // caller (despite the `consuming` contract) traps on mutation
        // rather than silently violating Sendable
        self._base._sealImmutable()

        parameterNBytes = _base.parameters().reduce(0) { $0 + $1.nbytes }
        parameterCount = _base.parameterCount
    }

    /// Return a `NestedDictionary<String, MaterializedArray>` for all parameters in the
    /// model (all layers).
    public func parameters() -> NestedDictionary<String, MaterializedArray> {
        // Note: the enclosed Module will have had all its parameters evaluated
        // in init, but because it can't replace all parameters they may be typed
        // as MLXArray.
        _base.parameters().mapValues { $0.materialized() }
    }

    public func description(indent: Int) -> String {
        _base.description(indent: indent)
    }
}

extension MaterializedModule where LayerType: UnaryLayer {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        _base(x)
    }
}
