// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// `NestedDictionary` structure of `MLXArray`
public typealias ModuleParameters = NestedDictionary<String, MLXArray>

/// `NestedDictionary` structure of `Module`
public typealias ModuleChildren = NestedDictionary<String, Module>

/// `NestedDictionary` structure of `ModuleValue` from ``Module/items()``
public typealias ModuleItems = NestedDictionary<String, ModuleValue>

/// Single item from ``Module/items()``
public typealias ModuleItem = NestedItem<String, ModuleValue>

/// Base class for building neural networks with MLX.
///
/// The workhorse of any neural network library is the ``Module`` class. In MLX
/// the ``Module`` class is a container of `MLXArray` or ``Module`` instances. Its
/// main function is to provide a way to recursively access and update its
/// parameters and those of its submodules.
///
/// All the layers provided in <doc:layers> subclass this class and
/// your models should do the same.  See <doc:custom-layers>.
///
/// ### Parameters
///
/// A `Module` can contain other `Module` instances or `MLXArray`
/// instances in structures of `Array` and `Dictionary`. The `Module`
/// then allows recursively extracting all the `MLXArray` instances
/// using ``parameters()``
///
/// In addition, the `Module` has the concept of trainable and non trainable
/// parameters (called "frozen"). When using `valueAndGrad()` or `grad()`
/// the gradients are returned only with respect to the trainable parameters.
/// All arrays in a module are trainable unless they are added in the "frozen"
/// set by calling ``freeze(recursive:keys:strict:)``.
///
/// ``valueAndGrad(model:_:)-12a2c`` the gradients returned will
/// be with respect to these trainable parameters.
///
/// ### Example
///
/// Here is an example multi-layer perception (MLP):
///
/// ```swift
/// import MLX
/// import MLXNN
///
/// class MyMLP : Module, UnaryLayer {
///     let inProjection: Linear
///     let outProjection: Linear
///
///     init(inputDimensions: Int, outputDimensions: Int, hiddenDimensions: Int = 16) {
///         self.inProjection = Linear(inputDimensions, hiddenDimensions)
///         sel.outProjection = Linear(hiddenDimensions, outputDimensions)
///     }
///
///     func callAsFunction(_ x: MLXArray) -> MLXArray {
///         var x = inProjection(x)
///         x = maximum(x, 0)
///         return outProjection(x)
///     }
/// }
///
/// let model = MyMLP(inputDimensions: 2, outputDimensions: 1)
///
/// // All the model parameters are created but since MLX is lazy by
/// // default, they are not evaluated yet. Calling `eval` actually
/// // allocates memory and initializes the parameters.
/// eval(model)
///
/// // and projecting a value through the model:
/// let input: MLXArray ...
/// let result = model(input)
/// ```
///
/// > Please read <doc:custom-layers> for more information about implementing custom layers
/// including how to override the module and parameter keys and allowing dynamic updates of the
/// module structure to occur via ``update(modules:verify:)``.
///
/// ### Training
///
/// See <doc:training>
///
/// ### Mutation
///
/// All mutation of parameters and modules must go through ``update(parameters:)`` and
/// ``update(modules:)``.  This is important because `Module` uses reflection (`Mirror`) to
/// find parameters and modules but caches the values.  These two methods make sure the cache
/// is kept up-to-date.
///
/// ### See Also
/// - <doc:custom-layers>
open class Module {

    /// Flag to indicate whether the module is being trained.  Manipulated via
    /// ``train(_:)``.
    ///
    /// ### See Also
    /// - ``didSetTrain(_:)``
    public private(set) var training = true

    /// See ``noGrad()``
    private var _noGrad = Set<String>()

    private var _items: ModuleItems!
    private var _setters: [String: TypeErasedSetter]!

    /// Initializes the module.
    public init() {
        buildCaches()
    }

    private func buildCaches() {
        var items = ModuleItems()
        var setters = [String: TypeErasedSetter]()

        mirrorUpToModule(module: self) { c in
            if let (key, value) = ModuleValue.fromMirror(c) {
                items[key] = value

                if let (_, _, setter) = isModuleInfo(c.value) {
                    setters[key] = setter
                }
            }

            return .next
        }

        self._items = items
        self._setters = setters
    }

    /// Return a `NestedDictionary` structure of ``ModuleItem`` representing the ivars of the `Module` instance.
    ///
    /// This is typically not used directly -- it is part of the implementation of ``filterMap(filter:map:isLeaf:)``
    /// and ``update(parameters:)`` for example.
    ///
    /// Subclasses could potentially override this to provide custom introspection.
    open func items() -> ModuleItems {
        _items
    }

    /// Describe extra parameters.
    ///
    /// This will print a description of ``ModuleValue/other(_:)`` ivars, e.g.:
    ///
    /// ```swift
    /// let eps: Float
    /// ```
    ///
    /// Subclasses can override this to print custom information, e.g. shape information
    /// derived from parameters:
    ///
    /// ```swift
    /// public override func describeExtra(_ indent: Int) -> String {
    ///     "(inputDimensions=\(weight.dim(1)), outputDimensions=\(weight.dim(0)), bias=\(bias == nil ? "false" : /// "true"))"
    /// }
    /// ```
    ///
    /// ### See Also
    /// - ``description(indent:)``
    open func describeExtra(_ indent: Int) -> String {
        let other = filterMap(filter: Self.filterOther, map: Self.mapOther())
        let kv = other.sorted { lhs, rhs in lhs.key < rhs.key }

        if !kv.isEmpty {
            var result = [String]()

            for (k, v) in kv {
                result.append("\(k)=\(String(describing: v))")
            }

            return "(\(result.joined(separator: ", ")))"
        } else {
            return ""
        }
    }

    /// Recursively filter and map the contents of the module and its children and produce a `NestedDictionary`
    /// with the results.
    ///
    /// Traverses the ``ModuleItems`` produced by ``items()`` and filters and maps their contents.  For each item
    /// in the `ModuleItems` this will call the `filter` to determine if it should be included.  There are a number of
    /// predefined filters available, see <doc:module-filters>.  ``filterAll`` will accept all values while
    /// ``filterValidParameters`` will only examine structure and parameters.
    ///
    /// The `map` function transforms the visited values.  By default it is identity and will just return the ``ModuleItem``
    /// directly.  There are a number of helper functions like ``mapParameters(map:)`` and ``mapModule(map:)``
    /// that can help deal with types like `MLXArray` or ``Module``.  For example:
    ///
    /// ```swift
    /// // produces NestedDictionary<String, [Int]> for the parameters attached
    /// // directly to this module
    /// let localParameterShapes = module.filterMap(
    ///     filter: Module.filterLocalParameters,
    ///     map: Module.mapParameters { $0.shape })
    /// ```
    ///
    /// The `isLeaf` function is called to determine if the value should be transformed via the `map` or
    /// if the structure should be traversed.  For example this will collect the leaf modules:
    ///
    /// ```swift
    /// // produces NestedDictionary<String, MLXArray> for all leaf modules
    /// let leafParameters = module.filterMap(
    ///     filter: Module.filterValidParameters,
    ///     map: Module.mapParameters { $0 },
    ///     isLeaf: Module.isLeafModuleNoChildren)
    /// ```
    ///
    /// - Parameters:
    ///   - filter: filter function that determines if the (Module, Key, Item) tuple should be examined
    ///   - map: Transformation of the values.  By default this is identity but the caller can transform to other types, etc.  See ``mapParameters(map:)`` and others for helper functions that can assist with common types.
    ///   - isLeaf: closure that determines if a value is a leaf or not.
    /// - Returns: `NestedDictionary` matching the structure with mapped values.
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``parameters()``
    /// - ``mapParameters(map:isLeaf:)``
    /// - ``modules()``
    /// - ``items()``
    open func filterMap<Result>(
        filter: (Module, String, ModuleItem) -> Bool,
        map: (ModuleItem) -> Result? = { $0 },
        isLeaf: (Module, String, ModuleItem) -> Bool = Module.isLeafDefault
    ) -> NestedDictionary<String, Result> {
        var result = NestedDictionary<String, Result>()

        // recursive unwrap rules
        func unwrap(_ vk: String, _ v: ModuleItem) -> NestedItem<String, Result> {
            // handle values we want to emit
            if isLeaf(self, vk, v) {
                if let v = map(v) {
                    return .value(v)
                } else {
                    return .none
                }
            }

            // handle structure
            switch v {
            case .none:
                return .none

            case .value(.module(let module)):
                return module.filterMap(filter: filter, map: map, isLeaf: isLeaf).asItem()

            case .value:
                // e.g. parameters, other -- these were handled via the isLeaf case above
                // or they can be ignored as they are not structural
                return .none

            case .dictionary(let dictionary):
                var result = [String: NestedItem<String, Result>]()
                var isAllNone = true

                for (k, v) in dictionary {
                    let tk = "\(vk).\(k)"
                    let ks = String(describing: k)
                    if filter(self, tk, v) {
                        result[ks] = unwrap(tk, v)
                        isAllNone = false
                    } else {
                        result[ks] = NestedItem<String, Result>.none
                    }
                }
                return isAllNone ? .none : .dictionary(result)

            case .array(let array):
                var result = [NestedItem<String, Result>]()
                var isAllNone = true

                for (i, vi) in array.enumerated() {
                    let tk = "\(vk).\(i)"
                    if filter(self, tk, vi) {
                        result.append(unwrap(tk, vi))
                        isAllNone = false
                    } else {
                        result.append(.none)
                    }
                }
                return isAllNone ? .none : .array(result)

            default:
                fatalError("Unexpected leaf \(vk) = \(v)")
            }
        }

        for (key, item) in items() {
            if !filter(self, key, item) {
                continue
            }

            let unwrapped = unwrap(key, item)
            switch unwrapped {
            case .none:
                break
            default:
                result[key] = unwrapped
            }
        }

        return result
    }

    /// Apply a `map` to all parameters (``ModuleValue/parameters(_:)``) in the module and its children.
    ///
    /// For example:
    ///
    /// ```swift
    /// // shapes is NestedDictionary<String, [Int]>
    /// let shapes = module.mapParameters { array in
    ///     array.shape
    /// }
    /// ```
    ///
    /// This is equivalent to:
    ///
    /// ```swift
    /// filterMap(
    ///     filter: Self.filterValidParameters,
    ///     map: Self.mapParameters(map: map),
    ///     isLeaf: isLeaf)
    /// ```
    ///
    /// - Parameters:
    ///   - map: closure that transforms `MLXArray` into `Result` type or nil
    ///   - isLeaf: optional leaf function
    /// - Returns: `NestedDictionary` of mapped results
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``mapParameters(map:)``
    open func mapParameters<Result>(
        map: @escaping (MLXArray) -> Result? = { $0 },
        isLeaf: (Module, String, ModuleItem) -> Bool = Module.isLeafDefault
    ) -> NestedDictionary<String, Result> {
        filterMap(
            filter: Self.filterValidParameters,
            map: Self.mapParameters(map: map),
            isLeaf: isLeaf)
    }

    /// Return a `NestedDictionary<String, MLXArray>` for all parameters in the
    /// model (all layers).
    open func parameters() -> ModuleParameters {
        filterMap(filter: Self.filterValidParameters, map: Self.mapParameters())
    }

    /// Return a `NestedDictionary<String, MLXArray>` for all trainable parameters in the
    /// model (all layers).
    ///
    /// This omits ``freeze(recursive:keys:strict:)`` (frozen) parameters.
    open func trainableParameters() -> ModuleParameters {
        filterMap(filter: Self.filterTrainableParameters, map: Self.mapParameters())
    }

    /// Produces a `NestedDictionary<String, Module>` for all direct children of the module.
    open func children() -> ModuleChildren {
        filterMap(filter: Self.filterValidChild, map: Self.mapModule(), isLeaf: Self.isLeafModule)
    }

    /// Produces a `NestedDictionary<String, Module>` for all leaf modules module.
    ///
    /// ### See Also
    /// - ``isLeafModuleNoChildren``
    open func leafModules() -> ModuleChildren {
        filterMap(
            filter: Self.filterValidChild, map: Self.mapModule(),
            isLeaf: Self.isLeafModuleNoChildren)
    }

    /// Options for verifying ``update(parameters:verify:)`` and ``update(modules:verify:)``.
    public struct VerifyUpdate: OptionSet, Sendable {
        public init(rawValue: Int) {
            self.rawValue = rawValue
        }

        public let rawValue: Int

        /// Check that all keys are used.  This is useful to ensure that e.g. all loaded parameters
        /// are used -- there are no names that don't match.
        static public let noUnusedKeys = VerifyUpdate(rawValue: 1 << 0)

        static public let allModelKeysSet = VerifyUpdate(rawValue: 1 << 1)
        static public let shapeMismatch = VerifyUpdate(rawValue: 1 << 2)

        static public let all = VerifyUpdate(rawValue: -1)
        static public let none = VerifyUpdate([])
    }

    /// A non-throwing version of ``update(parameters:verify:)``.
    ///
    /// This passes `verify: .none`.  Note that there may still be `fatalErrors()` if
    /// for example an `MLXArray` is set on a `Module`.
    @discardableResult
    public func update(parameters: ModuleParameters) -> Self {
        try! update(parameters: parameters, verify: .none)
    }

    /// Replace the parameters of this `Module` with the provided parameters.
    ///
    /// This will replace the parameters in the `Module` recursively with the given
    /// ``ModuleParameters`` structure.  For example:
    ///
    /// ```swift
    /// // double all the parameters in the model
    /// model.update(parameters: module.mapParameters(map: { $0 * 2 })
    ///
    /// // equivalent to
    /// model.apply { $0 * 2 }
    /// ```
    ///
    /// The `parameters` need not provide all values in the model -- any omitted values
    /// will be unchanged.
    ///
    /// The ``apply(filter:map:)`` can be used for similar purposes to apply changes
    /// in-place.
    ///
    /// If a parameter is missing from the update and validation indicates `.allModelKeysSet` this
    /// will call ``updateMissing(parameter:verify:path:modulePath:)`` which will
    /// throw an error.  Subclasses can override this if needed.
    ///
    /// - Parameters:
    ///   - parameters: replacement parameters in the same format that ``parameters()``
    ///     or ``mapParameters(map:isLeaf:)`` provides
    ///   - verify: options for verifying parameters
    ///
    /// ### See Also
    /// - <doc:custom-layers>
    /// - ``update(parameters:)``
    /// - ``apply(filter:map:)``
    /// - ``parameters()``
    /// - ``mapParameters(map:isLeaf:)``
    /// - ``update(modules:verify:)``
    @discardableResult
    open func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {

        let modulePath = modulePath + [describeType(self)]

        func apply(
            key: String, path: [String], _ item: ModuleItem, _ value: NestedItem<String, MLXArray>
        ) throws {
            if case .none = value, !verify.contains(.allModelKeysSet) {
                return
            }

            // item: single item from `items()`
            // value: single item with matching structure from `parameters()`
            //
            // match them up and apply the MLXArrays from value -> item

            switch (item, value) {
            case (.value(.parameters(let p)), .value(let newArray)):
                if verify.contains(.shapeMismatch), p.shape != newArray.shape {
                    throw UpdateError.mismatchedSize(
                        path: path, modules: modulePath, expectedShape: p.shape,
                        actualShape: newArray.shape)
                }
                p._updateInternal(newArray)

            case (.value(.parameters), .none):
                if Self.parameterIsValid(key) {
                    try updateMissing(
                        parameter: key, verify: verify, path: path, modulePath: modulePath)
                } else {
                    // ignore it -- this isn't a parameter that requires update
                }

            case (.array(let array), .array(let values)):
                for (i, (arrayItem, valueItem)) in zip(array, values).enumerated() {
                    try apply(key: "\(key).\(i)", path: path + ["\(i)"], arrayItem, valueItem)
                }
                if verify.contains(.allModelKeysSet) {
                    for i in values.count ..< array.count {
                        try apply(key: "\(key).\(i)", path: path + ["\(i)"], array[i], .none)
                    }
                }

            case (.array(let array), .none):
                for (i, arrayItem) in array.enumerated() {
                    try apply(key: "\(key).\(i)", path: path + ["\(i)"], arrayItem, .none)
                }

            case (.dictionary(let dictionary), .dictionary(let values)):
                for (dictionaryKey, dictionaryItem) in dictionary {
                    let newKey = "\(key).\(dictionaryKey)"
                    let path = path + [dictionaryKey]
                    if let valueItem = values[key] {
                        try apply(key: newKey, path: path, dictionaryItem, valueItem)
                    } else if verify.contains(.allModelKeysSet) {
                        try apply(key: newKey, path: path, dictionaryItem, .none)
                    }
                }

            case (.dictionary(let dictionary), .none):
                for (dictionaryKey, dictionaryItem) in dictionary {
                    let newKey = "\(key).\(dictionaryKey)"
                    let path = path + [dictionaryKey]
                    try apply(key: newKey, path: path, dictionaryItem, .none)
                }

            case (.value(.module(let module)), .dictionary(let values)):
                try module.update(
                    parameters: NestedDictionary(values: values), verify: verify, path: path,
                    modulePath: modulePath)

            case (.value(.module(let module)), .none):
                try module.update(
                    parameters: NestedDictionary(), verify: verify, path: path,
                    modulePath: modulePath)

            case (.none, .none), (.value(.none), .none), (.value(.other(_)), .none):
                break

            default:
                throw UpdateError.incompatibleItems(
                    path: path, modules: modulePath, item: item.description,
                    value: String(describing: (value.mapValues { $0.shape.description })))
            }
        }

        var processed = Set(parameters.keys)
        for (key, item) in items() {
            if let value = parameters[key] {
                processed.remove(key)
                try apply(key: key, path: path + [key], item, value)
            } else if verify.contains(.allModelKeysSet) {
                try apply(key: key, path: path + [key], item, .none)
            }
        }

        if verify.contains(.noUnusedKeys) && !processed.isEmpty {
            throw UpdateError.unhandledKeys(
                path: path, modules: modulePath, keys: processed.sorted())
        }

        return self
    }

    /// Called from ``update(parameters:verify:path:modulePath:)`` if a required parameter
    /// is missing.
    ///
    /// The default implementation will throw ``UpdateError/keyNotFound(path:modules:)``.
    ///
    /// - Parameters:
    ///   - parameter: the key for the missing parameter
    ///   - verify: verify settings
    ///   - path: path to the key (includes parameter)
    ///   - modulePath: path to the module
    open func updateMissing(
        parameter: String, verify: VerifyUpdate, path: [String], modulePath: [String]
    ) throws {
        throw UpdateError.keyNotFound(path: path, modules: modulePath)
    }

    /// Apply a closure to the parameters in a `Module` recursively.
    ///
    /// For example to change all floating point parameters to `DType.float16`:
    ///
    /// ```swift
    /// layer.apply { array in
    ///     array.dtype.isFloatingPoint ? array.asType(.float16) : array
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - filter: filter for parameters to apply to
    ///   - map: function to apply to the matched parameters
    @discardableResult
    open func apply(
        filter: (Module, String, ModuleItem) -> Bool = Module.filterValidParameters,
        map: @escaping (MLXArray) -> MLXArray
    ) -> Self {
        update(parameters: filterMap(filter: filter, map: Self.mapParameters(map: map)))
    }

    /// A non-throwing version of ``update(modules:verify:)``.
    ///
    /// This passes `verify: .none`.  Note that there may still be `fatalErrors()` if
    /// for example an `Module` is set on a `MLXArray`.
    @discardableResult
    public func update(modules: ModuleChildren) -> Self {
        try! update(modules: modules, verify: .none)
    }

    /// Replace the child modules of this `Module` with the provided replacements.
    ///
    /// This will replace the parameters in the `Module` recursively with the given
    /// ``ModuleChildren`` structure.  For example this is typically called via
    /// a helper function:
    ///
    /// ```swift
    /// if let quantization = configuration.quantization {
    ///     QuantizedLinear.quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits)
    /// }
    /// ```
    ///
    /// > Note that the modules being replace must use a ``ModuleInfo`` property wrapper -- this
    /// provides the mechanism to update the values.  Also note that the replacement models must
    /// be assignable to the ivar's type.
    ///
    /// For example:
    ///
    /// ```swift
    /// public class FeedForward : Module {
    ///
    ///     @ModuleInfo var w1: Linear
    ///     @ModuleInfo var w2: Linear
    ///     @ModuleInfo var w3: Linear
    /// ```
    ///
    /// Would be able to be replaced with ``QuantizedLinear/quantize(model:groupSize:bits:predicate:)``.
    ///
    /// The `modules` need not provide all values in the model -- any omitted values
    /// will be unchanged.
    ///
    /// - Parameters:
    ///   - modules: replacement modules in the same format as ``children()`` or ``leafModules()``
    ///   - verify: options for verifying parameters
    ///
    /// ### See Also
    /// - <doc:custom-layers>
    /// - ``update(modules:)``
    /// - ``update(parameters:verify:)``
    /// - ``children()``
    /// - ``leafModules()``
    /// - ``QuantizedLinear/quantize(model:groupSize:bits:predicate:)``
    @discardableResult
    open func update(
        modules: ModuleChildren, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {

        let modulePath = modulePath + [describeType(self)]

        func apply(
            key: String, path: [String], _ item: ModuleItem, _ value: NestedItem<String, Module>
        ) throws {
            // item: single item from `items()`
            // value: single item with matching structure from `children()`
            //
            // match them up and apply the Modules from value -> item

            switch (item, value) {
            case (.value(.parameters), .value):
                throw UpdateError.settingArrayWithModule(path: path, modules: modulePath)

            case (.array(let items), .array(let values)):
                // Could be:
                // - @ModuleInfo var modules = [ Linear(), Linear() ]
                //      - replace the array
                // - var modules: [TransformerBlock]
                //      - recurse into

                switch values.first {
                case .value:
                    // update array
                    var newModules = [Module]()
                    for (index, value) in values.enumerated() {
                        switch value {
                        case .value(let module):
                            newModules.append(module)

                        case .none:
                            // e.g. this is updating @ModuleInfo var mlp: (Linear, GELU, Linear)
                            if index < items.count {
                                switch items[index] {
                                case .value(.module(let m)):
                                    // if possible just copy forward the original item
                                    newModules.append(m)
                                default:
                                    // otherwise we don't know how to update it
                                    throw UpdateError.unableToCollectModulesFromContainer(
                                        path: path, modules: modulePath)
                                }
                            } else {
                                // past the end of items
                                throw UpdateError.unableToCollectModulesFromContainer(
                                    path: path, modules: modulePath)
                            }

                        default:
                            throw UpdateError.unableToCollectModulesFromContainer(
                                path: path, modules: modulePath)
                        }
                    }

                    try self.updateModule(key: key, newModules)

                case .dictionary:
                    // recurse
                    for (i, v) in zip(items, values) {
                        switch (i, v) {
                        case (.value(.module(let m)), .dictionary(let d)):
                            try m.update(modules: NestedDictionary(values: d), verify: verify)

                        default:
                            throw UpdateError.mismatchedContainers(
                                base: describeType(self), key: key)
                        }
                    }
                default:
                    throw UpdateError.unexpectedStructure(key: key, item: self.description)
                }

            case (.dictionary, .dictionary(let values)):
                // e.g. @ModuleInfo var modules = [ "a": Linear(), "b": Linear() ]

                var newModules = [String: Module]()
                for item in values {
                    switch item.value {
                    case .value(let module):
                        newModules[item.key] = module
                    default:
                        throw UpdateError.unableToCollectModulesFromContainer(
                            path: path, modules: modulePath)
                    }
                }

                try self.updateModule(key: key, newModules)

            case (.value(.module), .value(let newModule)):
                try self.updateModule(key: key, newModule)

            case (.none, .value(let newModule)):
                try self.updateModule(key: key, newModule)

            case (.value(.module), .none):
                try self.updateModule(key: key, Optional<Module>.none as Any)

            case (.value(.module(let module)), .dictionary(let values)):
                try module.update(modules: NestedDictionary(values: values), verify: verify)

            default:
                throw UpdateError.incompatibleItems(
                    path: path, modules: modulePath, item: item.description,
                    value: value.description)
            }
        }

        var processed = Set(modules.keys)
        for (key, item) in items() {
            if let value = modules[key] {
                processed.remove(key)
                try apply(key: key, path: path + [key], item, value)
            }
        }

        if verify.contains(.noUnusedKeys) && !processed.isEmpty {
            throw UpdateError.unhandledKeys(
                path: path, modules: modulePath, keys: processed.sorted())
        }

        // rebuild the caches because the modules may have changed
        buildCaches()

        return self
    }

    /// Set a module to a new value.
    ///
    /// The module property must be wrapped in a ``ModuleInfo``:
    ///
    /// ```swift
    /// @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    /// ```
    ///
    /// and the value must be a compatible type.
    ///
    /// This method is called via ``update(modules:)`` and is not typically called directly.  This
    /// is exposed as an overridable method for subclasses.
    ///
    /// - Parameters:
    ///   - key: module key, see ``ModuleInfo``
    ///   - value: the replacement module
    open func updateModule(key: String, _ value: Any) throws {
        if let setter = _setters[key] {
            do {
                try setter.updateModule(value)
            } catch {
                throw UpdateError.needModuleInfo(
                    "Unable to set modules for \(describeType(self)).\(key) -- maybe type mismatch: \(describeType(value)), \(error)"
                )
            }
        } else {
            throw UpdateError.needModuleInfo(
                "Unable to get @ModuleInfo for \(describeType(self)).\(key) -- must be wrapped to receive updates"
            )
        }
    }

    // `apply_to_modules()`
    open func visit(modules visitor: (String, Module) throws -> Void) rethrows {
        var stack = [(String, Module)]()
        stack.append(("", self))

        while !stack.isEmpty {
            let (prefix, module) = stack.removeLast()
            try visitor(prefix, module)

            stack.append(
                contentsOf: module.children().flattened(prefix: prefix.isEmpty ? nil : prefix))
        }
    }

    /// Return a flat array of all the `Module` in the instance (including self).
    ///
    /// ### See Also
    /// - ``namedModules()``
    /// - ``children()``
    /// - ``leafModules()``
    open func modules() -> [Module] {
        var result = [Module]()
        visit {
            result.append($1)
        }
        return result
    }

    /// Return a flat array of all the `Module` in the instance (including self) with their keys.
    ///
    /// ### See Also
    /// - ``modules()``
    /// - ``children()``
    /// - ``leafModules()``
    open func namedModules() -> [(String, Module)] {
        var result = [(String, Module)]()
        visit {
            result.append(($0, $1))
        }
        return result
    }

    private func freezeVisitor(
        keys: [String]? = nil, strict: Bool = false, update: @escaping (Module, [String]) -> Void
    ) -> (String, Module) throws -> Void {
        func visit(key: String, module: Module) throws {
            lazy var localKeys: [String] = {
                module.filterMap(filter: Self.filterLocalParameters).flattened().map { $0.0 }
            }()

            if strict, let keys {
                let localKeys = Set(localKeys)
                for key in keys {
                    if !localKeys.contains(key) {
                        throw UpdateError.keyNotFound(path: [key], modules: [describeType(self)])
                    }
                }
            }

            update(module, keys ?? localKeys)
        }

        return visit(key:module:)
    }

    /// Non-throwing variant of ``freeze(recursive:keys:strict:)`` (`strict: false)`.
    public func freeze(recursive: Bool = true, keys: [String]? = nil) {
        try! freeze(recursive: recursive, keys: keys, strict: false)
    }

    /// Freeze the `Module`'s parameters or subset.
    ///
    /// A frozen parameter does not compute gradients.  The function is idempotent -- freezing a frozen model is a no-op.
    ///
    /// For example to only train the attention parameters from a `Transformer`:
    ///
    /// ```swift
    /// model: Transformer
    /// model.freeze()
    /// try model.visit { k, m in
    ///     if k.hasSuffix("attention") {
    ///         m.unfreeze()
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - recursive: if `true` this will freeze the parameters of child `Module` recursively
    ///   - keys: optional keys to freeze -- if unspecified, will apply to all
    ///   - strict: if `true` validate that the passed keys exist
    ///
    /// ### See Also
    /// - ``freeze(recursive:keys:)``
    /// - ``unfreeze(recursive:keys:strict:)``
    open func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        let visitor = freezeVisitor(keys: keys, strict: strict) {
            $0._noGrad.formUnion($1)
            $0.didSetNoGrad($0._noGrad)
        }

        if recursive {
            try self.visit(modules: visitor)
        } else {
            try visitor("", self)
        }
    }

    /// Non-throwing variant of ``unfreeze(recursive:keys:strict:)`` (`strict: false)`.
    public func unfreeze(recursive: Bool = true, keys: [String]? = nil) {
        try! unfreeze(recursive: recursive, keys: keys, strict: false)
    }

    /// Unfreeze the `Module`'s parameters or subset.
    ///
    /// A frozen parameter does not compute gradients.  The function is idempotent -- unfreezing a frozen model is a no-op.
    ///
    /// For instance to only train the biases of a `Transformer` one can do:
    ///
    /// ```swift
    /// model: Transformer
    /// model.freeze()
    /// try model.unfreeze(keys: ["bias"])
    /// ```
    ///
    /// - Parameters:
    ///   - recursive: if `true` this will unfreeze the parameters of child `Module` recursively
    ///   - keys: optional keys to unfreeze -- if unspecified, will apply to all
    ///   - strict: if `true` validate that the passed keys exist
    ///
    /// ### See Also
    /// - ``Module/freeze(recursive:keys:)``
    /// - ``Module/unfreeze(recursive:keys:strict:)``
    open func unfreeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        let visitor = freezeVisitor(keys: keys, strict: strict) {
            $0._noGrad.subtract($1)
            $0.didSetNoGrad($0._noGrad)
        }

        if recursive {
            try self.visit(modules: visitor)
        } else {
            try visitor("", self)
        }
    }

    /// Set of property names that are frozen.  Manipulated via
    /// ``freeze(recursive:keys:strict:)`` and
    /// ``unfreeze(recursive:keys:strict:)``.
    open func noGrad() -> Set<String> {
        _noGrad
    }

    /// Called when ``noGrad()`` is updated.
    ///
    /// This is provided for subclasses to override.
    ///
    /// - Parameter noGrad: set of properties that are frozen
    ///
    /// ### See Also
    /// - ``noGrad()``
    open func didSetNoGrad(_ noGrad: Set<String>) {
    }

    /// Recursively set the model's training mode.
    ///
    /// Training mode only applies to certain layers. For example
    /// ``Dropout`` applies a random mask in training mode, but is the
    /// identity in evaluation mode.
    ///
    /// ### See Also
    /// - ``training``
    /// - ``didSetTrain(_:)``
    public func train(_ mode: Bool = true) {
        visit(modules: {
            $1.training = mode
            $1.didSetTrain(mode)
        })
    }

    /// Called when ``train(_:)`` is updated.
    ///
    /// This is provided for subclasses to override.
    ///
    /// - Parameter mode: `true` is training
    open func didSetTrain(_ mode: Bool) {
    }
}

extension Module: IndentedDescription {

    public func description(indent: Int) -> String {
        var result = ""

        result += "\(describeType(self))\(describeExtra(indent))"

        let children = self.children()

        if !children.isEmpty {
            let kv = children.sorted { lhs, rhs in lhs.key < rhs.key }
            let indentString = String(repeating: " ", count: indent)

            result += " {\n"
            for (key, value) in kv {
                result += "\(indentString)  \(key): \(indentedDescription(value, indent + 2)),\n"
            }
            result += "\(indentString)}"
        }

        return result
    }
}

extension Module: Updatable, Evaluatable {
    public func innerState() -> [MLXArray] {
        filterMap(filter: Self.filterAll, map: Self.mapParameters())
            .flattenedValues()
    }
}

/// A `Layer` (``Module`` subclass) that can be evaluated as a _unary function_.
///
/// This provides ``callAsFunction(_:)`` with a single `MLXArray` input and a single `MLXArray` output.
///
/// ### See Also
/// - <doc:layers>
/// - ``Sequential``
public protocol UnaryLayer: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

// MARK: - Filters and Maps

extension Module {

    /// Return `true` if the given parameter name is valid -- should be considered for
    /// validation, enumeration, etc.
    ///
    /// Specifically this will filter out parameters with keys starting with `_`.
    static public func parameterIsValid(_ key: String) -> Bool {
        !key.hasPrefix("_")
    }

    /// Filter that will accept all values.
    ///
    /// ### See Also
    /// - <doc:module-filters>
    static public let filterAll: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        true
    }

    /// Filter that will accept all structure (`.array` and `.dictionary`) and ``ModuleValue/module(_:)``.
    ///
    /// ### See Also
    /// - <doc:module-filters>
    static public let filterValidChild: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .array, .dictionary: true
        case .value(.module): true
        default: false
        }
    }

    /// Filter that will accept all structure (`.array` and `.dictionary`) and ``ModuleValue/parameters(_:)``
    /// or ``ModuleValue/module(_:)`` allowing recursion into sub-Modules (layers).
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``filterLocalParameters``
    /// - ``filterTrainableParameters``
    static public let filterValidParameters: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .array, .dictionary: parameterIsValid(key)
        case .value(.parameters), .value(.module): parameterIsValid(key)
        default: false
        }
    }

    /// Filter that will accept all structure (`.array` and `.dictionary`) and ``ModuleValue/parameters(_:)``
    /// without allowing recursion into sub-Modules (layers).
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``filterValidParameters``
    /// - ``filterTrainableParameters``
    static public let filterLocalParameters: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .array, .dictionary: parameterIsValid(key)
        case .value(.parameters): parameterIsValid(key)
        default: false
        }
    }

    /// Filter that will accept all structure (`.array` and `.dictionary`) and ``ModuleValue/parameters(_:)``
    /// or ``ModuleValue/module(_:)`` that are not in the `noGrad` set.
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``freeze(recursive:keys:strict:)``
    /// - ``filterValidParameters``
    /// - ``filterLocalParameters``
    static public let filterTrainableParameters: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .array, .dictionary, .value(.parameters), .value(.module):
            parameterIsValid(key) && !module.noGrad().contains(key)
        default: false
        }
    }

    /// Filter that will accept all structure (`.array` and `.dictionary`) and ``ModuleValue/other(_:)``.
    ///
    /// ### See Also
    /// - <doc:module-filters>
    static public let filterOther: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .value(.other): true
        default: false
        }
    }

    /// Function that will turn a `(MLXArray) -> Result?` into a function suitable for use in ``filterMap(filter:map:isLeaf:)``.
    ///
    /// ```swift
    /// // map is (ModuleItem) -> [Int]?
    /// let map = Module.mapParameters { array in
    ///     array.shape
    /// }
    ///
    /// // shapes is NestedDictionary<String, [Int]>
    /// let shapes = module.filterMap(filter: Module.filterValidParameters, map: map)
    /// ```
    ///
    /// This is also trivially done with ``mapParameters(map:isLeaf:)``, which uses this function internally:
    ///
    /// ```swift
    /// let shapes = module.mapParameters { array in
    ///     array.shape
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``ModuleValue/parameters(_:)``
    /// - ``mapParameters(map:isLeaf:)``
    /// - ``filterMap(filter:map:isLeaf:)``
    static public func mapParameters<Result>(map: @escaping (MLXArray) -> Result? = { $0 }) -> (
        ModuleItem
    ) -> Result? {
        func apply(item: ModuleItem) -> Result? {
            switch item {
            case .value(.parameters(let v)): map(v)
            default: nil
            }
        }

        return apply
    }

    /// Function that will turn a `(Module) -> Result?` into a function suitable for use in ``filterMap(filter:map:isLeaf:)``.
    ///
    /// For example:
    ///
    /// ```swift
    /// // typeNames is NestedDictionary<String, String>
    /// let typeNames = module.mapParameters { module in
    ///     String(describing: type(of: module))
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``ModuleValue/module(_:)``
    /// - ``filterMap(filter:map:isLeaf:)``
    static public func mapModule<Result>(map: @escaping (Module) -> Result? = { $0 }) -> (
        ModuleItem
    ) -> Result? {
        func apply(item: ModuleItem) -> Result? {
            switch item {
            case .value(.module(let v)): map(v)
            default: nil
            }
        }

        return apply
    }

    /// Function that will turn a `(Any) -> Result?` into a function suitable for use in ``filterMap(filter:map:isLeaf:)``.
    ///
    /// For example:
    ///
    /// ```swift
    /// // floatArguments is NestedDictionary<String, Float>
    /// let floatArguments = module.mapParameters { other in
    ///     other as? Float
    /// }
    /// ```
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``ModuleValue/other(_:)``
    /// - ``filterMap(filter:map:isLeaf:)``
    static public func mapOther<Result>(map: @escaping (Any) -> Result? = { $0 }) -> (ModuleItem) ->
        Result?
    {
        func apply(item: ModuleItem) -> Result? {
            switch item {
            case .value(.other(let v)): map(v)
            default: nil
            }
        }

        return apply
    }

    /// Default leaf filter -- treat ``ModuleValue/parameters(_:)`` and ``ModuleValue/other(_:)`` as leaves.
    ///
    /// This will allow recursion into `.array`, `.dictionary` and ``ModuleValue/module(_:)``.
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``filterMap(filter:map:isLeaf:)``
    static public let isLeafDefault: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .array, .dictionary, .none, .value(.module): false
        case .value(.parameters), .value(.other), .value(.none): true
        }
    }

    /// Leaf filter that will stop at ``ModuleValue/module(_:)`` and recurse into all other structure.
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``filterMap(filter:map:isLeaf:)``
    static public let isLeafModule: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .array, .dictionary, .none: false
        case .value(.module): true
        case .value: false
        }
    }

    /// Leaf filter that will stop at ``ModuleValue/module(_:)`` if they have no child modules and recurse into all other structure.
    ///
    /// ### See Also
    /// - <doc:module-filters>
    /// - ``filterMap(filter:map:isLeaf:)``
    static public let isLeafModuleNoChildren: @Sendable (Module, String, ModuleItem) -> Bool = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .array, .dictionary, .none: false
        case .value(.module(let m)): m.children().isEmpty
        case .value: false
        }
    }

}

// MARK: - items() support

/// A single value from ``Module``.
///
/// This is typically produced from ``Module/items()`` or indirectly
/// via ``Module/filterMap(filter:map:isLeaf:)``.
///
/// ### See Also
/// - ``Module/items()``
/// - ``Module/filterMap(filter:map:isLeaf:)``
/// - ``ModuleItems``
/// - ``ModuleItem``
public enum ModuleValue {
    case none

    /// An `MLXArray` parameters value.
    ///
    /// From code:
    ///
    /// ```swift
    /// let weights: MLXArray
    /// ```
    case parameters(MLXArray)

    /// A module value.
    ///
    /// From code:
    ///
    /// ```swift
    /// let layerNorm: RMSNorm
    /// ```
    case module(Module)

    /// A non-MLXArray and non-Module value.
    ///
    /// From code:
    ///
    /// ```swift
    /// let eps: Float
    /// ```
    case other(Any)

    /// Recursive build of ``ModuleValue`` from a value.
    private static func build(value: Any?) -> ModuleItem {
        guard let value else {
            return .none
        }

        switch value {
        case let v as MLXArray:
            return .value(.parameters(v))
        case let v as Module:
            return .value(.module(v))
        case let v as Module? where v == nil:
            return .none
        case let v as [Any]:
            return .array(v.map { build(value: $0) })

        // handle tuples like arrays.  sadly we can't use parameter packs here
        // so we have to enumerate the allowed tuples (up to 5 for now)
        case let v as (Any, Any):
            return .array([build(value: v.0), build(value: v.1)])
        case let v as (Any, Any, Any):
            return .array([build(value: v.0), build(value: v.1), build(value: v.2)])
        case let v as (Any, Any, Any, Any):
            return .array([
                build(value: v.0), build(value: v.1), build(value: v.2), build(value: v.3),
            ])
        case let v as (Any, Any, Any, Any, Any):
            return .array([
                build(value: v.0), build(value: v.1), build(value: v.2), build(value: v.3),
                build(value: v.4),
            ])

        case let v as [String: Any]:
            return .dictionary(
                Dictionary(uniqueKeysWithValues: v.map { ($0.key, build(value: $0.value)) }))
        default:
            return .value(.other(value))
        }
    }

    /// Return `(String, ModuleItem)` tuple or nil if the label cannot be determined.
    ///
    /// Called from ``Module/items()``
    public static func fromMirror(_ c: Mirror.Child) -> (String, ModuleItem)? {
        var label = c.label
        var value = c.value

        // handle @PropertyInfo, @ModuleInfo.  Fall back to the
        // normal label, dropping the leading _.  E.g. @PropertyInfo var items
        // has a label of `_items`
        if let (nl, nv) = unwrapProperty(value) {
            label = nl ?? label?.dropFirst().description
            value = nv ?? (Optional<MLXArray>.none as Any)
        }
        if let (nl, nv) = unwrapModule(value) {
            label = nl ?? label?.dropFirst().description
            value = nv
        }

        if let label {
            return (label, build(value: value))
        } else {
            return nil
        }
    }
}

/// ParameterInfo allows you to specify alternate keys for parameter propreties.
///
/// For example:
///
/// ```swift
/// class MyLayer : Module {
///
///     let weights: MLXArray
///     @ParameterInfo(key: "bias") var b: MLXArray
/// }
/// ```
///
/// will have keys `weights` and `bias`.
///
/// ### See Also
/// - <doc:custom-layers>
/// - ``ModuleInfo``
@propertyWrapper public class ParameterInfo<T> {
    var value: T?
    let key: String?

    public var wrappedValue: T {
        get {
            // note: this gives a warning but it does in fact do something
            // in the case where this is e.g. ParameterInfo<MLXArray?>
            if let value = value as? T {
                return value
            } else {
                return value!
            }
        }
        set {
            if value != nil {
                // do not allow set because the info cache on Module will not
                // see the new value
                fatalError("please call update() on the array rather than setting it")

            } else {
                // value is nil, we allow set, e.g. from Module init
                value = newValue

                if unwrapProperty(self) == nil {
                    fatalError("Unable to apply @ParameterInfo to \(T.self)")
                }
            }
        }
    }

    public init(wrappedValue defaultValue: T, key: String? = nil) {
        self.value = defaultValue
        self.key = key

        if unwrapProperty(self) == nil {
            fatalError("Unable to apply @ParameterInfo to \(T.self)")
        }
    }

    public init(key: String? = nil) {
        self.value = nil
        self.key = key

        // cannot check via unwapProperty -- see wrappedValue.set
    }
}

/// Helper protocol for writing back through ``ModuleInfo``, e.g. via ``Module/update(modules:)``
private protocol TypeErasedSetter {
    func updateModule(_ value: Any) throws
}

private protocol TypeErasedSetterProvider {
    func typeErasedSetter() -> TypeErasedSetter
}

/// ModuleInfo can provde information about child modules and act as an
/// update point for ``Module/update(modules:verify:)``.
///
/// The keys for modules and parameters are usually named after their instance variables,
/// but `feed_forward` would not be a very Swifty variable name:
///
/// ```python
/// class TransformerBlock(nn.Module):
///     def __init__(self, args: ModelArgs):
///         super().__init__()
///         ...
///         self.feed_forward = FeedForward(args=args)
/// ```
///
/// Instead we can use ``ModuleInfo`` to supply a replacement key that matches the python version:
///
/// ```swift
/// public class TransformerBlock : Module {
///
///     let attention: Attention
///
///     @ModuleInfo(key: "feed_forward") var feedForward: FeedForward
///     @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
///     @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
///
///     public init(_ args: Configuration) {
///         self.attention = Attention(args)
///         self._feedForward.wrappedValue = FeedForward(args)
///         self._attentionNorm.wrappedValue = RMSNorm(args.dimensions, eps: args.normEps)
///         self._ffnNorm.wrappedValue = RMSNorm(args.dimensions, eps: args.normEps)
///     }
/// ```
///
/// All ``Linear`` modules should use a ``ModuleInfo`` so that
/// ``QuantizedLinear/quantize(model:groupSize:bits:predicate:)`` can replace them at runtime:
///
/// ```swift
/// public class FeedForward : Module {
///
///     @ModuleInfo var w1: Linear
///     @ModuleInfo var w2: Linear
///     @ModuleInfo var w3: Linear
///
///     public init(_ args: Configuration) {
///         self.w1 = Linear(args.dimensions, args.hiddenDimensions, bias: false)
///         self.w2 = Linear(args.hiddenDimensions, args.dimensions, bias: false)
///         self.w3 = Linear(args.dimensions, args.hiddenDimensions, bias: false)
///     }
/// ```
///
/// The `ModuleInfo` provides a hook for ``QuantizedLinear`` and ``Module/update(modules:verify:)`` /// to
/// replace the contents of `w1`, etc. with a new compatible `Model` after it is created.
///
/// ### See Also
/// - <doc:custom-layers>
/// - ``ParameterInfo``
@propertyWrapper public class ModuleInfo<T>: TypeErasedSetterProvider {
    var module: T?
    let key: String?

    public var wrappedValue: T {
        get {
            // note: this gives a warning but it does in fact do something
            // in the case where this is e.g. ModuleInfo<Linear?>
            if let module = module as? T {
                return module
            } else {
                return module!
            }
        }
        set {
            if module != nil {
                // do not allow set because the info cache on Module will not
                // see the new value
                fatalError(
                    "please use Model.update(modules:) rather than "
                        + "mutating the Module property directly")
            } else {
                module = newValue
            }
        }
    }

    public init(wrappedValue defaultValue: T, key: String? = nil) {
        self.module = defaultValue
        self.key = key

        if unwrapModule(self) == nil {
            fatalError("Unable to apply @ModuleInfo to \(T.self)")
        }
    }

    public init(key: String? = nil) {
        self.module = nil
        self.key = key

        if unwrapModule(self) == nil {
            fatalError("Unable to apply @ModuleInfo to \(T.self)")
        }
    }

    struct Setter: TypeErasedSetter {
        unowned var info: ModuleInfo<T>

        func updateModule(_ value: Any) throws {
            if let value = value as? T {
                info.module = value
            } else if let value = value as? [Module] {
                // try to recast as a tuple, e.g.
                // @ModuleInfo var mlp: (Linear, GELU, Linear)

                if value.count == 2, let values = (value[0], value[1]) as? T {
                    info.module = values
                } else if value.count == 3, let values = (value[0], value[1], value[2]) as? T {
                    info.module = values
                } else if value.count == 4,
                    let values = (value[0], value[1], value[2], value[3]) as? T
                {
                    info.module = values
                } else if value.count == 5,
                    let values = (value[0], value[1], value[2], value[4], value[5]) as? T
                {
                    info.module = values
                } else {
                    throw UpdateError.unableToCast(String(describing: T.self))
                }
            } else {
                throw UpdateError.unableToCast(String(describing: T.self))
            }
        }
    }

    fileprivate func typeErasedSetter() -> TypeErasedSetter {
        Setter(info: self)
    }
}

public enum UpdateError: Error {
    case unableToCollectModulesFromContainer(path: [String], modules: [String])
    case mismatchedContainers(base: String, key: String)
    case mismatchedSize(path: [String], modules: [String], expectedShape: [Int], actualShape: [Int])
    case keyNotFound(path: [String], modules: [String])
    case needModuleInfo(String)
    case unableToSet(String)
    case unableToCast(String)
    case unhandledKeys(path: [String], modules: [String], keys: [String])
    case settingArrayWithModule(path: [String], modules: [String])
    case incompatibleItems(path: [String], modules: [String], item: String, value: String)
    case unexpectedStructure(key: String, item: String)
}

extension UpdateError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unableToCollectModulesFromContainer(let path, let modules):
            return
                "Unable to collect modules from container: \(path.joined(separator: ".")) in \(modules.joined(separator: "."))"
        case .mismatchedContainers(let base, let key):
            return "Mismatched containers: \(base) \(key)"
        case .mismatchedSize(
            let
                path, let modules, let expectedShape, let actualShape):
            return
                "Mismatched parameter \(path.joined(separator: ".")) in \(modules.joined(separator: ".")) shape. Actual \(actualShape), expected \(expectedShape)"
        case .keyNotFound(let path, let modules):
            return
                "Key \(path.joined(separator: ".")) not found in \(modules.joined(separator: "."))"
        case .needModuleInfo(let string):
            return string
        case .unableToSet(let string):
            return string
        case .unableToCast:
            return "Unable to cast value"
        case .unhandledKeys(let path, let modules, let keys):
            return
                "Unhandled keys \(keys) in \(path.joined(separator: ".")) in \(modules.joined(separator: "."))"
        case .settingArrayWithModule(let path, let modules):
            return
                "Unable to set \(path.joined(separator: ".")) on \(modules.joined(separator: ".")): parameters (MLXArray) cannot be updated with a Module"
        case .incompatibleItems(let path, let modules, let item, let value):
            return
                "Unable to set \(path.joined(separator: ".")) on \(modules.joined(separator: ".")): \(item) not compatible with \(value)"
        case .unexpectedStructure(let key, let item):
            return "Unexpected structure for \(key) on \(item): not @ModuleInfo var modules = [...]"

        }
    }
}

// MARK: - Private Functions

private func unwrapProperty(_ property: Any) -> (String?, Any?)? {
    let label: String?
    let value: Any?

    switch property {
    case let p as ParameterInfo<MLXArray>:
        label = p.key
        value = p.value!
    case let p as ParameterInfo<MLXArray?>:
        label = p.key
        value = p.value as Any?
    case let p as ParameterInfo<(MLXArray, MLXArray)>:
        label = p.key
        value = p.value!
    case let p as ParameterInfo<(MLXArray, MLXArray, MLXArray)>:
        label = p.key
        value = p.value!
    case let p as ParameterInfo<[MLXArray]>:
        label = p.key
        value = p.value!
    case let p as ParameterInfo<[String: MLXArray]>:
        label = p.key
        value = p.value!
    default:
        return nil
    }

    return (label, value)
}

private func isModuleInfo(_ property: Any) -> (String?, Any, TypeErasedSetter)? {
    let m = Mirror(reflecting: property)
    let c = m.children.map { $0 }

    if c.count == 2 && c[0].label == "module" && c[1].label == "key" {
        if let property = property as? TypeErasedSetterProvider {
            let setter = property.typeErasedSetter()
            return (c[1].value as? String, c[0].value, setter)
        }
    }
    return nil
}

private func unwrapModule(_ property: Any) -> (String?, Any)? {
    if let (key, value, _) = isModuleInfo(property) {
        return (key, value)
    }

    return nil
}

private enum MirrorAction {
    /// stop iterating
    case stop

    /// continue iterating
    case next
}

/// Mirror wrapper that traverses types up to `Module` and visits their children.
private func mirrorUpToModule(module: Module, visit: (Mirror.Child) throws -> MirrorAction) rethrows
{
    var m = Mirror(reflecting: module)
    repeat {
        for c in m.children {
            switch try visit(c) {
            case .stop: return
            case .next: break
            }
        }

        if let s = m.superclassMirror {
            m = s
        } else {
            break
        }
    } while m.subjectType != Module.self
}

/// convenience for describing the type of a value
private func describeType<T>(_ value: T) -> String {
    String(describing: type(of: value))
}
