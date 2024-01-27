import Foundation
import MLX

public typealias ModuleParameters = NestedDictionary<String, MLXArray>

open class Module {

    var training = false
    var noGrad = Set<String>()

    public init() {
    }

    public func items() -> [String: ModuleItem] {
        var result = [String: ModuleItem]()

        mirrorUpToModule(module: self) { c in
            if let (key, value) = ModuleItem.fromMirror(c) {
                result[key] = value
            }
            return .next
        }
        return result
    }

    public func describeExtra(_ indent: Int) -> String {
        let other = filterMap(filter: Self.filterOther, map: Self.mapOther)
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

    public func filterMap<Result>(
        filter: (Module, String, ModuleItem) -> Bool,
        map: (ModuleItem) -> Result? = { $0 },
        isLeaf: (Module, String, ModuleItem) -> Bool = Module.isLeafDefault
    ) -> NestedDictionary<String, Result> {
        var result = NestedDictionary<String, Result>()

        // recursive unwrap rules
        func unwrap(_ vk: String, _ v: ModuleItem) -> NestedItem<String, Result> {
            if isLeaf(self, vk, v) {
                if let v = map(v) {
                    return .value(v)
                } else {
                    return .none
                }
            }

            switch v {
            case .module(let module):
                return module.filterMap(filter: filter, map: map, isLeaf: isLeaf).asItem()

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

    public func mapParameters<Result>(
        map: (MLXArray) -> Result? = { $0 },
        isLeaf: (Module, String, ModuleItem) -> Bool = Module.isLeafDefault
    ) -> NestedDictionary<String, Result> {
        filterMap(
            filter: Self.filterValidParameters,
            map: { (item: ModuleItem) in
                switch item {
                case .parameters(let v): map(v)
                default: nil
                }
            }, isLeaf: isLeaf)
    }

    public func parameters() -> ModuleParameters {
        filterMap(filter: Self.filterValidParameters, map: Self.mapParameters)
    }

    public func trainableParameters() -> ModuleParameters {
        filterMap(filter: Self.filterTrainableParameters, map: Self.mapParameters)
    }

    public func children() -> NestedDictionary<String, Module> {
        filterMap(filter: Self.filterValidChild, map: Self.mapModule, isLeaf: Self.isLeafModule)
    }

    public func leafModules() -> NestedDictionary<String, Module> {
        filterMap(
            filter: Self.filterValidChild, map: Self.mapModule, isLeaf: Self.isLeafModuleNoChildren)
    }

    public struct Verify: OptionSet {
        public init(rawValue: Int) {
            self.rawValue = rawValue
        }

        public let rawValue: Int

        static public let noUnusedKeys = Verify(rawValue: 1 << 0)

        static public let all = Verify(rawValue: -1)
        static public let none = Verify([])
    }

    public func update(parameters: ModuleParameters) {
        try! update(parameters: parameters, verify: .none)
    }

    public func update(parameters: ModuleParameters, verify: Verify) throws {

        func apply(key: String, _ item: ModuleItem, _ value: NestedItem<String, MLXArray>) throws {
            if case .none = value {
                return
            }

            // item: representation of the module properties
            // value: nested dictionary structure
            //
            // match them up and apply the MLXArrays from value -> item

            switch (item, value) {
            case (.parameters(let p), .value(let newArray)):
                p.update(newArray)

            case (.array(let array), .array(let values)):
                for (i, (arrayItem, valueItem)) in zip(array, values).enumerated() {
                    try apply(key: "\(key).\(i)", arrayItem, valueItem)
                }

            case (.dictionary(let dictionary), .dictionary(let values)):
                for (valueKey, valueItem) in values {
                    if let dictionaryItem = dictionary[key] {
                        try apply(key: "\(key).\(valueKey)", dictionaryItem, valueItem)
                    }
                }

            case (.module(let module), .dictionary(let values)):
                try module.update(parameters: NestedDictionary(values: values), verify: verify)

            default:
                fatalError("Unable to set \(key) on \(self): \(item) not compatible with \(value)")
            }
        }

        var processed = Set(parameters.keys)
        for (key, item) in items() {
            if let value = parameters[key] {
                processed.remove(key)
                try apply(key: key, item, value)
            }
        }

        if verify.contains(.noUnusedKeys) && !processed.isEmpty {
            throw UpdateError.unhandledKeys(
                base: String(describing: type(of: self)), keys: processed.sorted())
        }
    }

    public func apply(
        filter: (Module, String, ModuleItem) -> Bool = Module.filterValidParameters,
        map: (MLXArray) -> MLXArray
    ) {
        update(parameters: mapParameters(map: map))
    }

    public func update(modules: NestedDictionary<String, Module>) {
        try! update(modules: modules, verify: .none)
    }

    public func update(modules: NestedDictionary<String, Module>, verify: Verify) throws {

        func apply(key: String, _ item: ModuleItem, _ value: NestedItem<String, Module>) throws {
            if case .none = value {
                return
            }

            // item: representation of the module properties
            // value: nested dictionary structure
            //
            // match them up and apply the MLXArrays from value -> item

            switch (item, value) {
            case (.parameters, .value):
                fatalError(
                    "Unable to set \(key) on \(self): parameters (MLXArray) cannot be updated with a Module"
                )

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
                    for item in values {
                        switch item {
                        case .value(let module):
                            newModules.append(module)
                        default:
                            throw UpdateError.unableToCollectModulesFromContainer(
                                base: String(describing: type(of: self)), key: key)
                        }
                    }

                    try MLXNN.update(module: self, key: key, newModules)

                case .dictionary:
                    // recurse
                    for (i, v) in zip(items, values) {
                        switch (i, v) {
                        case (.module(let m), .dictionary(let d)):
                            try m.update(modules: NestedDictionary(values: d), verify: verify)

                        default:
                            throw UpdateError.mismatchedContainers(
                                base: String(describing: type(of: self)), key: key)
                        }
                    }
                default:
                    fatalError(
                        "Unexpected structure for \(key) on \(self): not @ModuleInfo var modules = [...]"
                    )
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
                            base: String(describing: type(of: self)), key: key)
                    }
                }

                try MLXNN.update(module: self, key: key, newModules)

            case (.module, .value(let newModule)):
                try MLXNN.update(module: self, key: key, newModule)

            case (.module(let module), .dictionary(let values)):
                try module.update(modules: NestedDictionary(values: values), verify: verify)

            default:
                fatalError("Unable to set \(key) on \(self): \(item) not compatible with \(value)")
            }
        }

        var processed = Set(modules.keys)
        for (key, item) in items() {
            if let value = modules[key] {
                processed.remove(key)
                try apply(key: key, item, value)
            }
        }

        if verify.contains(.noUnusedKeys) && !processed.isEmpty {
            throw UpdateError.unhandledKeys(
                base: String(describing: type(of: self)), keys: processed.sorted())
        }
    }

    public func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) {
        // TODO: implement
    }

    public func unfreeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) {
        // TODO: implement
    }
}

extension Module: IndentedDescription {

    public func description(indent: Int) -> String {
        print("\(type(of: self)) \(indent)")
        var result = ""

        result += "\(String(describing: type(of: self)))\(describeExtra(indent))"

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

public protocol UnaryModel {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

// MARK: - Filters and Maps

extension Module {

    static public let filterAll = { (module: Module, key: String, item: ModuleItem) in
        true
    }

    static public let filterValidChild = { (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .array, .dictionary, .module: true
        default: false
        }
    }

    static public let filterValidParameters = { (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .parameters, .array, .dictionary, .module: !key.hasPrefix("_")
        default: false
        }
    }

    static public let filterTrainableParameters = {
        (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .parameters, .array, .dictionary, .module:
            !key.hasPrefix("_") && !module.noGrad.contains(key)
        default: false
        }
    }

    static public let filterOther = { (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .other: true
        default: false
        }
    }

    static public let mapParameters = { (item: ModuleItem) -> MLXArray? in
        switch item {
        case .parameters(let v): return v
        default: return nil
        }
    }

    static public let mapModule = { (item: ModuleItem) -> Module? in
        switch item {
        case .module(let v): return v
        default: return nil
        }
    }

    static public let mapOther = { (item: ModuleItem) -> Any? in
        switch item {
        case .other(let v): return v
        default: return nil
        }
    }

    static public let isLeafDefault = { (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .parameters, .other: true
        case .array, .dictionary, .module: false
        }
    }

    static public let isLeafModule = { (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .module: true
        case .parameters, .array, .dictionary, .other: false
        }
    }

    static public let isLeafModuleNoChildren = { (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .module(let m): m.children().isEmpty
        case .parameters, .array, .dictionary, .other: false
        }
    }

}

// MARK: - items() support

public enum ModuleItem {
    case parameters(MLXArray)
    case array([ModuleItem])
    case dictionary([AnyHashable: ModuleItem])
    case module(Module)
    case other(Any)

    public init(_ value: Any) {
        switch value {
        case let v as MLXArray:
            self = .parameters(v)
        case let v as Module:
            self = .module(v)
        case let v as [Any]:
            self = .array(v.map { ModuleItem($0) })
        case let v as [AnyHashable: Any]:
            let d = Dictionary(uniqueKeysWithValues: v.map { ($0.key, ModuleItem($0.value)) })
            self = .dictionary(d)
        default:
            self = .other(value)
        }
    }

    public static func fromMirror(_ c: Mirror.Child) -> (String, ModuleItem)? {
        var label = c.label
        var value = c.value

        // handle @PropertyInfo, @ModuleInfo.  Fall back to the
        // normal label, dropping the leading _.  E.g. @PropertyInfo var items
        // has a label of `_items`
        if let (nl, nv) = unwrapProperty(value) {
            label = nl ?? label?.dropFirst().description
            value = nv
        }
        if let (nl, nv) = unwrapModule(value) {
            label = nl ?? label?.dropFirst().description
            value = nv
        }

        if let label {
            return (label, ModuleItem(value))
        } else {
            return nil
        }
    }
}

@propertyWrapper public class ParameterInfo<T> {
    var value: T?
    let key: String?

    public var wrappedValue: T {
        get {
            value!
        }
        set {
            value = newValue
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

        if unwrapProperty(self) == nil {
            fatalError("Unable to apply @ParameterInfo to \(T.self)")
        }
    }
}

private protocol TypeErasedSetter {
    func updateModule(_ value: Any) throws
}

@propertyWrapper public class ModuleInfo<T>: TypeErasedSetter {
    var module: T?
    let key: String?

    public var wrappedValue: T {
        get {
            module!
        }
        set {
            module = newValue
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

    func updateModule(_ value: Any) throws {
        if let value = value as? T {
            self.wrappedValue = value
        } else {
            throw UpdateError.unableToCast
        }
    }
}

// MARK: - Private Functions

enum UpdateError: Error {
    case unableToCollectModulesFromContainer(base: String, key: String)
    case mismatchedContainers(base: String, key: String)
    case keyNotFound(base: String, key: String)
    case needModuleInfo(String)
    case unableToSet(String)
    case unableToCast
    case unhandledKeys(base: String, keys: [String])
}

private func matches(key: String, _ c: Mirror.Child) -> Bool {
    if key == c.label {
        return true
    }

    if let (label, _) = unwrapProperty(c.value) {
        // match vs the configured key or the label without the leading _
        return key == label ?? String((c.label ?? "").dropFirst())
    }
    if let (label, _) = unwrapModule(c.value) {
        return key == label ?? String((c.label ?? "").dropFirst())
    }

    return false
}

private func update(module: Module, key: String, _ value: Any) throws {
    var found = false

    try mirrorUpToModule(module: module) { c in
        if matches(key: key, c) {
            if let (_, _, setter) = isModuleInfo(c.value) {
                do {
                    try setter.updateModule(value)
                    found = true
                    return .stop
                } catch {
                    throw UpdateError.needModuleInfo(
                        "Unable to set modules for \(String(describing: type(of: module))).\(key) -- maybe type mismatch: \(String(describing: type(of: value)))"
                    )
                }
            } else {
                throw UpdateError.needModuleInfo(
                    "Unable to get @ModuleInfo for \(String(describing: type(of: module))).\(key) -- must be wrapped to receive updates"
                )
            }
        }
        return .next
    }
    if !found {
        throw UpdateError.keyNotFound(base: String(describing: type(of: module)), key: key)
    }
}

private func unwrapProperty(_ property: Any) -> (String?, Any)? {
    let label: String?
    let value: Any

    switch property {
    case let p as ParameterInfo<MLXArray>:
        label = p.key
        value = p.value!
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
        if let setter = property as? TypeErasedSetter {
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
    case stop
    case next
}

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
