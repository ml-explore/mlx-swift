import Foundation

// Module
// Linear
// RoPE
// RMSNorm
// treemap -- used to apply a dtype to parameters

public typealias ModuleParameters = NestedDictionary<String, MLXArray>

public enum ModuleItem {
    case parameters(MLXArray)
    case array([ModuleItem])
    case dictionary([AnyHashable:ModuleItem])
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
        case let v as [AnyHashable:Any]:
            let d = Dictionary(uniqueKeysWithValues: v.map { ($0.key, ModuleItem($0.value))})
            self = .dictionary(d)
        default:
            self = .other(value)
        }
    }
}

@propertyWrapper struct UserDefaultsBacked<Value> {
    var wrappedValue: Value {
        get {
            let value = storage.value(forKey: key) as? Value
            return value ?? defaultValue
        }
        set {
            storage.setValue(newValue, forKey: key)
        }
    }

    private let key: String
    private let defaultValue: Value
    private let storage: UserDefaults

    init(wrappedValue defaultValue: Value,
         key: String,
         storage: UserDefaults = .standard) {
        self.defaultValue = defaultValue
        self.key = key
        self.storage = storage
    }
}

@propertyWrapper public class Property<T> {
    var value: T
    let key: String?

    public var wrappedValue: T {
        get {
            value
        }
        set {
            value = newValue
        }
    }
    
    public init(wrappedValue defaultValue: T, key: String? = nil) {
        self.value = defaultValue
        self.key = key
        
        if unwrapProperty(self) == nil {
            fatalError("Unable to apply @Property to \(T.self)")
        }
    }
}

open class Module {
    
    var training = false
    var noGrad = Set<String>()
    
    public init() {
    }
    
    public func items() -> [String:ModuleItem] {
        var result = [String:ModuleItem]()
        
        let m = Mirror(reflecting: self)
        for c in m.children {
            var label = c.label
            var value = c.value
            
            if let (nl, nv) = unwrapProperty(value) {
                label = nl ?? label
                value = nv
            }
            
            if let label {
                result[label] = ModuleItem(value)
            }
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
    
}

extension Module {
    
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
                var result = [String:NestedItem<String, Result>]()
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
        filterMap(filter: Self.filterValidParameters, map: { (item: ModuleItem) in
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
    
    public func update(parameters: ModuleParameters) {
        
        func apply(key: String, _ item: ModuleItem, _ value: NestedItem<String, MLXArray>) {
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
                    apply(key: "\(key).\(i)", arrayItem, valueItem)
                }
                
            case (.dictionary(let dictionary), .dictionary(let values)):
                for (valueKey, valueItem) in values {
                    if let dictionaryItem = dictionary[key] {
                        apply(key: "\(key).\(valueKey)", dictionaryItem, valueItem)
                    }
                }
                
            case (.module(let module), .dictionary(let values)):
                module.update(parameters: NestedDictionary(values: values))

            default:
                fatalError("Unable to set \(key) on \(self): \(item) not compatible with \(value)")
            }
        }
        
        for (key, item) in items() {
            if let value = parameters[key] {
                apply(key: key, item, value)
            }
        }
    }
    
    public func apply(
        filter: (Module, String, ModuleItem) -> Bool = Module.filterValidParameters,
        map: (MLXArray) -> MLXArray
    ) {
        update(parameters: mapParameters(map: map))
    }
}

extension Module : CustomStringConvertible {
    
    public func description(_ indent: Int) -> String {
        var result = ""
        
        result += "\(String(describing: type(of: self)))\(describeExtra(indent))"
        
        let children = self.children()
        
        if !children.isEmpty {
            let kv = children.sorted { lhs, rhs in lhs.key < rhs.key }
            let indentString = String(repeating: " ", count: indent)
            
            result += " {\n"
            for (key, value) in kv {
                result += "\(indentString)  \(key): \(value.description(indent + 2)),\n"
            }
            result += "\(indentString)}"
        }
        
        return result
    }
    
    public var description: String {
        description(0)
    }
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
    
    static public let filterTrainableParameters = { (module: Module, key: String, item: ModuleItem) in
        switch item {
        case .parameters, .array, .dictionary, .module: !key.hasPrefix("_") && !module.noGrad.contains(key)
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

}

// MARK: - Private Functions

private func unwrapProperty(_ property: Any) -> (String?, Any)? {
    let label: String?
    let value: Any
    
    switch property {
    case let p as Property<MLXArray>:
        label = p.key
        value = p.value
    case let p as Property<(MLXArray, MLXArray)>:
        label = p.key
        value = p.value
    case let p as Property<(MLXArray, MLXArray, MLXArray)>:
        label = p.key
        value = p.value
    case let p as Property<[MLXArray]>:
        label = p.key
        value = p.value
    case let p as Property<[String:MLXArray]>:
        label = p.key
        value = p.value
    default:
        return nil
    }
    
    return (label, value)
}

