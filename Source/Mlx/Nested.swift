import Foundation

public indirect enum NestedItem<Key: Hashable, Element> : CustomStringConvertible {
    case none
    case value(Element)
    case array([NestedItem<Key, Element>])
    case dictionary([Key: NestedItem<Key, Element>])
    
    func unwrap() -> Any? {
        switch self {
        case .none:
            return nil
        case .value(let element):
            return element
        case .array(let array):
            return array.map { $0.unwrap() }
        case .dictionary(let dictionary):
            return dictionary.mapValues { $0.unwrap() }
        }
    }
    
    public func description(_ indent: Int) -> String {
        let indentString = String(repeating: " ", count: indent)

        let d = switch self {
        case .none:
            "none"
            
        case .value(let element):
            String(describing: element)
            
        case .array(let array):
            "[\n" +
            indentString + "  " +
            array.map { $0.description(indent + 2) }.joined(separator: ",\n\(indentString)  ") +
            "\n" +
            indentString + "]"
            
        case .dictionary(let dictionary):
            "[\n" +
            indentString + "  " +
            dictionary
                .sorted { lhs, rhs in String(describing: lhs.key) < String(describing: rhs.key) }
                .map { "\($0.key): \($0.value.description(indent + 2))"}.joined(separator: ",\n\(indentString)  ") +
                "\n" +
                indentString + "]"
            }
        
        return d
    }
    
    public var description: String {
        description(0)
    }
}

public struct NestedDictionary<Key: Hashable, Element> : CustomStringConvertible {
    var contents = [Key:NestedItem<Key, Element>]()
    
    public init() {        
    }
    
    public init(values: [Key : NestedItem<Key, Element>]) {
        contents = values
    }
    
    public subscript(key: Key) -> NestedItem<Key, Element>? {
        get {
            contents[key]
        }
        set {
            contents[key] = newValue
        }
    }
    
    public subscript(unwrapping key: Key) -> Element? {
        get {
            if let value = contents[key] {
                switch value {
                case .value(let v): return v
                default: return nil
                }
            } else {
                return nil
            }
        }
        set {
            if let value = newValue {
                contents[key] = .value(value)
            } else {
                contents[key] = nil
            }
        }
    }
    
    public var keys: Dictionary<Key, NestedItem<Key, Element>>.Keys { contents.keys }
    public var values: Dictionary<Key, NestedItem<Key, Element>>.Values { contents.values }
    public var isEmpty: Bool { contents.isEmpty }
    public var count: Int { contents.count }

    public func asItem() -> NestedItem<Key, Element> {
        .dictionary(contents)
    }
    
    public func asDictionary() -> [Key:Any] {
        asItem().unwrap() as! [Key:Any]
    }
    
    public var description: String {
        asItem().description
    }

}

extension NestedDictionary : Collection {
        
    public typealias Index = Dictionary<Key, NestedItem<Key, Element>>.Index
    
    public var startIndex: Index { contents.startIndex }
    public var endIndex: Index { contents.endIndex }
    
    public subscript(position: Index) -> Dictionary<Key, NestedItem<Key, Element>>.Element {
        contents[position]
    }
    
    public func index(after i: Index) -> Index {
        contents.index(after: i)
    }
}
