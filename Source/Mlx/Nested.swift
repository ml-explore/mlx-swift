import Foundation

public indirect enum NestedItem<Key: Hashable, Element>: CustomStringConvertible {
    case none
    case value(Element)
    case array([NestedItem<Key, Element>])
    case dictionary([Key: NestedItem<Key, Element>])

    public func unwrap() -> Any? {
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

    public func mapValues<Result>(_ transform: (Element) throws -> Result) rethrows -> NestedItem<
        Key, Result
    > {
        switch self {
        case .none:
            return .none
        case .value(let element):
            return try .value(transform(element))
        case .array(let array):
            return try .array(array.map { try $0.mapValues(transform) })
        case .dictionary(let dictionary):
            return try .dictionary(dictionary.mapValues { try $0.mapValues(transform) })
        }
    }

    public func compactMapValues<Result>(_ transform: (Element) throws -> Result?) rethrows
        -> NestedItem<Key, Result>
    {
        switch self {
        case .none:
            return .none
        case .value(let element):
            if let value = try transform(element) {
                return .value(value)
            } else {
                return .none
            }
        case .array(let array):
            let newArray =
                try array
                .map { try $0.compactMapValues(transform) }
                .compactMap { v -> NestedItem<Key, Result>? in
                    switch v {
                    case .none: nil
                    default: v
                    }
                }
            if newArray.isEmpty {
                return .none
            } else {
                return .array(newArray)
            }
        case .dictionary(let dictionary):
            let newDictionary =
                try dictionary
                .mapValues { try $0.compactMapValues(transform) }
                .compactMap { v -> (Key, NestedItem<Key, Result>)? in
                    switch v.value {
                    case .none: nil
                    default: v
                    }
                }
            if newDictionary.isEmpty {
                return .none
            } else {
                return .dictionary(Dictionary(uniqueKeysWithValues: newDictionary))
            }
        }
    }

    public func flatten(prefix: String? = nil) -> [(String, Element)] {
        func newPrefix(_ i: CustomStringConvertible) -> String {
            if let prefix {
                return "\(prefix).\(i)"
            } else {
                return i.description
            }
        }
        switch self {
        case .none:
            return []
        case .value(let element):
            return [(prefix ?? "", element)]
        case .array(let array):
            return array.enumerated().flatMap { (index, value) in
                value.flatten(prefix: newPrefix(index))
            }
        case .dictionary(let dictionary):
            return dictionary.flatMap { (key, value) in
                value.flatten(prefix: newPrefix(String(describing: key)))
            }
        }
    }

    public static func unflatten(_ tree: [(Key, Element)]) -> NestedItem<Key, Element>
    where Key == String {
        if tree.isEmpty {
            return .dictionary([:])
        }

        return unflattenRecurse(tree)
    }

    private enum UnflattenKind {
        case list
        case dictionary

        static func detect(key: String) -> UnflattenKind {
            let first = key.prefix { $0 != "." }
            if Int(first) != nil {
                return .list
            } else {
                return .dictionary
            }
        }
    }

    private static func unflattenRecurse(_ tree: [(String, Element)]) -> NestedItem<String, Element>
    {
        if tree.count == 1 && tree[0].0 == "" {
            return .value(tree[0].1)
        }

        var children = [String: [(String, Element)]]()
        for (key, value) in tree {
            let pieces = key.split(separator: ".", maxSplits: 1)
            let current = pieces[0]
            let next = pieces.count > 1 ? pieces[1] : ""
            children[String(current), default: []].append((String(next), value))
        }

        switch UnflattenKind.detect(key: tree[0].0) {
        case .list:
            if children.isEmpty {
                return .array([])
            }

            let items =
                children
                .map { (Int($0.key)!, $0.value) }
            let maxIndex = items.lazy.map { $0.0 }.max()!

            var result = [NestedItem<String, Element>](repeating: .none, count: maxIndex + 1)
            for (index, value) in items {
                result[index] = unflattenRecurse(value)
            }
            return .array(result)
        case .dictionary:
            var result = [String: NestedItem<String, Element>]()
            for (key, value) in children {
                result[key] = unflattenRecurse(value)
            }
            return .dictionary(result)
        }
    }

    public func description(_ indent: Int) -> String {
        let indentString = String(repeating: " ", count: indent)

        let d =
            switch self {
            case .none:
                "none"

            case .value(let element):
                String(describing: element)

            case .array(let array):
                "[\n" + indentString + "  "
                    + array.map { $0.description(indent + 2) }.joined(
                        separator: ",\n\(indentString)  ") + "\n" + indentString + "]"

            case .dictionary(let dictionary):
                "[\n" + indentString + "  "
                    + dictionary
                    .sorted { lhs, rhs in String(describing: lhs.key) < String(describing: rhs.key)
                    }
                    .map { "\($0.key): \($0.value.description(indent + 2))" }.joined(
                        separator: ",\n\(indentString)  ") + "\n" + indentString + "]"
            }

        return d
    }

    public var description: String {
        description(0)
    }
}

extension NestedItem: Equatable where Element: Equatable {
}

public struct NestedDictionary<Key: Hashable, Element>: CustomStringConvertible {
    var contents = [Key: NestedItem<Key, Element>]()

    public init() {
    }

    public init(values: [Key: NestedItem<Key, Element>]) {
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

    public func asDictionary() -> [Key: Any] {
        asItem().unwrap() as! [Key: Any]
    }

    public var description: String {
        asItem().description
    }

    public func mapValues<Result>(transform: (Element) throws -> Result) rethrows
        -> NestedDictionary<Key, Result>
    {
        switch try NestedItem.dictionary(contents).mapValues(transform) {
        case .dictionary(let values):
            return NestedDictionary<Key, Result>(values: values)
        default:
            fatalError()
        }
    }

    public func compactMapValues<Result>(transform: (Element) throws -> Result?) rethrows
        -> NestedDictionary<Key, Result>
    {
        switch try NestedItem.dictionary(contents).compactMapValues(transform) {
        case .dictionary(let values):
            return NestedDictionary<Key, Result>(values: values)
        default:
            fatalError()
        }
    }

    public func flatten() -> [(String, Element)] {
        contents.flatMap { key, value in value.flatten(prefix: String(describing: key)) }
    }

    static public func unflatten(_ flat: [(Key, Element)]) -> NestedDictionary<String, Element>
    where Key == String {
        switch NestedItem.unflatten(flat) {
        case .dictionary(let values):
            return NestedDictionary(values: values)
        default:
            fatalError()
        }
    }
}

extension NestedDictionary: Equatable where Element: Equatable {
}

extension NestedDictionary: Collection {

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
