// Copyright © 2024 Apple Inc.

import Foundation

/// Protocol for types that can provide an indented description, e.g. nested types.
public protocol IndentedDescription: CustomStringConvertible {
    
    /// Return the `description` with the given indent level.
    ///
    /// This should apply successively nested indents to any children that can also be indented.
    /// See ``indentedDescription(_:_:)`` for an easy way to accomplish this.
    func description(indent: Int) -> String
}

extension IndentedDescription {
    public var description: String {
        description(indent: 0)
    }
}

/// Return `description` or ``IndentedDescription/description(indent:)`` if possible.
public func indentedDescription(_ value: Any, _ indent: Int) -> String {
    if let value = value as? IndentedDescription {
        return value.description(indent: indent)
    } else {
        return String(describing: value)
    }
}

/// Backing storage for ``NestedDictionary``.
///
/// `NestedItem` holds the array / dictionary / value structure contained in a `NestedDictionary`.
/// Typically creation of these values is handled by code that returns `NestedDictionary` but
/// these are used when traversing the structure manually.
///
/// ### See Also
/// - ``NestedDictionary``
/// - ``NestedDictionary/mapValues(transform:)``
/// - ``NestedDictionary/flattened(prefix:)``
/// - ``NestedDictionary/unflattened(_:)-4p8bn``
public indirect enum NestedItem<Key: Hashable, Element>: IndentedDescription {
    case none
    case value(Element)
    case array([NestedItem<Key, Element>])
    case dictionary([Key: NestedItem<Key, Element>])

    /// Return the values contained inside as a type erased swift structure, e.g. normal `Array` and `Dictionary`.
    /// This is suitable for tests and debugging but should not be used in typical code.
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

    /// Transform the values in the nested structure using the `transform()` function.
    ///
    /// This is typically called via ``NestedDictionary/mapValues(transform:)``.
    ///
    /// ### See Also
    /// - ``NestedDictionary/mapValues(transform:)``
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

    /// Transform the values in the nested structure using the `transform()` function.
    ///
    /// This is typically called via ``NestedDictionary/compactMapValues(transform:)``.
    ///
    /// ### See Also
    /// - ``NestedDictionary/compactMapValues(transform:)``
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

    /// Return a flattened representation of the structured contents as an array of key/value tuples.
    ///
    /// This is typically called via ``NestedDictionary/flattened(prefix:)``.
    ///
    /// ### See Also
    /// - ``unflattened(_:)``
    /// - ``NestedDictionary/flattened(prefix:)``
    /// - ``NestedDictionary/unflattened(_:)-4p8bn``
    /// - ``NestedDictionary/unflattened(_:)-7xuiv``
    public func flattened(prefix: String? = nil) -> [(String, Element)] {
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
                value.flattened(prefix: newPrefix(index))
            }
        case .dictionary(let dictionary):
            return dictionary.flatMap { (key, value) in
                value.flattened(prefix: newPrefix(String(describing: key)))
            }
        }
    }

    /// Convert a flattened list of key/value tuples back into a `NestedItem` structure.
    ///
    /// This is typically called via ``NestedDictionary/unflattened(_:)-4p8bn``.
    ///
    /// ### See Also
    /// - ``flattened(prefix:)``
    /// - ``NestedDictionary/flattened(prefix:)``
    /// - ``NestedDictionary/unflattened(_:)-4p8bn``
    /// - ``NestedDictionary/unflattened(_:)-7xuiv``
    public static func unflattened(_ tree: [(Key, Element)]) -> NestedItem<Key, Element>
    where Key == String {
        if tree.isEmpty {
            return .dictionary([:])
        }

        return unflattenedRecurse(tree)
    }

    // see unflattenedRecurse() -- the inferred type of structure to create
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

    private static func unflattenedRecurse(_ tree: [(String, Element)]) -> NestedItem<
        String, Element
    > {
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
                result[index] = unflattenedRecurse(value)
            }
            return .array(result)
        case .dictionary:
            var result = [String: NestedItem<String, Element>]()
            for (key, value) in children {
                result[key] = unflattenedRecurse(value)
            }
            return .dictionary(result)
        }
    }

    public func description(indent: Int) -> String {
        let indentString = String(repeating: " ", count: indent)

        let d =
            switch self {
            case .none:
                "none"

            case .value(let element):
                indentedDescription(element, indent + 2)

            case .array(let array):
                "[\n" + indentString + "  "
                    + array.map {
                        indentedDescription($0, indent + 2)
                    }
                    .joined(separator: ",\n\(indentString)  ") + "\n" + indentString + "]"

            case .dictionary(let dictionary):
                "[\n" + indentString + "  "
                    + dictionary
                    .sorted { lhs, rhs in String(describing: lhs.key) < String(describing: rhs.key)
                    }
                    .map {
                        "\($0.key): \(indentedDescription($0.value, indent + 2))"
                    }.joined(separator: ",\n\(indentString)  ") + "\n" + indentString + "]"
            }

        return d
    }
}

extension NestedItem: Equatable where Element: Equatable {
}

/// Nested structure of arrays, dictionaries and values.
///
/// Some of the capabilities of `MLX` (especially in `MLXNN`) need to deal with arbitrarily structured
/// values.  This is simple in python but swift needs type to describe the structure.
///
/// For example a nested dictionary / `[Int]` strucure might look like this:
///
/// ```
/// [
///   w1: [
///     weight: [64, 20]
///   ],
///   w2: [
///     weight: [20, 64]
///   ]
/// ]
/// ```
///
/// This could be created with:
///
/// ```swift
/// let contents: NestedItem<String, [Int]> =
///     .dictionary([
///         "w1": .dictionary([
///             "weight": .value([64, 20])
///         ]),
///         "w2": .dictionary([
///             "weight": .value([20, 64])
///         ]),
///     ])
/// let nd = NestedDictionary(item: contents)
/// ```
///
/// In practice these structures are created programatically by traversing other nested structures.  The
/// above example was actually created from code like this:
///
/// ```swift
/// import MLXNN
///
/// class M : Module {
///     let w1 = Linear(64, 20)
///     let w2 = Linear(20, 64)
/// }
///
/// let nd = M().mapParameters { $0.shape }
/// ```
///
/// ### See Also
/// - ``NestedItem``
public struct NestedDictionary<Key: Hashable, Element>: CustomStringConvertible {
    var contents = [Key: NestedItem<Key, Element>]()

    /// Initialize an empty `NestedDictionary`.
    ///
    /// ### See Also
    /// - ``NestedDictionary/subscript(_:)-7bphj``
    public init() {
    }

    /// Initialize with a dictionary of ``NestedItem``.
    public init(values: [Key: NestedItem<Key, Element>]) {
        contents = values
    }

    /// Initialize with a ``NestedItem``.
    ///
    /// This must be a ``NestedItem/dictionary(_:)``.
    public init(item: NestedItem<Key, Element>) {
        switch item {
        case .dictionary(let values):
            self.contents = values
        default:
            fatalError("cannot initialize from non .dictionary item: \(item)")
        }
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

    public func asDictionary() -> [Key: Any?] {
        contents.mapValues { $0.unwrap() }
    }

    public var description: String {
        asItem().description
    }

    /// Transform the values in the nested structure using the `transform()` function.
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

    /// Transform the values in the nested structure using the `transform()` function.
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

    /// Return a flattened representation of the structured contents as an array of key/value tuples.
    ///
    /// ### See Also
    /// - ``unflattened(_:)-4p8bn``
    /// - ``unflattened(_:)-7xuiv``
    public func flattened(prefix: String? = nil) -> [(String, Element)] {
        contents.flatMap { key, value in
            let keyString = String(describing: key)
            let keyPrefix = prefix == nil ? keyString : "\(prefix!).\(keyString)"
            return value.flattened(prefix: keyPrefix)
        }
    }

    /// Convert a flattened list of key/value tuples back into a `NestedDictionary` structure.
    ///
    /// ### See Also
    /// - ``flattened(prefix:)``
    /// - ``unflattened(_:)-7xuiv``
    static public func unflattened(_ flat: [(Key, Element)]) -> NestedDictionary<String, Element>
    where Key == String {
        switch NestedItem.unflattened(flat) {
        case .dictionary(let values):
            return NestedDictionary(values: values)
        default:
            fatalError()
        }
    }

    /// Convert a flattened dictionary back into a `NestedDictionary` structure.
    ///
    /// ### See Also
    /// - ``flattened(prefix:)``
    /// - ``unflattened(_:)-4p8bn``
    static public func unflattened(_ flat: [Key: Element]) -> NestedDictionary<String, Element>
    where Key == String {
        unflattened(flat.map { $0 })
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
