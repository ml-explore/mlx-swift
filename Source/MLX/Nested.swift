// Copyright © 2024 Apple Inc.

import Foundation

/// Protocol for types that can provide an indented description, e.g. nested types.
public protocol IndentedDescription: CustomStringConvertible {

    /// Return the `description` with the given indent level.
    ///
    /// This should apply successively nested indents to any children that can also be indented.
    /// See `indentedDescription(_:_:)` for an easy way to accomplish this.
    func description(indent: Int) -> String
}

extension IndentedDescription {
    public var description: String {
        description(indent: 0)
    }
}

/// Return `description` or ``IndentedDescription/description(indent:)`` if possible.
@_documentation(visibility: internal)
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
/// - ``NestedDictionary/mapValues(transform:)-((Element)->Result)``
/// - ``NestedDictionary/mapValues(transform:)-((Element)->Result)``
/// - ``NestedDictionary/flattened(prefix:)``
/// - ``NestedDictionary/unflattened(_:)-([String:Element])``
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
    /// This is typically called via ``NestedDictionary/mapValues(transform:)-((Element)->Result)``.
    ///
    /// ### See Also
    /// - ``NestedDictionary/mapValues(transform:)-((Element)->Result)``
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

    /// Transform the values in the nested structure using the `transform()` function
    /// while simultaneously iterating the same structure in a second `NestedItem`.
    ///
    /// This transforms both values simultaneously and produces two new `NestedItem`
    /// conceptually somewhat like `map(zip(i1, i2))`.
    ///
    /// The structure of the second `NestedItem` should be identical to the first but these
    /// exceptions are allowed:
    ///
    /// - `.none` will match any value, e.g. you can iterate an empty second item
    /// - arrays do not need to be the same length -- the receiver's length is matched
    /// - dictionaries do not need to have the same keys -- the receivers keys are matched
    public func mapValues<E2, R1, R2>(
        _ item: NestedItem<Key, E2>, _ transform: (Element, E2?) throws -> (R1, R2?)
    ) rethrows -> (NestedItem<Key, R1>, NestedItem<Key, R2>) {
        switch (self, item) {
        case (.none, _):
            return (.none, .none)

        case (.value(let e1), .none):
            // allow Element, nil
            let (r1, r2) = try transform(e1, nil)
            if let r2 {
                return (.value(r1), .value(r2))
            } else {
                return (.value(r1), .none)
            }
        case (.value(let e1), .value(let e2)):
            // Element, E2
            let (r1, r2) = try transform(e1, e2)
            if let r2 {
                return (.value(r1), .value(r2))
            } else {
                return (.value(r1), .none)
            }

        case (.array(let array1), .none):
            // [Element], nil
            let r = try array1.map { try $0.mapValues(.none, transform) }
            return (.array(r.map { $0.0 }), .array(r.map { $0.1 }))
        case (.array(let array1), .array(let array2)):
            // [Element], [E2]
            var r1 = [NestedItem<Key, R1>]()
            var r2 = [NestedItem<Key, R2>]()

            for (index, v1) in array1.enumerated() {
                let v2 = (index >= array2.count) ? .none : array2[index]
                let (e1, e2) = try v1.mapValues(v2, transform)
                r1.append(e1)
                r2.append(e2)
            }

            return (.array(r1), .array(r2))

        case (.dictionary(let dictionary1), .none):
            // [Key:Element], nil
            var r1 = [Key: NestedItem<Key, R1>]()
            var r2 = [Key: NestedItem<Key, R2>]()

            for (key, v1) in dictionary1 {
                let (e1, e2) = try v1.mapValues(.none, transform)
                r1[key] = e1
                r2[key] = e2
            }

            return (.dictionary(r1), .dictionary(r2))

        case (.dictionary(let dictionary1), .dictionary(let dictionary2)):
            // [Key:Element], [Key:E2]
            var r1 = [Key: NestedItem<Key, R1>]()
            var r2 = [Key: NestedItem<Key, R2>]()

            for (key, v1) in dictionary1 {
                let v2 = dictionary2[key] ?? .none
                let (e1, e2) = try v1.mapValues(v2, transform)
                r1[key] = e1
                r2[key] = e2
            }

            return (.dictionary(r1), .dictionary(r2))

        case (.value, _), (.array, _), (.dictionary, _):
            fatalError("Unable to mapValues where item0 is \(self) and item1 is \(item)")
        }
    }

    /// helper for mapping 3 values
    static private func mapValues<E1, E2, E3, R1, R2, R3>(
        _ v1: E1, _ v2: E2?, _ v3: E3?, _ transform: (E1, E2?, E3?) throws -> (R1, R2?, R3?)
    ) rethrows -> (NestedItem<Key, R1>, NestedItem<Key, R2>, NestedItem<Key, R3>) {
        let (r1, r2, r3) = try transform(v1, v2, v3)
        let wr1 = NestedItem<Key, R1>.value(r1)
        let wr2 = r2 == nil ? .none : NestedItem<Key, R2>.value(r2!)
        let wr3 = r3 == nil ? .none : NestedItem<Key, R3>.value(r3!)
        return (wr1, wr2, wr3)
    }

    /// helper for mapping 3 arrays
    static private func mapArrays<E1, E2, E3, R1, R2, R3>(
        _ v1: [NestedItem<Key, E1>], _ v2: [NestedItem<Key, E2>], _ v3: [NestedItem<Key, E3>],
        _ transform: (E1, E2?, E3?) throws -> (R1, R2?, R3?)
    ) rethrows -> (NestedItem<Key, R1>, NestedItem<Key, R2>, NestedItem<Key, R3>) {
        var result1 = [NestedItem<Key, R1>]()
        var result2 = [NestedItem<Key, R2>]()
        var result3 = [NestedItem<Key, R3>]()

        for (index, i1) in v1.enumerated() {
            let i2 = (index >= v2.count) ? .none : v2[index]
            let i3 = (index >= v3.count) ? .none : v3[index]

            let (r1, r2, r3) = try i1.mapValues(i2, i3, transform)
            result1.append(r1)
            result2.append(r2)
            result3.append(r3)
        }

        return (.array(result1), .array(result2), .array(result3))
    }

    /// helper for mapping 3 dictionaries
    static private func mapDictionaries<E1, E2, E3, R1, R2, R3>(
        _ v1: [Key: NestedItem<Key, E1>], _ v2: [Key: NestedItem<Key, E2>],
        _ v3: [Key: NestedItem<Key, E3>], _ transform: (E1, E2?, E3?) throws -> (R1, R2?, R3?)
    ) rethrows -> (NestedItem<Key, R1>, NestedItem<Key, R2>, NestedItem<Key, R3>) {
        var result1 = [Key: NestedItem<Key, R1>]()
        var result2 = [Key: NestedItem<Key, R2>]()
        var result3 = [Key: NestedItem<Key, R3>]()

        func lookup<T>(_ dict: [Key: NestedItem<Key, T>], _ key: Key) -> NestedItem<Key, T> {
            if let v = dict[key] {
                return v
            } else {
                return .none
            }
        }

        for (key, i1) in v1 {
            let i2 = lookup(v2, key)
            let i3 = lookup(v3, key)

            let (r1, r2, r3) = try i1.mapValues(i2, i3, transform)
            result1[key] = r1
            result2[key] = r2
            result3[key] = r3
        }

        return (.dictionary(result1), .dictionary(result2), .dictionary(result3))
    }

    /// Transform the values in the nested structure using the `transform()` function
    /// while simultaneously iterating the same structure in a second and third `NestedItem`.
    ///
    /// This transforms all three values simultaneously and produces three new `NestedItem`
    /// conceptually somewhat like `map(zip(i1, i2, i3))`.
    ///
    /// The structure of the second and third `NestedItem` should be identical to the first but these
    /// exceptions are allowed:
    ///
    /// - `.none` will match any value, e.g. you can iterate an empty second item
    /// - arrays do not need to be the same length -- the receiver's length is matched
    /// - dictionaries do not need to have the same keys -- the receivers keys are matched
    public func mapValues<E2, E3, R1, R2, R3>(
        _ item1: NestedItem<Key, E2>, _ item2: NestedItem<Key, E3>,
        _ transform: (Element, E2?, E3?) throws -> (R1, R2?, R3?)
    ) rethrows -> (NestedItem<Key, R1>, NestedItem<Key, R2>, NestedItem<Key, R3>) {
        switch (self, item1, item2) {
        case (.none, _, _):
            return (.none, .none, .none)

        // handle various combinations of .value
        case (.value(let e1), .none, .none):
            return try Self.mapValues(e1, nil as E2?, nil as E3?, transform)
        case (.value(let e1), .value(let e2), .none):
            return try Self.mapValues(e1, e2, nil as E3?, transform)
        case (.value(let e1), .none, .value(let e3)):
            return try Self.mapValues(e1, nil as E2?, e3, transform)
        case (.value(let e1), .value(let e2), .value(let e3)):
            return try Self.mapValues(e1, e2, e3, transform)

        // combinations of .array
        case (.array(let a1), .none, .none):
            return try Self.mapArrays(a1, [], [], transform)
        case (.array(let a1), .array(let a2), .none):
            return try Self.mapArrays(a1, a2, [], transform)
        case (.array(let a1), .none, .array(let a3)):
            return try Self.mapArrays(a1, [], a3, transform)
        case (.array(let a1), .array(let a2), .array(let a3)):
            return try Self.mapArrays(a1, a2, a3, transform)

        // combinations of .dictionary
        case (.dictionary(let d1), .none, .none):
            return try Self.mapDictionaries(d1, [:], [:], transform)
        case (.dictionary(let d1), .dictionary(let d2), .none):
            return try Self.mapDictionaries(d1, d2, [:], transform)
        case (.dictionary(let d1), .none, .dictionary(let d3)):
            return try Self.mapDictionaries(d1, [:], d3, transform)
        case (.dictionary(let d1), .dictionary(let d2), .dictionary(let d3)):
            return try Self.mapDictionaries(d1, d2, d3, transform)

        case (.value, _, _), (.array, _, _), (.dictionary, _, _):
            fatalError(
                "Unable to mapValues where item0 is \(self) and item1 is \(item1) and item2 is \(item2)"
            )
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

    /// A combination of ``mapValues(_:)``, ``flattened(prefix:)`` and ``unflattened(_:)``.
    ///
    /// This transforms a nested structure into a new nested structure with the same shape but transformed
    /// values.  The `transform` function receives a dotted path and the value to transform.
    public func mapValues<Result>(
        prefix: String? = nil, _ transform: (String, Element) throws -> Result
    ) rethrows -> NestedItem<Key, Result> {
        func newPrefix(_ i: CustomStringConvertible) -> String {
            if let prefix {
                return "\(prefix).\(i)"
            } else {
                return i.description
            }
        }
        switch self {
        case .none:
            return .none
        case .value(let element):
            return try .value(transform(prefix ?? "", element))
        case .array(let array):
            return try .array(
                array.enumerated().map { (index, value) in
                    try value.mapValues(prefix: newPrefix(index), transform)
                })
        case .dictionary(let dictionary):
            // produce the dictionary in canonical order (sorted keys)
            return try .dictionary(
                Dictionary(
                    uniqueKeysWithValues:
                        dictionary
                        .sorted { lhs, rhs in String(describing: lhs.0) < String(describing: rhs.0)
                        }
                        .map { (key, value) in
                            try (
                                key,
                                value.mapValues(
                                    prefix: newPrefix(String(describing: key)), transform)
                            )
                        }))
        }
    }

    /// Reduces the contents of the NestedDictionary by visiting each value and applying the `reducer`
    /// function, accumulating the result.
    ///
    /// Typically called via ``NestedDictionary/reduce(_:_:)``.
    public func reduce<R>(_ initialValue: R, _ reducer: (R, Element) throws -> R) rethrows -> R {
        switch self {
        case .none:
            return initialValue
        case .value(let element):
            return try reducer(initialValue, element)
        case .array(let array):
            var v = initialValue
            for item in array {
                v = try item.reduce(v, reducer)
            }
            return v
        case .dictionary(let dictionary):
            var v = initialValue
            for item in dictionary.values {
                v = try item.reduce(v, reducer)
            }
            return v
        }
    }

    /// Return a flattened representation of the structured contents as an array of key/value tuples.
    ///
    /// This is typically called via ``NestedDictionary/flattened(prefix:)``.
    ///
    /// ### See Also
    /// - ``flattenedValues()``
    /// - ``unflattened(_:)``
    /// - ``NestedDictionary/flattened(prefix:)``
    /// - ``NestedDictionary/unflattened(_:)-4p8bn``
    /// - ``NestedDictionary/unflattened(_:)-([String:Element])``
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
            // produce the dictionary in canonical order (sorted keys)
            return
                dictionary
                .sorted { lhs, rhs in String(describing: lhs.0) < String(describing: rhs.0) }
                .flatMap { (key, value) in
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
    /// - ``NestedDictionary/unflattened(_:)-([String:Element])``
    public static func unflattened(_ tree: some Collection<(Key, Element)>) -> NestedItem<
        Key, Element
    >
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

    private static func unflattenedRecurse(_ tree: some Collection<(String, Element)>)
        -> NestedItem<
            String, Element
        >
    {
        if tree.count == 1, case ("", let value)? = tree.first {
            return .value(value)
        }

        var children = [String: [(String, Element)]]()
        for (key, value) in tree {
            let current: String
            let next: String
            if let dotIndex = key.firstIndex(of: ".") {
                current = String(key.prefix(upTo: dotIndex))
                next = String(key.suffix(from: key.index(after: dotIndex)))
            } else {
                current = key
                next = ""
            }
            children[String(current), default: []].append((String(next), value))
        }

        switch UnflattenKind.detect(key: tree.first!.0) {
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

    /// Return a flattened representation of the structured contents as an array of values.
    ///
    /// This is typically called via ``NestedDictionary/flattenedValues()``.
    /// Note that unlike ``flattened(prefix:)`` this cannot be reconstructed
    /// back into the original structure, but can be used
    /// with ``NestedDictionary/replacingValues(with:)`` in a similar fashion.
    ///
    /// ### See Also
    /// - ``flattened(prefix:)``
    /// - ``NestedDictionary/flattenedValues()``
    /// - ``NestedDictionary/replacingValues(with:)``
    public func flattenedValues() -> [Element] {
        switch self {
        case .none:
            return []
        case .value(let element):
            return [element]
        case .array(let array):
            return array.flatMap { $0.flattenedValues() }
        case .dictionary(let dictionary):
            // produce the dictionary in canonical order (sorted keys)
            return
                dictionary
                .sorted { lhs, rhs in String(describing: lhs.0) < String(describing: rhs.0) }
                .flatMap { (key, value) in
                    value.flattenedValues()
                }
        }
    }

    func replacingValues<Values: Collection<Element>>(with values: Values, index: Values.Index) -> (
        Values.Index, NestedItem<Key, Element>
    ) {
        switch self {
        case .none:
            return (index, .none)
        case .value:
            return (values.index(after: index), .value(values[index]))
        case .array(let array):
            var result = [NestedItem<Key, Element>]()
            var index = index

            for element in array {
                let (newIndex, replaced) = element.replacingValues(with: values, index: index)
                result.append(replaced)
                index = newIndex
            }

            return (index, .array(result))
        case .dictionary(let dictionary):
            var result = [Key: NestedItem<Key, Element>]()
            var index = index

            let sorted = dictionary.sorted { lhs, rhs in
                String(describing: lhs) < String(describing: rhs)
            }

            for (key, element) in sorted {
                let (newIndex, replaced) = element.replacingValues(with: values, index: index)
                result[key] = replaced
                index = newIndex
            }

            return (index, .dictionary(result))
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

extension NestedItem: Sendable where Element: Sendable, Key: Sendable {
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
    /// - ``NestedDictionary/subscript(_:)-(Key)``
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

    /// Transform the values in the nested structure using the `transform()` function
    /// while simultaneously iterating the same structure in a second `NestedDictionary`.
    ///
    /// This transforms both values simultaneously and produces two new `NestedDictionary`
    /// conceptually somewhat like `map(zip(i1, i2))`.
    ///
    /// The structure of the second `NestedDictionary` should be identical to the first but these
    /// exceptions are allowed:
    ///
    /// - `.none` will match any value, e.g. you can iterate an empty second item
    /// - arrays do not need to be the same length -- the receiver's length is matched
    /// - dictionaries do not need to have the same keys -- the receivers keys are matched
    ///
    /// ### See Also
    /// - ``NestedItem/mapValues(_:_:)``
    /// - ``mapValues(_:_:transform:)-52e9l``
    public func mapValues<E2, R1, R2>(
        _ dictionary1: NestedDictionary<Key, E2>, transform: (Element, E2?) throws -> (R1, R2?)
    ) rethrows
        -> (NestedDictionary<Key, R1>, NestedDictionary<Key, R2>)
    {
        let (r1, r2) = try NestedItem.dictionary(contents).mapValues(
            .dictionary(dictionary1.contents), transform)

        switch (r1, r2) {
        case (.dictionary(let v1), .dictionary(let v2)):
            return (NestedDictionary<Key, R1>(values: v1), NestedDictionary<Key, R2>(values: v2))
        default:
            fatalError()
        }
    }

    public func mapValues<E2, R1>(
        _ dictionary1: NestedDictionary<Key, E2>, transform: (Element, E2?) -> R1
    )
        -> NestedDictionary<Key, R1>
    {
        let wrappedTransform = { (e1: Element, e2: E2?) -> (R1, Int) in
            let r1 = transform(e1, e2)
            return (r1, 0)
        }
        let (r1, _) = NestedItem.dictionary(contents).mapValues(
            .dictionary(dictionary1.contents), wrappedTransform)

        switch r1 {
        case .dictionary(let v1):
            return NestedDictionary<Key, R1>(values: v1)
        default:
            fatalError()
        }
    }

    /// Transform the values in the nested structure using the `transform()` function
    /// while simultaneously iterating the same structure in a second and third `NestedDictionary`.
    ///
    /// This transforms all three values simultaneously and produces three new `NestedDictionary`
    /// conceptually somewhat like `map(zip(i1, i2, i3))`.
    ///
    /// The structure of the second and third `NestedDictionary` should be identical to the first but these
    /// exceptions are allowed:
    ///
    /// - `.none` will match any value, e.g. you can iterate an empty second item
    /// - arrays do not need to be the same length -- the receiver's length is matched
    /// - dictionaries do not need to have the same keys -- the receivers keys are matched
    ///
    /// ### See Also
    /// - ``NestedItem/mapValues(_:_:_:)``
    /// - ``mapValues(_:transform:)-4ctis``
    public func mapValues<E2, E3, R1, R2, R3>(
        _ dictionary1: NestedDictionary<Key, E2>, _ dictionary2: NestedDictionary<Key, E3>,
        transform: (Element, E2?, E3?) throws -> (R1, R2?, R3?)
    ) rethrows
        -> (NestedDictionary<Key, R1>, NestedDictionary<Key, R2>, NestedDictionary<Key, R3>)
    {
        let (r1, r2, r3) = try NestedItem.dictionary(contents).mapValues(
            .dictionary(dictionary1.contents), .dictionary(dictionary2.contents), transform)

        switch (r1, r2, r3) {
        case (.dictionary(let v1), .dictionary(let v2), .dictionary(let v3)):
            return (
                NestedDictionary<Key, R1>(values: v1), NestedDictionary<Key, R2>(values: v2),
                NestedDictionary<Key, R3>(values: v3)
            )
        default:
            fatalError()
        }
    }

    /// Transform the values in the nested structure using the `transform()` function
    /// while simultaneously iterating the same structure in a second and third `NestedDictionary`.
    ///
    /// This transforms all three values simultaneously and produces **two** new `NestedDictionary`
    /// conceptually somewhat like `map(zip(i1, i2, i3))`.
    ///
    /// The structure of the second and third `NestedDictionary` should be identical to the first but these
    /// exceptions are allowed:
    ///
    /// - `.none` will match any value, e.g. you can iterate an empty second item
    /// - arrays do not need to be the same length -- the receiver's length is matched
    /// - dictionaries do not need to have the same keys -- the receivers keys are matched
    ///
    /// ### See Also
    /// - ``NestedItem/mapValues(_:_:_:)``
    /// - ``mapValues(_:transform:)-4ctis``
    public func mapValues<E2, E3, R1, R2>(
        _ dictionary1: NestedDictionary<Key, E2>, _ dictionary2: NestedDictionary<Key, E3>,
        transform: (Element, E2?, E3?) -> (R1, R2?)
    )
        -> (NestedDictionary<Key, R1>, NestedDictionary<Key, R2>)
    {
        let wrappedTransform = { (e1: Element, e2: E2?, e3: E3?) -> (R1, R2?, Int) in
            let (r1, r2) = transform(e1, e2, e3)
            return (r1, r2, 0)
        }
        let (r1, r2, _) = NestedItem.dictionary(contents).mapValues(
            .dictionary(dictionary1.contents), .dictionary(dictionary2.contents), wrappedTransform)

        switch (r1, r2) {
        case (.dictionary(let v1), .dictionary(let v2)):
            return (
                NestedDictionary<Key, R1>(values: v1), NestedDictionary<Key, R2>(values: v2)
            )
        default:
            fatalError()
        }
    }

    /// Reduces the contents of the NestedDictionary by visiting each value and applying the `reducer`
    /// function, accumulating the result.
    public func reduce<R>(_ initialValue: R, _ reducer: (R, Element) throws -> R) rethrows -> R {
        try NestedItem.dictionary(contents).reduce(initialValue, reducer)
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

    /// Transform the values in the nested structure using the `transform()` function.
    ///
    /// This function receives both keys in dotted form and values.  This is roughly equivalent
    /// to:
    ///
    /// ```swift
    /// let transformed = nd.flattened().map { key, value in (key, transformedValue) }
    /// let result = NestedDictionary.unflattened(transformed)
    /// ```
    public func mapValues<Result>(transform: (String, Element) throws -> Result) rethrows
        -> NestedDictionary<Key, Result>
    {
        switch try asItem().mapValues(transform) {
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
    /// - ``unflattened(_:)-([String:Element])``
    public func flattened(prefix: String? = nil) -> [(String, Element)] {
        asItem().flattened(prefix: prefix)
    }

    /// Convert a flattened list of key/value tuples back into a `NestedDictionary` structure.
    ///
    /// ### See Also
    /// - ``flattened(prefix:)``
    /// - ``unflattened(_:)-([String:Element])``
    static public func unflattened(_ flat: some Collection<(String, Element)>) -> Self
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
    static public func unflattened(_ flat: [String: Element]) -> Self where Key == String {
        unflattened(flat.map { $0 })
    }

    /// Return a flattened representation of the structured contents as an array of values.
    ///
    /// Note that unlike ``flattened(prefix:)`` this cannot be reconstructed
    /// back into the original structure
    ///
    /// ### See Also
    /// - ``flattened(prefix:)``
    public func flattenedValues() -> [Element] {
        asItem().flattenedValues()
    }

    /// Return a new `NestedDictionary` with the values replaced
    /// by a flat array of values.
    ///
    /// ### See Also
    /// - ``flattenedValues()``
    public func replacingValues(with values: some Collection<Element>) -> Self {
        switch asItem().replacingValues(with: values, index: values.startIndex) {
        case (_, .dictionary(let values)):
            return NestedDictionary(values: values)
        default:
            fatalError()
        }
    }
}

extension NestedDictionary: Equatable where Element: Equatable {
}

extension NestedDictionary: Sendable where Element: Sendable, Key: Sendable {
}

extension NestedDictionary: Collection {
    public typealias CollectionElement = Dictionary<Key, NestedItem<Key, Element>>.Element
    public typealias Index = Dictionary<Key, NestedItem<Key, Element>>.Index

    public var startIndex: Index { contents.startIndex }
    public var endIndex: Index { contents.endIndex }

    public subscript(position: Index) -> CollectionElement {
        contents[position]
    }

    public func index(after i: Index) -> Index {
        contents.index(after: i)
    }
}
