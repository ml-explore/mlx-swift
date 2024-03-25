// Copyright © 2024 Apple Inc.

import Foundation

/// Parameter for convolutions allowing single integers or tuples.
///
/// For example the numeric parameters here are all ``IntOrPair``:
///
/// ```swift
/// conv2d(input, weights, stride: 3, padding: [2, 2], padding: .init((2, 3)))
/// ```
///
/// ### See Also:
/// - ``IntOrArray``
public struct IntOrPair: ExpressibleByIntegerLiteral, ExpressibleByArrayLiteral {
    public let values: (Int, Int)

    public var first: Int { values.0 }
    public var second: Int { values.1 }

    public init(integerLiteral value: Int) {
        self.values = (value, value)
    }

    public init(arrayLiteral elements: Int...) {
        precondition(elements.count == 2)
        self.values = (elements[0], elements[1])
    }

    public init(_ values: [Int]) {
        precondition(values.count == 2)
        self.values = (values[0], values[1])
    }

    public init(_ values: (Int, Int)) {
        self.values = values
    }
}

/// Parameter for convolutions allowing single integers or arrays.
///
/// For example the numeric parameters here are all ``IntOrArray``:
///
/// ```swift
/// convGeneral(input, weights, strides: 3, padding: [2, 2, 1, 0, 0], ...)
/// ```
///
/// ### See Also:
/// - ``IntOrPair``
/// - ``FloatOrArray``
public enum IntOrArray: ExpressibleByIntegerLiteral, ExpressibleByArrayLiteral {
    case int(Int)
    case array([Int])

    public init(integerLiteral value: Int) {
        self = .int(value)
    }

    public init(arrayLiteral elements: Int...) {
        self = .array(elements)
    }

    public var asArray: [Int] {
        switch self {
        case .int(let v):
            [v]
        case .array(let v):
            v
        }
    }

    public var asInt32Array: [Int32] {
        switch self {
        case .int(let v):
            [Int32(v)]
        case .array(let v):
            v.asInt32
        }
    }

    public var count: Int {
        switch self {
        case .int:
            1
        case .array(let array):
            array.count
        }
    }

}

/// Parameter for taking a single or multiple Float values.
///
/// For example the numeric parameters here are all ``FloatOrArray``:
///
/// ```swift
/// Upsample(scaleFactor: 10.5)
/// Upsample(scaleFactor: [4.5, 3.5])
/// ```
///
/// ### See Also:
/// - ``IntOrArray``
public enum FloatOrArray: ExpressibleByFloatLiteral, ExpressibleByArrayLiteral {
    case float(Float)
    case array([Float])

    public init(floatLiteral value: Float) {
        self = .float(value)
    }

    public init(arrayLiteral elements: Float...) {
        self = .array(elements)
    }

    public var asArray: [Float] {
        switch self {
        case .float(let v):
            [v]
        case .array(let v):
            v
        }
    }

    public func asArray(dimensions: Int) -> [Float] {
        switch self {
        case .float(let float):
            return Array(repeating: float, count: dimensions)
        case .array(let array):
            precondition(
                dimensions == array.count,
                "FloatOrArray dimensions mismatch: \(dimensions) != \(array.count)")
            return array
        }
    }

    public var count: Int {
        switch self {
        case .float:
            1
        case .array(let array):
            array.count
        }
    }

}
