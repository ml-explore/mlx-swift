// Copyright © 2026 Apple Inc.

import Numerics

/// A value that can appear in a nested array used to create an ``MLXArray``.
///
/// Conformance is recursive: the built-in ``HasDType`` scalars conform and so does an `Array` of
/// conforming values, adding one dimension.  A nested array therefore describes both the shape
/// and the contents of an ``MLXArray``:
///
/// ```swift
/// // shape [2, 3]
/// let a = MLXArray([[1, 2, 3], [4, 5, 6]])
/// ```
///
/// This exists to express the nested array initializers, e.g. ``MLXArray/init(_:)-([[N]])`` --
/// there should be no need to conform your own types to it.
///
/// ### See Also
/// - <doc:initialization>
public protocol NestedArrayElement {

    /// The scalar type at the leaves of the nesting, e.g. `Int` for `[[Int]]`.
    associatedtype Scalar: HasDType

    /// The number of dimensions this type contributes -- `0` for a scalar, `1` for `[Scalar]`.
    static var nestedRank: Int { get }

    /// The shape of the receiver.
    ///
    /// Note: the nesting must be rectangular -- every element at a given level must have the
    /// same shape -- or the precondition will fail.
    ///
    /// Note: an empty level says nothing about the levels below it, so those are reported as
    /// `0` -- `[[Int]]()` has shape `[0, 0]`.
    var nestedShape: [Int] { get }

    /// Append the receiver's scalars to `values` in row major order.
    func appendNestedScalars(to values: inout [Scalar])
}

extension NestedArrayElement where Self: HasDType {
    public static var nestedRank: Int { 0 }

    public var nestedShape: [Int] { [] }

    public func appendNestedScalars(to values: inout [Self]) {
        values.append(self)
    }
}

// the scalars, providing the base case of the recursion.  HasDType deliberately does not refine
// this protocol: conditional conformances to HasDType (Complex<Float> is one) would not imply it,
// so that would break any such conformance outside this package.  Keep this list in sync with
// the HasDType conformances in DType.swift.
extension Bool: NestedArrayElement {}
extension Int: NestedArrayElement {}
extension Int8: NestedArrayElement {}
extension Int16: NestedArrayElement {}
extension Int32: NestedArrayElement {}
extension Int64: NestedArrayElement {}
extension UInt8: NestedArrayElement {}
extension UInt16: NestedArrayElement {}
extension UInt32: NestedArrayElement {}
extension UInt64: NestedArrayElement {}
extension UInt: NestedArrayElement {}
#if !arch(x86_64)
    extension Float16: NestedArrayElement {}
#endif
extension Float32: NestedArrayElement {}
extension Float64: NestedArrayElement {}
extension Complex<Float>: NestedArrayElement {}

extension Array: NestedArrayElement where Element: NestedArrayElement {
    public typealias Scalar = Element.Scalar

    public static var nestedRank: Int { 1 + Element.nestedRank }

    public var nestedShape: [Int] {
        // an empty level tells us nothing about the levels below it, but the static rank does
        let inner = first?.nestedShape ?? [Int](repeating: 0, count: Element.nestedRank)
        for element in dropFirst() {
            precondition(
                element.nestedShape == inner,
                "nested array is not rectangular: \(element.nestedShape) != \(inner)")
        }
        return [count] + inner
    }

    public func appendNestedScalars(to values: inout [Element.Scalar]) {
        for element in self {
            element.appendNestedScalars(to: &values)
        }
    }
}
