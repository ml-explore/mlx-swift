import Foundation
import MLX
import XCTest

class NestedTests: XCTestCase {

    static let defaultValues = [10, 1, 2, 1, 2, 3, 10, 20, 30]

    func newNested(values: [Int] = defaultValues) -> NestedDictionary<String, Int> {
        var values = values
        var result = NestedDictionary<String, Int>()
        result["value"] = .value(values.removeFirst())
        result["array1"] = .array([.value(values.removeFirst()), .value(values.removeFirst())])
        result["array2"] = .array([
            .dictionary([
                "a": .value(values.removeFirst()),
                "b": .array([.value(values.removeFirst()), .value(values.removeFirst())]),
            ])
        ])
        result["dictionary"] = .dictionary([
            "a": .value(values.removeFirst()),
            "b": .array([.value(values.removeFirst()), .value(values.removeFirst())]),
        ])
        return result
    }

    func testCollection() {
        let n = newNested()
        XCTAssertEqual(n.count, 4)
        XCTAssertEqual(Set(n.keys), Set(["value", "array1", "array2", "dictionary"]))
    }

    func testMapValues() {
        let n = newNested()
        let incremented = n.mapValues { $0 + 1 }
        let expected = newNested(values: Self.defaultValues.map { $0 + 1 })

        XCTAssertEqual(incremented, expected)
    }

    func testCompactMapValues() {
        let n = newNested()
        let trimmed = n.compactMapValues { $0 % 2 == 1 ? $0 : nil }
        XCTAssertEqual(trimmed.count, 2)
        XCTAssertEqual(trimmed["array1"], .array([.value(1)]))
    }

    func testFlatten1() {
        let v: NestedItem<String, Int> = .array([.array([.array([.value(10)])])])

        let f = v.flattened()
        XCTAssertEqual(f.description, #"[("0.0.0", 10)]"#)

        let uf = NestedItem<String, Int>.unflattened(f)
        XCTAssertEqual(uf, v)
    }

    func testFlatten() {
        let n = newNested()
        let f = n.flattened()
        let n2 = NestedDictionary<String, Int>.unflattened(f)
        XCTAssertEqual(n, n2)
    }

}
