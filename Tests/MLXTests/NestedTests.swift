// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import Testing

@Suite
struct NestedTests {

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

    @Test func collection() {
        let n = newNested()
        #expect(n.count == 4)
        #expect(Set(n.keys) == Set(["value", "array1", "array2", "dictionary"]))
    }

    @Test func mapValues() {
        let n = newNested()
        let incremented = n.mapValues { $0 + 1 }
        let expected = newNested(values: Self.defaultValues.map { $0 + 1 })

        #expect(incremented == expected)
    }

    @Test func compactMapValues() {
        let n = newNested()
        let trimmed = n.compactMapValues { $0 % 2 == 1 ? $0 : nil }
        #expect(trimmed.count == 2)
        #expect(trimmed["array1"] == .array([.value(1)]))
    }

    @Test func flatten1() {
        let v: NestedItem<String, Int> = .array([.array([.array([.value(10)])])])

        let f = v.flattened()
        #expect(f.description == #"[("0.0.0", 10)]"#)

        let uf = NestedItem<String, Int>.unflattened(f)
        #expect(uf == v)
    }

    @Test func flatten() {
        let n = newNested()
        let f = n.flattened()
        let n2 = NestedDictionary<String, Int>.unflattened(f)
        #expect(n == n2)
    }

    @Test func flattenedValues() {
        // replacingValues() uses the original structure
        // and makes an identical structure with new values
        // from a flat array of values
        let n = newNested()

        let f = n.flattenedValues().map { $0 + 1 }
        let n2 = n.replacingValues(with: f)

        let expected = n.mapValues { $0 + 1 }
        #expect(n2 == expected)
    }

    @Test func map2() {
        // map 2 parallel structures
        var d1 = NestedDictionary<String, Int>()
        d1["layers"] = .array([
            .dictionary([
                "w": .value(10)
            ]),
            .dictionary([
                "w": .value(20)
            ]),
        ])

        // map both the input dictionary and another empty dictionary and produce
        // a new nested (r2) with the same structure but the values are strings
        let (_, r2) = d1.mapValues(NestedDictionary<String, String>()) { v1, v2 in
            return (v1, "value = \(v1)")
        }

        let expected = d1.mapValues { "value = \($0)" }

        #expect(r2 == expected)
    }

    @Test func map3() {
        // map 3 parallel structures
        var d1 = NestedDictionary<String, Int>()
        d1["layers"] = .array([
            .dictionary([
                "w": .value(10)
            ]),
            .dictionary([
                "w": .value(20)
            ]),
        ])

        var d2 = NestedDictionary<String, Float>()
        d2["layers"] = .array([
            .dictionary([
                "w": .value(3.5)
            ]),
            .dictionary([
                "w": .value(123.5)
            ]),
        ])

        let (_, r2, r3) = d1.mapValues(d2, NestedDictionary<String, String>()) { v1, v2, v3 in
            return (v1, -(v2 ?? 0), "value = \(v1) + \(v2 ?? 0)")
        }

        let expected2 = d2.mapValues { -$0 }
        let (_, expected3) = d1.mapValues(d2) { (0, "value = \($0) + \($1 ?? 0)") }

        #expect(r2 == expected2)
        #expect(r3 == expected3)
    }

}
