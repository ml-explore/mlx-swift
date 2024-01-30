// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXNN

class ModuleTests: XCTestCase {

    func newTestModule() -> Module {
        class ChildTestModule: Module {
            let p = MLXArray([100, 200])
            let i = 10
            let a = [
                MLXArray([200, 300]),
                MLXArray([400, 500]),
            ]
            let d = [
                "key": MLXArray([1, 2])
            ]
        }

        class TestModule: Module {
            let p = MLXArray([100, 200])

            @ModuleInfo(key: "alternateChild")
            var child = ChildTestModule()

            let d = [
                "a": ChildTestModule(),
                "b": ChildTestModule(),
            ]

            let topLevelParam = 50.0

            @ParameterInfo(key: "alternateNameArray")
            var topLevelArrayOfParams = [
                MLXArray([200, 300]),
                MLXArray([400, 500]),
            ]
            @ParameterInfo(key: "alternateNameDict")
            var topLevelDictionaryOfParams = [
                "key": MLXArray([1, 2])
            ]
        }

        return TestModule()
    }

    func testModuleDescription() {
        let m = newTestModule()

        let expected =
            """
            TestModule(topLevelParam=50.0) {
              alternateChild: ChildTestModule(i=10),
              d: [
                a: ChildTestModule(i=10),
                b: ChildTestModule(i=10)
              ],
            }
            """

        let d = m.description
        XCTAssertEqual(d, expected)
    }

    func testNested() {
        let m = newTestModule()
        let v = m.filterMap(filter: Module.filterAll)

        let expected =
            """
            [
              alternateChild: [
                a: [
                  parameters(array([200, 300], dtype=int64)),
                  parameters(array([400, 500], dtype=int64))
                ],
                d: [
                  key: parameters(array([1, 2], dtype=int64))
                ],
                i: other(10),
                p: parameters(array([100, 200], dtype=int64))
              ],
              alternateNameArray: [
                parameters(array([200, 300], dtype=int64)),
                parameters(array([400, 500], dtype=int64))
              ],
              alternateNameDict: [
                key: parameters(array([1, 2], dtype=int64))
              ],
              d: [
                a: [
                  a: [
                    parameters(array([200, 300], dtype=int64)),
                    parameters(array([400, 500], dtype=int64))
                  ],
                  d: [
                    key: parameters(array([1, 2], dtype=int64))
                  ],
                  i: other(10),
                  p: parameters(array([100, 200], dtype=int64))
                ],
                b: [
                  a: [
                    parameters(array([200, 300], dtype=int64)),
                    parameters(array([400, 500], dtype=int64))
                  ],
                  d: [
                    key: parameters(array([1, 2], dtype=int64))
                  ],
                  i: other(10),
                  p: parameters(array([100, 200], dtype=int64))
                ]
              ],
              p: parameters(array([100, 200], dtype=int64)),
              topLevelParam: other(50.0)
            ]
            """

        XCTAssertEqual(v.count, 6)
        let d = v.description
        XCTAssertEqual(d, expected)
    }

    func testModuleItems() {
        let m = newTestModule()

        // all the ivars from TestModule
        let t = m.items()

        XCTAssertEqual(t.count, 6)

        for (k, v) in t {
            switch (k, v) {
            case ("p", .value(.parameters)): break
            case ("alternateChild", .value(.module)): break
            case ("d", .dictionary): break
            case ("topLevelParam", .value(.other)): break
            case ("alternateNameArray", .array): break
            case ("alternateNameDict", .dictionary): break
            default:
                XCTFail("Unexpected: \(k) = \(v)))")
            }
        }
    }

    func testModuleParameters() {
        let m = newTestModule()

        let p = m.parameters()
        XCTAssertEqual(p.count, 5)

        for (k, v) in p {
            switch (k, v) {
            case ("p", .value): break
            case ("alternateNameArray", .array): break
            case ("alternateNameDict", .dictionary): break

            case ("d", .dictionary): break
            case ("alternateChild", .dictionary): break
            default:
                XCTFail("Unexpected: \(k) = \(v)))")
            }
        }
    }

    func testChildren() {
        let m = newTestModule()

        let c = m.children()
        XCTAssertEqual(c.count, 2)

        for (k, v) in c {
            switch (k, v) {
            case ("alternateChild", .value): break
            case ("d", .dictionary): break
            default:
                XCTFail("Unexpected: \(k) = \(v)))")
            }
        }
    }

    func testSequentialBuilder() {
        let b = Bool.random()
        let s = Sequential {
            Tanh()
            if b {
                Tanh()
            } else {
                Sigmoid()
            }
            for _ in 0 ..< 3 {
                Linear(10, 20)
            }
        }
        XCTAssertEqual(s.layers.count, 5)
    }

    func newStructureModule() -> Module {

        class Leaf: Module {
        }

        class InteriorSingle: Module {
            @ModuleInfo
            var child = Linear(10, 10)
        }

        class InteriorMultiple: Module {
            @ModuleInfo
            var children = [
                Linear(10, 10),
                Linear(10, 10),
            ]
        }

        class StructureModel: Module {
            let parameters = MLXArray(10)

            let leaf = Leaf()
            let interior1 = InteriorSingle()
            let interior2 = InteriorMultiple()

            @ModuleInfo
            var child = Linear(10, 10)

            var nonWrappedChild = Linear(10, 10)
        }

        return StructureModel()
    }

    func testLeafModules() {
        let m = newStructureModule()
        let modules = m.leafModules()
        let v = Dictionary(uniqueKeysWithValues: modules.flattened())

        XCTAssertEqual(v.count, 6)
        XCTAssertEqual(
            Set(v.keys),
            Set([
                "child", "interior2.children.1", "leaf", "interior2.children.0", "interior1.child",
                "nonWrappedChild",
            ]))
    }

    func testChildren2() {
        let m = newStructureModule()
        // only the modules attached to the top level
        let modules = m.children()
        let v = Dictionary(uniqueKeysWithValues: modules.flattened())

        XCTAssertEqual(v.count, 5)
        XCTAssertEqual(
            Set(v.keys), Set(["child", "interior2", "leaf", "interior1", "nonWrappedChild"]))
    }

    func testVisitModule() {
        let m = newStructureModule()
        var collect = [String: String]()

        m.visit { key, m in
            collect[key] = String(describing: type(of: m))
        }

        XCTAssertEqual(collect.count, 9)
        XCTAssertEqual(collect[""], "StructureModel")
        XCTAssertEqual(collect["interior2.children.1"], "Linear")
    }

    func testUpdateModuleSameType() throws {
        // set a single module with same type
        let m = newStructureModule()
        var u = NestedDictionary<String, Module>()
        u["child"] = .value(Linear(4, 4))
        m.update(modules: u)

        let modules = m.children()
        if let child = modules["child"], let child = child.unwrap() as? Linear {
            // make sure it was updated
            XCTAssertEqual(child.shape.0, 4)
        } else {
            XCTFail("child is nil or not Linear")
        }
    }

    func testUpdateModuleSubType() throws {
        // set a single module with a subtype
        let m = newStructureModule()
        var u = NestedDictionary<String, Module>()
        u["child"] = .value(QuantizedLinear(256, 256))
        m.update(modules: u)

        let modules = m.children()
        if let child = modules["child"], let child = child.unwrap() as? QuantizedLinear {
            // make sure it was updated
            XCTAssertEqual(child.shape.0, 256)
        } else {
            XCTFail("child is nil or not QuantizedLinear")
        }
    }

    func testUpdateModuleArray() throws {
        // update an array of modules
        let m = newStructureModule()
        var u = NestedDictionary<String, Module>()
        u["interior2"] = .dictionary([
            "children": .array([
                .value(Linear(5, 5)),
                .value(Linear(5, 5)),
                .value(Linear(5, 5)),
            ])
        ])
        m.update(modules: u)

        let modules = m.children()
        if let interior2 = modules["interior2"], let interior2 = interior2.unwrap() as? Module {
            let modules = interior2.children()
            if let children = modules["children"], let children = children.unwrap() as? [Linear] {
                XCTAssertEqual(children.count, 3)
                XCTAssertEqual(children[0].shape.0, 5)
            } else {
                XCTFail("children is nil or wrong type")
            }
        } else {
            XCTFail("interior2 is nil")
        }
    }

    func testUpdateModuleNotWrapped() throws {
        // a module that doesn't have @ModuleInfo
        let m = newStructureModule()
        var u = NestedDictionary<String, Module>()
        u["nonWrappedChild"] = .value(Linear(4, 4))

        do {
            try m.update(modules: u, verify: .all)
            XCTFail("should have thrown")
        } catch {
            print("Expected: \(error)")
        }
    }

    func testQuantize() throws {
        class C: Module {
            @ModuleInfo
            var child = Linear(256, 256)

            var other = Sigmoid()
        }
        class M: Module {
            let module = C()
        }

        let m = M()
        QuantizedLinear.quantize(model: m)

        XCTAssertTrue(m.module.child is QuantizedLinear)
    }

}
