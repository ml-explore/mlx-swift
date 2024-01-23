import Foundation
import XCTest
@testable import Mlx

class ModuleTests : XCTestCase {
    
    func newTestModule() -> Module {
        class ChildTestModule : Module {
            let p = MLXArray([100, 200])
            let i = 10
            let a = [
                MLXArray([200, 300]),
                MLXArray([400, 500]),
            ]
            let d = [
                "key": MLXArray([1, 2]),
            ]
        }
        
        class TestModule : Module {
            let p = MLXArray([100, 200])
            let child = ChildTestModule()
            let d = [
                "a": ChildTestModule(),
                "b": ChildTestModule(),
            ]
            
            let topLevelParam = 50.0
            
            @Property(key: "alternateNameArray")
            var topLevelArrayOfParams = [
                MLXArray([200, 300]),
                MLXArray([400, 500]),
            ]
            @Property(key: "alternateNameDict")
            var topLevelDictionaryOfParams = [
                "key": MLXArray([1, 2]),
            ]
        }
        
        return TestModule()
    }
    
    func testModuleDescription() {
        let m = newTestModule()
        
        let expected =
            """
            TestModule(topLevelParam=50.0) {
              child: ChildTestModule(i=10),
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
              alternateNameArray: [
                parameters(array([200, 300], dtype=int64)),
                parameters(array([400, 500], dtype=int64))
              ],
              alternateNameDict: [
                key: parameters(array([1, 2], dtype=int64))
              ],
              child: [
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
            case ("p", .parameters): break
            case ("child", .module): break
            case ("d", .dictionary): break
            case ("topLevelParam", .other): break
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
            case ("child", .dictionary): break
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
            case ("child", .value): break
            case ("d", .dictionary): break
            default:
                XCTFail("Unexpected: \(k) = \(v)))")
            }
        }
    }

}
