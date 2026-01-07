// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXNN

class ModuleTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

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
                  parameters(array([200, 300], dtype=int32)),
                  parameters(array([400, 500], dtype=int32))
                ],
                d: [
                  key: parameters(array([1, 2], dtype=int32))
                ],
                i: other(10),
                p: parameters(array([100, 200], dtype=int32))
              ],
              alternateNameArray: [
                parameters(array([200, 300], dtype=int32)),
                parameters(array([400, 500], dtype=int32))
              ],
              alternateNameDict: [
                key: parameters(array([1, 2], dtype=int32))
              ],
              d: [
                a: [
                  a: [
                    parameters(array([200, 300], dtype=int32)),
                    parameters(array([400, 500], dtype=int32))
                  ],
                  d: [
                    key: parameters(array([1, 2], dtype=int32))
                  ],
                  i: other(10),
                  p: parameters(array([100, 200], dtype=int32))
                ],
                b: [
                  a: [
                    parameters(array([200, 300], dtype=int32)),
                    parameters(array([400, 500], dtype=int32))
                  ],
                  d: [
                    key: parameters(array([1, 2], dtype=int32))
                  ],
                  i: other(10),
                  p: parameters(array([100, 200], dtype=int32))
                ]
              ],
              p: parameters(array([100, 200], dtype=int32)),
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

    func testTupleParameters() {
        // make sure tuples of MLXArray work
        class Test: Module {
            let tuple: (MLXArray, MLXArray)
            let array: [MLXArray]

            override init() {
                tuple = (MLXArray(0), MLXArray(1))
                array = [MLXArray(2), MLXArray(3)]
            }
        }

        let t = Test()
        let p = t.parameters()

        XCTAssertEqual(p.count, 2)
        switch p["array"] {
        case .array(let items):
            XCTAssertEqual(items.count, 2)
        default:
            XCTFail("should be a .array with 2 MLXArray: \(String(describing: p["item"]))")
        }
        switch p["tuple"] {
        case .array(let items):
            XCTAssertEqual(items.count, 2)
        default:
            XCTFail("should be a .array with 2 MLXArray: \(String(describing: p["item"]))")
        }

        // update the second item on the array
        var u1 = ModuleParameters()
        u1["array"] = .array([.none, .value(MLXArray(7))])
        t.update(parameters: u1)

        // make sure it was written
        if let v1 = t.parameters()["array"]?.flattenedValues() {
            XCTAssertEqual(v1.count, 2)
            XCTAssertEqual(v1[1].item(Int.self), 7)
        } else {
            XCTFail("unable to read array")
        }

        // update the second item on the tuple
        var u2 = ModuleParameters()
        u2["tuple"] = .array([.none, .value(MLXArray(11))])
        t.update(parameters: u2)

        // make sure it was written
        if let v2 = t.parameters()["tuple"]?.flattenedValues() {
            XCTAssertEqual(v2.count, 2)
            XCTAssertEqual(v2[1].item(Int.self), 11)
        } else {
            XCTFail("unable to read tuple")
        }
    }

    func testTupleModules() {
        class Layer1: Module {
            let x: MLXArray

            init(_ v: Int = 0) {
                self.x = MLXArray(v)
            }
        }
        class Layer2: Module {
            let y: MLXArray

            init(_ v: Int = 1) {
                self.y = MLXArray(v)
            }
        }

        class Test: Module {
            var tuple: (Layer1, Layer2)
            var array: [Module]

            override init() {
                tuple = (Layer1(), Layer2())
                array = [Layer1(), Layer2()]
            }
        }

        let t = Test()

        // top level + 2 + 2
        XCTAssertEqual(t.modules().count, 5)

        // update the second item on the array
        var u1 = ModuleParameters()
        u1["array"] = .array([
            .none,
            .dictionary(["y": .value(MLXArray(7))]),
        ])
        t.update(parameters: u1)

        // make sure it was written
        if let v1 = t.parameters()["array"]?.flattenedValues() {
            XCTAssertEqual(v1.count, 2)
            XCTAssertEqual(v1[1].item(Int.self), 7)
        } else {
            XCTFail("unable to read array")
        }

        // update the second item on the tuple
        var u2 = ModuleParameters()
        u2["tuple"] = .array([
            .none,
            .dictionary(["y": .value(MLXArray(11))]),
        ])
        t.update(parameters: u2)

        // make sure it was written
        if let v2 = t.parameters()["tuple"]?.flattenedValues() {
            XCTAssertEqual(v2.count, 2)
            XCTAssertEqual(v2[1].item(Int.self), 11)
        } else {
            XCTFail("unable to read tuple")
        }

        print(t.parameters())
        print(t.parameters().flattened())
    }

    func testOptionInfos() {
        class Layer1: Module {
            @ParameterInfo var x: MLXArray?
        }

        class Test: Module {
            @ModuleInfo var a: Layer1
            @ModuleInfo var b: Layer1?

            override init() {
                self._a.wrappedValue = Layer1()
            }
        }

        let t = Test()

        XCTAssertNil(t.b)
        XCTAssertNotNil(t.a)
        XCTAssertNil(t.a.x)
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

    func testInit() {
        // ensure that different types of vars and init work

        class Test: Module {
            @ParameterInfo var a: MLXArray
            @ParameterInfo var b: MLXArray?
            @ModuleInfo var c: Linear
            @ModuleInfo var d: Linear?

            override init() {
                _a.wrappedValue = MLXArray.zeros([10])
                _b.wrappedValue = MLXArray.zeros([10])
                _c.wrappedValue = Linear(10, 10)
                _d.wrappedValue = Linear(10, 10)
            }
        }

        let t = Test()

        // 2 + 2 * 2 (Linear)
        XCTAssertEqual(t.parameters().flattenedValues().count, 6)

        // self + 2 Linears
        XCTAssertEqual(t.modules().count, 3)
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

    func testLinearUpdateParametersAllSet() throws {
        class M: Module {
            let a: Linear
            let b: Linear

            override init() {
                self.a = Linear(1, 2)
                self.b = Linear(3, 4)
            }
        }
        let m = M()
        let weightsA = MLXArray(0 ..< 2).reshaped([2, 1])
        let weightsB = MLXArray(0 ..< 12).reshaped([4, 3])
        let biasA = MLXArray(0 ..< 2)
        let biasB = MLXArray(0 ..< 4)

        try m.update(
            parameters: .init(
                item: .dictionary([
                    "a": .dictionary(["weight": .value(weightsA), "bias": .value(biasA)]),
                    "b": .dictionary(["weight": .value(weightsB), "bias": .value(biasB)]),
                ])),
            verify: .all)
        let wA = m.a.weight.asArray(Int.self)
        XCTAssertEqual(wA, [0, 1])
        let wB = m.b.weight.asArray(Int.self)
        XCTAssertEqual(wB, Array(0 ..< 12))
        XCTAssertThrowsError(
            try m.update(
                parameters: .init(
                    item: .dictionary([
                        "a": .dictionary(["weight": .value(weightsA)]),
                        "b": .dictionary(["weight": .value(weightsB)]),
                    ])),
                verify: .all)
        ) { error in
            guard let error = error as? UpdateError,
                case .keyNotFound(let path, let modules) = error
            else {
                XCTFail("Expected to fail with UpdateError.keyNotFound, but got: \(error)")
                return
            }
            // should be a.bias or b.bias (random order as it is a dict)
            XCTAssertEqual(path.last, "bias")
            XCTAssertEqual(modules, ["M", "Linear"])
        }
    }

    func testLinearUpdateParametersVerifyAll() throws {
        let linear = Linear(1, 2, bias: false)

        XCTAssertEqual(linear.weight.shape, [2, 1])

        let correctWeights = MLXArray(0 ..< 2).reshaped([2, 1])

        try linear.update(
            parameters: .init(item: .dictionary(["weight": .value(correctWeights)])),
            verify: .all)

        XCTAssertEqual(linear.weight.shape, [2, 1])

        let transposedWeights = MLXArray(0 ..< 2).reshaped([1, 2])
        XCTAssertThrowsError(
            try linear.update(
                parameters: .init(item: .dictionary(["weight": .value(transposedWeights)])),
                verify: .all)
        ) { error in
            guard let error = error as? UpdateError,
                case .mismatchedSize(
                    let
                        path, let modules, expectedShape: let expectedShape,
                    actualShape: let actualShape) =
                    error
            else {
                XCTFail("Expected to fail with UpdateError.mismatchedSize, but got: \(error)")
                return
            }
            XCTAssertEqual(expectedShape, [2, 1])
            XCTAssertEqual(actualShape, [1, 2])
            XCTAssertEqual(path, ["weight"])
            XCTAssertEqual(
                error.errorDescription,
                "Mismatched parameter weight in Linear shape. Actual [1, 2], expected [2, 1]")
        }
    }

    func testLinearUpdateParametersVerifyNone() throws {
        let linear = Linear(1, 2, bias: false)

        XCTAssertEqual(linear.weight.shape, [2, 1])

        let transposedWeights = MLXArray(0 ..< 2).reshaped([1, 2])
        try linear.update(
            parameters: .init(item: .dictionary(["weight": .value(transposedWeights)])),
            verify: .none)

        XCTAssertEqual(
            linear.weight.shape, [1, 2],
            "In verify none mode, parameters can be updated with a different shape")
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
        quantize(model: m)

        XCTAssertTrue(m.module.child is QuantizedLinear)
        XCTAssertTrue(m.module.child.shape == (256, 256))
    }

    func testAlreadyQuantized() throws {
        // should not quantize an already quantize layer -- that would be silly
        class C: Module {
            @ModuleInfo
            var child: Linear = QuantizedLinear(256, 256)

            var other = Sigmoid()
        }
        class M: Module {
            let module = C()
        }

        let m = M()
        quantize(model: m)

        XCTAssertTrue(m.module.child is QuantizedLinear)
        XCTAssertTrue(m.module.child.shape == (256, 256))
    }

    func testQuantizePredicate() throws {
        class C: Module {
            @ModuleInfo
            var child1 = Linear(256, 256)

            @ModuleInfo
            var child2 = Linear(256, 8)

            var other = Sigmoid()
        }
        class M: Module {
            let module = C()
        }

        let m = M()
        quantize(
            model: m,
            filter: { _, layer in
                if let layer = layer as? Linear {
                    layer.weight.dim(0) > 8
                } else {
                    false
                }
            })

        XCTAssertTrue(m.module.child1 is QuantizedLinear)
        XCTAssertFalse(m.module.child2 is QuantizedLinear)
    }

    func testNilParameters() {
        class M: Module {
            @ParameterInfo var a: MLXArray?
            let b: MLXArray?

            internal init(a: MLXArray? = nil, b: MLXArray? = nil) {
                self.a = a
                self.b = b
            }
        }

        let m = M()
        let parameters = m.parameters()

        XCTAssertTrue(parameters.isEmpty)
    }

    func testNilModules() {
        class M: Module {
            @ModuleInfo var a: Linear?
            let b: Linear?

            internal init(a: Linear? = nil, b: Linear? = nil) {
                self.a = a
                self.b = b
            }
        }

        let m = M()
        let children = m.children()

        XCTAssertTrue(children.isEmpty)
    }

    func testBatchNormItems() {
        // BatchNorm is interesting because:
        // - it has optional parameters
        // - it has optional parameters with keys
        // - it mutates its parameters
        let n1 = BatchNorm(featureCount: 4)

        // switch it into training mode so it update parameters
        n1.train(true)

        // it should have all the parameters present
        XCTAssertEqual(
            Set(n1.parameters().flattened().map { $0.0 }),
            Set(["weight", "bias", "running_mean", "running_var"]))

        // only two of them are trainable
        XCTAssertEqual(
            Set(n1.trainableParameters().flattened().map { $0.0 }), Set(["weight", "bias"]))

        _ = n1(MLXArray(0 ..< 4, [1, 1, 4]))

        let parameters = n1.parameters()

        let m = parameters["running_mean"]
        switch m {
        case .value(let a):
            let a = a.asArray(Float.self)
            let e: [Float] = [0, 0.1, 0.2, 0.3]

            for item in zip(a, e) {
                XCTAssertEqual(item.0, item.1, accuracy: 0.01)
            }
        default:
            XCTFail("should be a .value(array)")
        }

        let v = parameters["running_var"]
        switch v {
        case .value(let a):
            let a = a.asArray(Float.self)
            let e: [Float] = [0.9, 0.9, 0.9, 0.9]

            for item in zip(a, e) {
                XCTAssertEqual(item.0, item.1, accuracy: 0.01)
            }
        default:
            XCTFail("should be a .value(array)")
        }
    }

    func testUpdateModulesNil() {
        // test where a tuple is updated with a nil value, e.g. when
        // a triple with two Linear modules is quantized, the third value
        // will be nil but is not nullable.  The code should copy forward
        // the previous value (no mutation)

        class PatchMerger: Module {
            @ModuleInfo var mlp: (Linear, GELU, Linear)

            override init() {
                mlp = (
                    Linear(128, 128),
                    GELU(),
                    Linear(128, 128)
                )
            }
        }

        let pm = PatchMerger()

        quantize(model: pm, groupSize: 64, bits: 8)

        XCTAssertTrue(pm.mlp.0 is QuantizedLinear)
        XCTAssertTrue(pm.mlp.2 is QuantizedLinear)
    }

    func testCompositeLayer() throws {
        // a test of making a LoRA layer as a composite -- it adapts
        // another layer but mixes their properties.  this is not
        // necessarily the way to implement LoRA (you can also use subclasses
        // like Linear/QuantizedLinear) but this verifies that it can
        // be written this way.

        class LoRA: Module, UnaryLayer {

            let adapts: UnaryLayer
            let scale: Float

            @ParameterInfo(key: "lora_a") var loraA: MLXArray
            @ParameterInfo(key: "lora_b") var loraB: MLXArray

            public init(
                adapts: UnaryLayer, inputDimensions: Int, outputDimensions: Int, rank: Int = 8,
                scale: Float = 20.0
            ) {
                self.adapts = adapts

                self.scale = scale

                let loraScale = 1 / sqrt(Float(inputDimensions))
                self._loraA.wrappedValue = MLXRandom.uniform(
                    low: -loraScale, high: loraScale, [inputDimensions, rank])
                self._loraB.wrappedValue = MLXArray.zeros([rank, outputDimensions])

                super.init()

                freeze()
            }

            public convenience init(linear: Linear, rank: Int = 8, scale: Float = 20.0) {
                let (outputDimensions, inputDimensions) = linear.shape
                self.init(
                    adapts: linear,
                    inputDimensions: inputDimensions, outputDimensions: outputDimensions,
                    rank: rank, scale: scale)
            }

            // produce a merged view of properties (flatten LoRA into adapts)
            override func items() -> ModuleItems {
                var result = adapts.items()
                for (key, value) in super.items() {
                    if key == "adapts" { continue }
                    result[key] = value
                }
                return result
            }

            // forward module updates -> adapt
            override func updateModule(key: String, _ value: Any) throws {
                try adapts.updateModule(key: key, value)
            }

            override func modules() -> [Module] {
                adapts.modules()
            }

            override func freeze(
                recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
            ) throws {
                try adapts.freeze(recursive: recursive, keys: keys, strict: strict)
            }

            override func noGrad() -> Set<String> {
                adapts.noGrad()
            }

            public func callAsFunction(_ x: MLXArray) -> MLXArray {
                let y = adapts(x)
                let z = matmul(matmul(x, self.loraA), self.loraB)
                return y + scale * z
            }
        }

        let linear = Linear(10, 20)
        let lora = LoRA(linear: linear)

        let linearProperties = linear.parameters()
        let loraProperties = lora.parameters()

        XCTAssertEqual(linearProperties.count, 2)
        XCTAssertEqual(loraProperties.count, 4)

        try lora.update(parameters: loraProperties.mapValues { $0 + 1 }, verify: .all)

        let trainable = lora.trainableParameters()
        XCTAssertEqual(trainable.count, 2)
        XCTAssertTrue(trainable["lora_a"] != nil)
    }

    func testModulesWithMLXArrayProperties() throws {
        // https://github.com/ml-explore/mlx-swift-examples/issues/218

        class SuScaledRotaryEmbedding: Module {
            let _freqs: MLXArray

            override init() {
                _freqs = MLXArray(7)
            }
        }

        let rope = SuScaledRotaryEmbedding()

        // no parameters
        XCTAssertEqual(rope.parameters().count, 0)

        // but it can see the _freqs property
        XCTAssertEqual(rope.items().count, 1)

        // this should not throw because _freqs is not considered
        try rope.update(parameters: .init(), verify: .all)
    }

    func testOptionalModuleUpdates() throws {
        class M: Module {
            @ModuleInfo(key: "a") var a: Linear = Linear(1, 1)
            @ModuleInfo(key: "b") var b: Linear?
        }

        let m = M()
        let b = Linear(1, 1)

        do {
            // Test setting a value for an optional module
            let u = ModuleChildren(values: ["b": .value(b)])
            try m.update(modules: u, verify: .noUnusedKeys)
            XCTAssertEqual(m.children().count, 2)
        }

        do {
            // Test resetting the optional module to nil
            let u = ModuleChildren(values: ["b": .none])
            try m.update(modules: u, verify: .noUnusedKeys)
            XCTAssertEqual(m.children().count, 1)
        }

        do {
            // Test that setting nil on a non-optional module throws an error
            let u = ModuleChildren(values: ["a": .none])
            XCTAssertThrowsError(
                try m.update(modules: u, verify: .noUnusedKeys)
            ) { error in
                switch error {
                case _ as UpdateError:
                    XCTAssertEqual(m.children().count, 1)
                default:
                    XCTFail("Wrong error type")
                }
            }
        }
    }

    func testSequentialQuantization() {
        let s = Sequential {
            Linear(64, 64)
            Linear(64, 64)
            Linear(64, 64)
        }

        // https://github.com/ml-explore/mlx-swift/issues/317
        // Unable to get @ModuleInfo for Sequential.layers -- must be wrapped to receive updates
        quantize(model: s)
    }
}
