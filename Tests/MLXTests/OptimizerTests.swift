// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXOptimizers

class OptimizerTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    class ShapeModule: Module {
        // ranks 1 through 3: some optimizers (Adafactor, Muon) treat them differently
        let first = [MLXArray.zeros([10]), MLXArray.zeros([1])]
        let second = MLXArray.zeros([1])
        let matrix = MLXArray.zeros([3, 5])
        let tensor = MLXArray.zeros([2, 3, 4])
    }

    func checkShape<T>(optimizer: OptimizerBase<T>, steps: Int = 2) {
        let model = ShapeModule()
        let params = model.parameters()
        let grads = params.mapValues { MLXArray.ones(like: $0) }

        // note: more than one step, and using the optimizer that was passed in --
        // this used to build its own SGD, so every caller was really testing SGD
        var update = params
        for _ in 0 ..< steps {
            update = optimizer.apply(gradients: grads, modelParameters: update)
        }
        eval(update)

        let shapesEqual = params.mapValues(update) { (e1, e2) -> Bool in
            e1.shape == e2!.shape
        }.allSatisfy {
            switch $0.value {
            case .value(let b): b
            default: true
            }
        }

        XCTAssertTrue(shapesEqual)
    }

    // A very simple model that implements the equation
    // for a linear function: y = mx + b.  This can be trained
    // to match data -- in this case an unknown (to the model)
    // linear function.
    //
    // This is a nice example because most people know how
    // linear functions work and we can see how the slope
    // and intercept converge.
    class LinearFunctionModel: Module, UnaryLayer {
        let m = MLXRandom.uniform(low: -5.0, high: 5.0)
        let b = MLXRandom.uniform(low: -5.0, high: 5.0)

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            m * x + b
        }
    }

    func checkTrain<T>(optimizer: OptimizerBase<T>, compile: Bool = false) {

        // measure the distance from the prediction (model(x)) and the
        // ground truth (y).  this gives feedback on how close the
        // prediction is from matching the truth
        func loss(model: LinearFunctionModel, x: MLXArray, y: MLXArray) -> MLXArray {
            mseLoss(predictions: model(x), targets: y, reduction: .mean)
        }

        let model = LinearFunctionModel()
        eval(model)

        // the optimizer will use the gradients update the model parameters
        let optimizer = SGD(learningRate: 1e-1)

        // these are the target parameters
        let m = 0.25
        let b = 7

        let lg = valueAndGrad(model: model, loss)

        func step(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
            let (loss, grads) = lg(model, x, y)
            optimizer.update(model: model, gradients: grads)
            return loss
        }

        let resolvedStep =
            compile
            ? MLX.compile(inputs: [model, optimizer], outputs: [model, optimizer], step) : step

        // run a number of epochs
        var lastLoss: MLXArray!
        for _ in 0 ..< 30 {
            // print("target: b = \(b), m = \(m)")
            // print("parameters: \(model.parameters())")

            // generate random training data along with the ground truth.
            // notice that the shape is [B, 1] where B is the batch
            // dimension -- this allows us to train on 10 samples simultaneously
            let x = MLXRandom.uniform(low: -5.0, high: 5.0, [10, 1])
            let y = m * x + b
            eval(x, y)

            // compute the loss and gradients.  use the optimizer
            // to adjust the parameters closer to the target
            let loss = resolvedStep(x, y)

            eval(model, optimizer)

            lastLoss = loss
        }

        // it should reach this loss
        XCTAssertLessThan(lastLoss.item(Float.self), 0.1)

        print("final loss: \(lastLoss!)")
    }

    // MARK: - Integration Tests
    //
    // integration tests:
    // - verify shapes match input (sort of a smoke test, copied from python)
    // - verify that the otpimizer actually converges for a simple model

    func testSGD() {
        checkShape(optimizer: SGD(learningRate: 0.1))
        checkTrain(optimizer: SGD(learningRate: 0.1))
        checkTrain(optimizer: SGD(learningRate: 0.1), compile: true)
    }

    func testRMSprop() {
        checkShape(optimizer: RMSprop(learningRate: 0.1))
        checkTrain(optimizer: RMSprop(learningRate: 0.1))
        checkTrain(optimizer: RMSprop(learningRate: 0.1), compile: true)
    }

    func testAdaGrad() {
        checkShape(optimizer: AdaGrad(learningRate: 0.1))
        checkTrain(optimizer: AdaGrad(learningRate: 0.1))
        checkTrain(optimizer: AdaGrad(learningRate: 0.1), compile: true)
    }

    func testAdaDelta() {
        checkShape(optimizer: AdaDelta(learningRate: 0.1))
        checkTrain(optimizer: AdaDelta(learningRate: 0.1))
        checkTrain(optimizer: AdaDelta(learningRate: 0.1), compile: true)
    }

    func testAdam() {
        checkShape(optimizer: Adam(learningRate: 0.1))
        checkTrain(optimizer: Adam(learningRate: 0.1))
        checkTrain(optimizer: Adam(learningRate: 0.1), compile: true)
    }

    func testAdamBiasCorrection() {
        let parameter = MLXArray([1.0 as Float, -2.0, 3.0])
        let gradient = MLXArray([0.5 as Float, -0.25, 2.0])
        let optimizer = Adam(learningRate: 0.1 as Float, biasCorrection: true)

        let result = optimizer.applySingle(
            gradient: gradient, parameter: parameter,
            state: optimizer.newState(parameter: parameter))

        let step = MLXArray(1.0 as Float)
        let c1 = Float(0.1) / (1 - pow(Float(0.9), step))
        let c2 = rsqrt(1 - pow(Float(0.999), step))
        let m = (1 - Float(0.9)) * gradient
        let v = (1 - Float(0.999)) * square(gradient)
        let expected = parameter - (c1 * m) / (sqrt(v) * c2 + Float(1e-8))
        assertEqual(result.0, expected, atol: 1e-6)
    }

    func testAdamW() {
        checkShape(optimizer: AdamW(learningRate: 0.1))
        checkTrain(optimizer: AdamW(learningRate: 0.1))
        checkTrain(optimizer: AdamW(learningRate: 0.1), compile: true)
    }

    func testAdamWBiasCorrection() {
        let parameter = MLXArray([1.0 as Float, -2.0, 3.0])
        let gradient = MLXArray([0.5 as Float, -0.25, 2.0])
        let optimizer = AdamW(
            learningRate: 0.1 as Float, weightDecay: 0.01 as Float, biasCorrection: true)

        let result = optimizer.applySingle(
            gradient: gradient, parameter: parameter,
            state: optimizer.newState(parameter: parameter))

        let decayed = parameter * (1 - Float(0.1) * Float(0.01))
        let step = MLXArray(1.0 as Float)
        let c1 = Float(0.1) / (1 - pow(Float(0.9), step))
        let c2 = rsqrt(1 - pow(Float(0.999), step))
        let m = (1 - Float(0.9)) * gradient
        let v = (1 - Float(0.999)) * square(gradient)
        let expected = decayed - (c1 * m) / (sqrt(v) * c2 + Float(1e-8))
        assertEqual(result.0, expected, atol: 1e-6)
    }

    func testAdamax() {
        checkShape(optimizer: Adamax(learningRate: 0.1))
        checkTrain(optimizer: Adamax(learningRate: 0.1))
        checkTrain(optimizer: Adamax(learningRate: 0.1), compile: true)
    }

    func testLion() {
        checkShape(optimizer: Lion(learningRate: 0.1))
        checkTrain(optimizer: Lion(learningRate: 0.1))
        checkTrain(optimizer: Lion(learningRate: 0.1), compile: true)
    }

    func testAdafactor() {
        checkShape(optimizer: Adafactor(learningRate: 0.1))
        checkTrain(optimizer: Adafactor(learningRate: 0.1))
        checkTrain(optimizer: Adafactor(learningRate: 0.1), compile: true)
    }

    // MARK: - Muon

    func testMuonUpdateIsConditioned() {
        // the Newton-Schulz iteration used by Muon is a *crude* polar approximation:
        // it pulls the singular values of the update into a band around 1 (it does
        // not converge to an exactly orthogonal matrix).  With no momentum and
        // nesterov off the update direction is the orthogonalized gradient, so the
        // singular values of the update should be far better conditioned than the
        // gradient's.
        let optimizer = Muon(
            learningRate: 1.0, momentum: 0.0, weightDecay: 0.0, nesterov: false, nsSteps: 5)

        // full rank and deterministic: `arange` reshaped is rank 2, which the
        // iteration cannot orthogonalize
        let values = (0 ..< 24).map { Float(sin(Double($0) * 1.7) + 0.1 * Double($0 % 5)) }
        let parameter = MLXArray(values, [6, 4])
        let parameters = ModuleParameters.unflattened([("w", parameter)])

        let updated = optimizer.apply(gradients: parameters, modelParameters: parameters)

        // parameter - learningRate * scale * update, with scale = sqrt(max(1, 6/4))
        let scale = Float((6.0 / 4.0).squareRoot())
        let update = (parameter - updated[unwrapping: "w"]!) / scale

        // singular values via the gram matrix, which avoids picking between the svd
        // overloads: eigvalsh returns them ascending
        func singularValues(_ array: MLXArray) -> MLXArray {
            MLX.sqrt(
                MLX.maximum(0, MLX.eigvalsh(matmul(array.T, array), stream: .cpu)))
        }

        let inputSingular = singularValues(parameter)
        let updateSingular = singularValues(update)

        let inputCondition =
            inputSingular.max().item(Float.self) / inputSingular.min().item(Float.self)
        let updateCondition =
            updateSingular.max().item(Float.self) / updateSingular.min().item(Float.self)

        XCTAssertGreaterThan(inputCondition, 10)
        XCTAssertLessThan(updateCondition, 2)
        // and the values are pulled toward 1 rather than being rescaled arbitrarily
        XCTAssertGreaterThan(updateSingular.min().item(Float.self), 0.5)
        XCTAssertLessThan(updateSingular.max().item(Float.self), 1.5)
    }

    func testMuonLeavesLowRankParametersAlone() {
        // rank 0/1 parameters skip the orthogonalization: the update is the plain
        // momentum direction, so a single step moves by learningRate * gradient
        let optimizer = Muon(
            learningRate: 0.1, momentum: 0.0, weightDecay: 0.0, nesterov: false, nsSteps: 5)

        let parameter = MLXArray(converting: [1.0, 2.0, 3.0])
        let gradients = ModuleParameters.unflattened([("b", MLXArray.ones([3]))])
        let updated = optimizer.apply(
            gradients: gradients,
            modelParameters: ModuleParameters.unflattened([("b", parameter)]))

        assertEqual(
            updated[unwrapping: "b"]!, MLXArray(converting: [0.9, 1.9, 2.9]), atol: 1e-6)
    }

    // MARK: - defaults
    //
    // these have to match python: `tools/audit_defaults.py` compares them
    // automatically, and this is the regression test for the one that was wrong

    func testLionDefaults() {
        let lion = Lion(learningRate: 0.1)
        XCTAssertEqual(lion.betas.0, 0.9)
        // python's Lion uses 0.99 here -- Adam is the one with 0.999
        XCTAssertEqual(lion.betas.1, 0.99)
        XCTAssertEqual(lion.weightDecay, 0)
    }

    func testAdamAndAdamWDefaults() {
        XCTAssertEqual(Adam(learningRate: 0.1).betas.1, 0.999)
        XCTAssertEqual(AdamW(learningRate: 0.1).betas.1, 0.999)
        XCTAssertEqual(AdamW(learningRate: 0.1).weightDecay, 0.01)
    }

    func testMuonDefaults() {
        let muon = Muon(learningRate: 0.1)
        XCTAssertEqual(muon.momentum, 0.95)
        XCTAssertEqual(muon.weightDecay, 0.01)
        XCTAssertTrue(muon.nesterov)
        XCTAssertEqual(muon.nsSteps, 5)
    }

    // MARK: - Adafactor
    //
    // Adafactor keeps *factored* state for parameters of rank 2 and above and
    // unfactored state below that, so its state and its update path have to agree
    // about the rank -- rank 3+ was broken (an outer product via matmul, which only
    // works for rank 2) and the mismatch then crashed on a nil state.

    func testAdafactorRanks() throws {
        let optimizer = Adafactor(learningRate: 0.1, relativeStep: false)

        var parameters = ModuleParameters.unflattened([
            ("vector", MLXArray.ones([7])),
            ("matrix", MLXArray.ones([3, 5])),
            ("tensor", MLXArray.ones([2, 3, 4])),
            ("big", MLXArray.ones([2, 3, 4, 5])),
        ])

        for _ in 0 ..< 3 {
            let gradients = parameters.mapValues { 0.5 * $0 }
            parameters = optimizer.apply(gradients: gradients, modelParameters: parameters)
            eval(parameters)
        }

        for (key, shape) in [
            ("vector", [7]), ("matrix", [3, 5]), ("tensor", [2, 3, 4]), ("big", [2, 3, 4, 5]),
        ] {
            let value = parameters[unwrapping: key]!
            XCTAssertEqual(value.shape, shape, key)
            XCTAssertTrue(isFinite(value).all().item(Bool.self), "\(key) is not finite")
            // the update should have moved the parameter off its starting value
            XCTAssertFalse(
                value.allClose(MLXArray.ones(like: value), rtol: 1e-6).item(Bool.self),
                "\(key) did not change")
        }
    }

    func testAdafactorFactoredUpdateIsBatched() {
        // the row/column factors combine per leading index: slice `i` of a rank 3
        // update must equal the rank 2 update of slice `i` of the factors
        let optimizer = Adafactor(learningRate: 0.1, relativeStep: false)

        let row = MLXArray(converting: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        let column = MLXArray(converting: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], [2, 4])

        let combined = optimizer.approvateExpMovingAverage(
            expAvgSqRow: row, expAvgSqCol: column)
        XCTAssertEqual(combined.shape, [2, 3, 4])

        for i in 0 ..< 2 {
            let slice = optimizer.approvateExpMovingAverage(
                expAvgSqRow: row[i], expAvgSqCol: column[i])
            assertEqual(combined[i], slice, atol: 1e-6)
        }
    }

    func testAdafactorRankChangeDoesNotCrash() {
        // the state is created for the rank seen on the first step; a later call with
        // a different rank for the same key used to hit a nil state
        let optimizer = Adafactor(learningRate: 0.1, relativeStep: false)

        var matrix = ModuleParameters.unflattened([("w", MLXArray.ones([3, 4]))])
        matrix = optimizer.apply(
            gradients: matrix.mapValues { 0.5 * $0 }, modelParameters: matrix)

        var vector = ModuleParameters.unflattened([("w", MLXArray.ones([5]))])
        vector = optimizer.apply(
            gradients: vector.mapValues { 0.5 * $0 }, modelParameters: vector)
        eval(vector)

        XCTAssertEqual(vector[unwrapping: "w"]!.shape, [5])
    }

    class TwoParameterModel: Module {
        let weight = MLXArray.zeros([3])
        let bias = MLXArray.zeros([3])
    }

    func testMultiOptimizer() {
        let model = TwoParameterModel()
        let grads = model.parameters().mapValues { MLXArray.ones(like: $0) }

        // Route `bias` to a high learning-rate SGD; every other parameter falls back to the
        // low learning-rate SGD. Distinct results prove each parameter was routed correctly.
        let optimizer = MultiOptimizer(
            optimizers: [SGD(learningRate: 1.0), SGD(learningRate: 0.1)],
            filters: [{ key, _ in key == "bias" }])

        optimizer.update(model: model, gradients: grads)
        eval(model)

        // plain SGD from zeros with unit gradients: new = -learningRate
        assertEqual(model.bias, MLXArray([Float(-1.0), -1.0, -1.0]), atol: 1e-6)
        assertEqual(model.weight, MLXArray([Float(-0.1), -0.1, -0.1]), atol: 1e-6)

        // innerState surfaces the sub-optimizers' state.
        XCTAssertEqual(optimizer.innerState().count, 2)
    }

    func testMuon() {
        checkShape(optimizer: Muon(learningRate: 0.1))

        // 1D (and 0D) parameters skip the Newton-Schulz orthogonalization and
        // use a plain momentum/Nesterov update — pin that math exactly.
        // v = 0.95*0 + 0.05*g = 0.05*g; nesterov update = g*(1-m) + v*m;
        // param' = param - lr*update.
        let opt = Muon(learningRate: 0.1, momentum: 0.95, weightDecay: 0)
        let (p, v) = opt.applySingle(
            gradient: MLXArray([1.0, 1.0] as [Float]),
            parameter: MLXArray([1.0, 2.0] as [Float]),
            state: MLXArray([0.0, 0.0] as [Float]))
        eval(p, v)
        // update = 1*0.05 + 0.05*0.95 = 0.0975 ; param0' = 1 - 0.1*0.0975
        XCTAssertEqual(p[0].item(Float.self), 0.99025, accuracy: 1e-5)
        XCTAssertEqual(p[1].item(Float.self), 1.99025, accuracy: 1e-5)
        XCTAssertEqual(v[0].item(Float.self), 0.05, accuracy: 1e-6)

        // 2D parameters take the orthogonalization path: it must run, preserve
        // shape, and produce finite values that move the parameter.
        let g2 = MLXArray(converting: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        let (p2, _) = opt.applySingle(
            gradient: g2, parameter: MLXArray.zeros([2, 3]), state: MLXArray.zeros([2, 3]))
        eval(p2)
        XCTAssertEqual(p2.shape, [2, 3])
        XCTAssertTrue(p2.sum().item(Float.self).isFinite)
        XCTAssertGreaterThan(abs(p2).sum().item(Float.self), 0)
    }

}
