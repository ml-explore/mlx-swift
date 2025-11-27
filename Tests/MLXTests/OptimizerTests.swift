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
        let first = [MLXArray.zeros([10]), MLXArray.zeros([1])]
        let second = MLXArray.zeros([1])
    }

    func checkShape<T>(optimizer: OptimizerBase<T>) {
        let model = ShapeModule()
        let params = model.parameters()
        let grads = params.mapValues { MLXArray.ones(like: $0) }

        let optimizer = SGD(learningRate: 0.1)
        let update = optimizer.apply(gradients: grads, modelParameters: model.parameters())
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

    func testAdamW() {
        checkShape(optimizer: AdamW(learningRate: 0.1))
        checkTrain(optimizer: AdamW(learningRate: 0.1))
        checkTrain(optimizer: AdamW(learningRate: 0.1), compile: true)
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

}
