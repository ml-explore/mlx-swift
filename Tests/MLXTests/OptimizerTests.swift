// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXNN

class OptimizerTests: XCTestCase {

    func testSGD() {
        class M: Module {
            let first = [MLXArray.zeros([10]), MLXArray.zeros([1])]
            let second = MLXArray.zeros([1])
        }

        let model = M()
        let params = model.parameters()
        let grads = params.mapValues { MLXArray.ones(like: $0) }

        let optimizer = SGD(learningRate: 0.1)
        let update = optimizer.apply(gradients: grads, model: model)
        eval(update)

        let shapesEqual = params.mapValues(update) { (e1, e2) -> Bool in
            e1.shape == e2!.shape
        }
        print(shapesEqual)

        print(update)
    }

    func testTrain() {
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

        // measure the distance from the prediction (model(x)) and the
        // ground truth (y).  this gives feedback on how close the
        // prediction is from matching the truth
        func loss(model: LinearFunctionModel, x: MLXArray, y: MLXArray) -> MLXArray {
            mseLoss(predictions: model(x), targets: y, reduction: .mean)
        }

        let model = LinearFunctionModel()
        eval(model.parameters())

        // compute the loss and gradients
        let lg = valueAndGrad(model: model, loss)

        // the optimizer will use the gradients update the model parameters
        let optimizer = SGD(learningRate: 1e-1)

        // these are the target parameters
        let m = 0.25
        let b = 7

        // run a number of epochs
        var lastLoss: MLXArray!
        for _ in 0 ..< 30 {
            print("target: b = \(b), m = \(m)")
            print("parameters: \(model.parameters())")

            // generate random training data along with the ground truth.
            // notice that the shape is [B, 1] where B is the batch
            // dimension -- this allows us to train on 10 samples simultaneously
            let x = MLXRandom.uniform(low: -5.0, high: 5.0, [10, 1])
            let y = m * x + b
            eval(x, y)

            // compute the loss and gradients.  use the optimizer
            // to adjust the parameters closer to the target
            let (loss, grads) = lg(model, x, y)
            optimizer.update(model: model, gradients: grads)

            eval(model.parameters(), optimizer.parameters())

            lastLoss = loss
        }

        // it should reach this loss
        XCTAssertLessThan(lastLoss.item(Float.self), 0.1)

        print(lastLoss!)

        // ideally this should be pretty close to the m and b values above
        print(model.parameters())
    }
}
