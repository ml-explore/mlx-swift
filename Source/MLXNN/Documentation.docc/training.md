# Training a Model

A model training loop.

The model training loop in `MLX` consists of:

- defining a model
- defining a loss function that measures the distance between _predicted_ and _expected_ values
- using the ``valueAndGrad(model:_:)-12a2c`` function to create a new function to compute the gradient
- presenting training data and expected values to the model, measuring the loss and computing the gradient
- using an optimizer to apply the gradient to the model parameters
    - see more about optimizers in [MLXOptimizers](https://ml-explore.github.io/mlx-swift/MLXOptimizers/documentation/mlxoptimizers/)
- repeat

Here is an example showing a simple model that learns a linear
function, literally _f(x) = mx + b_.  This model is simpler than
most, but it is easy to understand and see how it works.

```swift
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
```

Next we define a loss function -- there are a number of <doc:losses>
available to use.  I chose one that accepted simple `predictions` and `targets`:

```swift
// measure the distance from the prediction (model(x)) and the
// ground truth (y).  this gives feedback on how close the
// prediction is from matching the truth
func loss(model: LinearFunctionModel, x: MLXArray, y: MLXArray) -> MLXArray {
    mseLoss(predictions: model(x), targets: y, reduction: .mean)
}
```

Now we create the model, build the `lg` (loss and gradient) function
and create the optimizer.

```swift
let model = LinearFunctionModel()
eval(model)

// compute the loss and gradients
let lg = valueAndGrad(model: model, loss)

// the optimizer will use the gradients update the model parameters
let optimizer = SGD(learningRate: 1e-1)
```

We could define any `f(x)` -- I will use a simple one
that the model should be able to match very closely.

```swift
func f(_ x: MLXArray) -> MLXArray {
    // these are the target parameters
    let m = 0.25
    let b = 7

    // our actual function
    return m * x + b
}
```

Now we run the training loop for a number of epochs.  In each
epoch we produce training data (input x values) and expected values
(just evaluate `f(x)`).

From this we can evaluate the model and compute a loss and gradient.
The gradients are given to the optimizer to update the model parameters.

```swift
// run a number of epochs
for _ in 0 ..< 30 {
    print("target: b = \(b), m = \(m)")
    print("parameters: \(model.parameters())")

    // generate random training data along with the ground truth.
    // notice that the shape is [B, 1] where B is the batch
    // dimension -- this allows us to train on 10 samples simultaneously
    let x = MLXRandom.uniform(low: -5.0, high: 5.0, [10, 1])
    let y = f(x)
    eval(x, y)

    // compute the loss and gradients.  use the optimizer
    // to adjust the parameters closer to the target
    let (loss, grads) = lg(model, x, y)
    optimizer.update(model: model, gradients: grads)

    eval(model, optimizer)
}
```
