# ``MLXOptimizers``

Built-in optimizers.

MLX has a number of built in optimizers that are useful for training models.  Here is a
simple training loop:

```
func loss(model: Model, x: MLXArray, y: MLXArray) -> MLXArray {
    // choose the loss function
    mseLoss(predictions: model(x), targets: y, reduction: .mean)
}

// function to compute the value (loss) and gradient
let lg = valueAndGrad(model: model, loss)

let optimizer = SGD(learningRate: 1e-1)

for _ in 0 ..< epochs {
    let (x, y) = ...

    // evaluate the training data
    let (loss, grads) = lg(model, x, y)

    // use the optimizer to update the model parameters
    optimizer.update(model: model, gradients: grads)

    eval(model, optimizer)
}
```

## Checkpoints

``Optimizer/state()`` is the buffers training has accumulated (momentum, moments,
and per-parameter step counts), in the same shape as the model parameters.
``Optimizer/update(parameters:)`` writes that tree back. The learning rate and
other hyperparameters stay on the optimizer you construct:

```swift
eval(optimizer)
let state = optimizer.state()

let resumed = Adam(learningRate: 1e-3)
resumed.update(parameters: state)
```

## Other MLX Packages

- [MLX](mlx)
- [MLXNN](mlxnn)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)

## Topics

### Optimizers

- ``AdaDelta``
- ``Adafactor``
- ``AdaGrad``
- ``AdamW``
- ``Adam``
- ``Adamax``
- ``Lion``
- ``RMSprop``
- ``SGD``


### Base Classes and Protocols

- ``Optimizer``
- ``OptimizerBase``
- ``OptimizerBaseArrayState``
