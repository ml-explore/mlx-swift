# Optimizers Reference

MLXOptimizers provides gradient-based optimization algorithms for training neural networks.

## Optimizer Protocol

All optimizers conform to the `Optimizer` protocol:

```swift
public protocol Optimizer: Updatable, Evaluatable {
    func update(model: Module, gradients: ModuleParameters)
}
```

## Basic Usage

```swift
import MLX
import MLXNN
import MLXOptimizers

let model = MLP(inputDim: 784, hiddenDim: 256, outputDim: 10)
let optimizer = Adam(learningRate: 0.001)

// Define loss function
func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
    let output = model(x)
    return crossEntropy(logits: output, targets: y, reduction: .mean)
}

// valueAndGrad returns a function - call it to get results
let lossAndGradFn = valueAndGrad(model: model, loss)
let (lossValue, grads) = lossAndGradFn(model, input, target)

// Apply gradients
optimizer.update(model: model, gradients: grads)

// Evaluate both
eval(model, optimizer)
```

## SGD

Stochastic Gradient Descent with optional momentum:

```swift
let optimizer = SGD(
    learningRate: 0.01,
    momentum: 0.9,          // Momentum strength (default: 0)
    weightDecay: 0.0001,    // L2 penalty (default: 0)
    dampening: 0.0,         // Dampening for momentum (default: 0)
    nesterov: false         // Nesterov momentum (default: false)
)
```

### SGD Update Rule

Without momentum:
```
parameter = parameter - learningRate * gradient
```

With momentum:
```
v = momentum * v + gradient
parameter = parameter - learningRate * v
```

With Nesterov:
```
v = momentum * v + gradient
parameter = parameter - learningRate * (gradient + momentum * v)
```

## Adam

Adaptive Moment Estimation:

```swift
let optimizer = Adam(
    learningRate: 0.001,
    betas: (0.9, 0.999),    // (β1, β2) for moment estimates
    eps: 1e-8               // Numerical stability
)
```

### Adam Update Rule

```
m = β1 * m + (1 - β1) * gradient
v = β2 * v + (1 - β2) * gradient²
parameter = parameter - learningRate * m / (√v + eps)
```

## AdamW

Adam with decoupled weight decay:

```swift
let optimizer = AdamW(
    learningRate: 0.001,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weightDecay: 0.01       // Weight decay factor
)
```

### AdamW Update Rule

```
parameter = parameter * (1 - learningRate * weightDecay)
// Then standard Adam update
```

## RMSprop

Root Mean Square Propagation:

```swift
let optimizer = RMSprop(
    learningRate: 0.01,
    alpha: 0.99,            // Smoothing constant
    eps: 1e-8
)
```

### RMSprop Update Rule

```
v = alpha * v + (1 - alpha) * gradient²
parameter = parameter - learningRate * gradient / (√v + eps)
```

## AdaGrad

Adaptive Gradient:

```swift
let optimizer = AdaGrad(
    learningRate: 0.01,
    eps: 1e-8
)
```

### AdaGrad Update Rule

```
v = v + gradient²
parameter = parameter - learningRate * gradient / (√v + eps)
```

## AdaDelta

Adaptive learning rate with running windows:

```swift
let optimizer = AdaDelta(
    learningRate: 1.0,      // Often 1.0 for AdaDelta
    rho: 0.9,               // Decay rate
    eps: 1e-6
)
```

## Adamax

Adam variant using infinity norm:

```swift
let optimizer = Adamax(
    learningRate: 0.002,
    betas: (0.9, 0.999),
    eps: 1e-8
)
```

## Learning Rate Scheduling

MLX doesn't have built-in schedulers, but you can adjust learning rates directly:

```swift
var optimizer = Adam(learningRate: 0.001)

// Manual decay
for epoch in 0..<100 {
    if epoch == 30 {
        optimizer.learningRate *= 0.1
    }
    if epoch == 60 {
        optimizer.learningRate *= 0.1
    }
    // ... training
}

// Exponential decay
optimizer.learningRate = initialLR * pow(0.95, Float(epoch))

// Cosine annealing
let progress = Float(epoch) / Float(totalEpochs)
optimizer.learningRate = minLR + 0.5 * (maxLR - minLR) * (1 + cos(.pi * progress))
```

## Gradient Clipping

Apply gradient clipping before optimization:

```swift
// Clip by value
func clipByValue(_ grads: ModuleParameters, maxValue: Float) -> ModuleParameters {
    grads.mapValues { clip($0, min: -maxValue, max: maxValue) }
}

// Clip by norm
func clipByNorm(_ grads: ModuleParameters, maxNorm: Float) -> ModuleParameters {
    let flatGrads = grads.flattenedValues()
    // Sum squared norms of each gradient array
    var totalNormSquared = MLXArray(0.0)
    for g in flatGrads {
        totalNormSquared = totalNormSquared + sum(g * g)
    }
    let totalNorm = sqrt(totalNormSquared)
    let scale = minimum(MLXArray(maxNorm) / (totalNorm + 1e-6), MLXArray(1.0))

    return grads.mapValues { $0 * scale }
}

// Usage
let clippedGrads = clipByNorm(grads, maxNorm: 1.0)
optimizer.update(model: model, gradients: clippedGrads)
```

## Optimizer State

Optimizers maintain internal state (momentum buffers, etc.):

```swift
// Access internal state
let state = optimizer.innerState()

// Evaluate optimizer state
eval(optimizer)

// State is automatically managed - persists across steps
```

## Custom Optimizers

Subclass `OptimizerBase` to create custom optimizers:

```swift
class MyOptimizer: OptimizerBase<MLXArray> {
    var learningRate: Float

    init(learningRate: Float) {
        self.learningRate = learningRate
    }

    override func newState(parameter: MLXArray) -> MLXArray {
        MLXArray.zeros(like: parameter)
    }

    override func applySingle(
        gradient: MLXArray,
        parameter: MLXArray,
        state: MLXArray
    ) -> (MLXArray, MLXArray) {
        // Custom update rule
        let newParam = parameter - learningRate * gradient
        return (newParam, state)
    }
}
```

### With Tuple State

For optimizers needing multiple state arrays:

```swift
class MyMomentumOptimizer: OptimizerBase<TupleState> {
    override func newState(parameter: MLXArray) -> TupleState {
        TupleState(zeros: parameter)  // Two zero arrays
    }

    override func applySingle(
        gradient: MLXArray,
        parameter: MLXArray,
        state: TupleState
    ) -> (MLXArray, TupleState) {
        let (m, v) = state.values
        // ... update logic
        return (newParam, TupleState(newM, newV))
    }
}
```

## Training Loop Example

```swift
import MLX
import MLXNN
import MLXOptimizers

let model = MLP(inputDim: 784, hiddenDim: 256, outputDim: 10)
let optimizer = AdamW(learningRate: 0.001, weightDecay: 0.01)

// Training loop (compile only works with pure MLXArray functions, not models)
func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
    let logits = model(x)
    return crossEntropy(logits: logits, targets: y, reduction: .mean)
}

let lossAndGradFn = valueAndGrad(model: model, loss)

for epoch in 0..<numEpochs {
    var epochLoss: MLXArray = MLXArray(0.0)
    for (x, y) in dataLoader {
        let (lossValue, grads) = lossAndGradFn(model, x, y)
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)
        epochLoss = lossValue
    }

    // Validation
    model.train(false)
    let valLoss = validate(model)
    model.train(true)

    print("Epoch \(epoch): loss=\(epochLoss.item(Float.self)), val=\(valLoss)")
}
```
