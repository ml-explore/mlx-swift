# Transforms Reference

MLX provides powerful function transforms for automatic differentiation, vectorization, and JIT compilation.

## Automatic Differentiation

### grad - Compute Gradients

```swift
// Gradient of a scalar function
let f: (MLXArray) -> MLXArray = { x in
    sum(x * x)
}

let gradF = grad(f)
let g = gradF(MLXArray([1.0, 2.0, 3.0]))
// g = [2.0, 4.0, 6.0]

// Gradient with respect to specific argument
let f2: (MLXArray, MLXArray) -> MLXArray = { a, b in
    sum(a * b)
}

// Gradient w.r.t. first argument (default)
let gradA = grad(f2, argumentNumbers: [0])

// Gradient w.r.t. second argument
let gradB = grad(f2, argumentNumbers: [1])

// Gradient w.r.t. both
let gradBoth = grad(f2, argumentNumbers: [0, 1])
```

### valueAndGrad - Value and Gradient Together

```swift
// Get both value and gradient in one pass
let f: (MLXArray) -> MLXArray = { x in
    sum(x * x)
}

let valueAndGradF = valueAndGrad(f)
let (value, gradient) = valueAndGradF(MLXArray([1.0, 2.0, 3.0]))
// value = 14.0
// gradient = [2.0, 4.0, 6.0]
```

### Model Gradients

For neural network training:

```swift
import MLXNN

// Using valueAndGrad with a model - it returns a function
let model: MLP = ...
let input: MLXArray = ...
let target: MLXArray = ...

// Define loss function
func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
    let output = model(x)
    return mseLoss(predictions: output, targets: y, reduction: .mean)
}

// valueAndGrad returns a function that computes both loss and gradients
let lossAndGradFn = valueAndGrad(model: model, loss)
let (lossValue, grads) = lossAndGradFn(model, input, target)

// grads is ModuleParameters matching model.parameters()
```

### jvp - Jacobian-Vector Product (Forward Mode)

```swift
// Forward mode automatic differentiation
let f: ([MLXArray]) -> [MLXArray] = { inputs in
    [sum(inputs[0] * inputs[0])]
}

let primals = [MLXArray([1.0, 2.0, 3.0])]
let tangents = [MLXArray([1.0, 1.0, 1.0])]

let (outputs, jvpOutputs) = jvp(f, primals: primals, tangents: tangents)
```

### vjp - Vector-Jacobian Product (Reverse Mode)

```swift
// Reverse mode automatic differentiation
let f: ([MLXArray]) -> [MLXArray] = { inputs in
    [sum(inputs[0] * inputs[0])]
}

let primals = [MLXArray([1.0, 2.0, 3.0])]
let cotangents = [MLXArray(1.0)]  // Seed for backward pass

let (outputs, vjpOutputs) = vjp(f, primals: primals, cotangents: cotangents)
```

## Vectorization

### vmap - Automatic Vectorization

```swift
// Vectorize a function over a batch dimension
let f: (MLXArray) -> MLXArray = { x in
    sum(x * x)
}

// Apply to batch dimension
let vmappedF = vmap(f, inAxes: [0], outAxes: [0])

let batch = MLXArray(0 ..< 12, [3, 4])  // 3 samples of size 4
let results = vmappedF([batch])
// results[0].shape = [3]  // One scalar per sample
```

### vmap with Multiple Inputs

```swift
let f: (MLXArray, MLXArray) -> MLXArray = { a, b in
    sum(a * b)
}

// Both inputs batched on axis 0
let vmappedF = vmap(f, inAxes: [0, 0], outAxes: [0])

// First input batched, second broadcasted
let vmappedF2 = vmap(f, inAxes: [0, nil], outAxes: [0])
```

## Evaluation

### eval - Force Computation

MLX uses lazy evaluation. Arrays are not computed until needed. Use `eval` to force computation:

```swift
let a = MLXArray([1, 2, 3])
let b = MLXArray([4, 5, 6])
let c = a + b  // NOT computed yet

// Force evaluation
eval(c)

// Or evaluate multiple (more efficient)
eval(a, b, c)

// Method form
c.eval()
```

### When to Use eval

```swift
// 1. Before timing measurements
let start = Date()
eval(result)
let elapsed = Date().timeIntervalSince(start)

// 2. In training loops to control memory
let lossAndGradFn = valueAndGrad(model: model, loss)
for batch in batches {
    let (lossValue, grads) = lossAndGradFn(model, batch.x, batch.y)
    optimizer.update(model: model, gradients: grads)
    eval(model, optimizer)  // Evaluate before next iteration
}

// 3. Before accessing values
let value: Float = result.item()  // Implicit eval
print(result)                      // Implicit eval
```

### Evaluating Models and Optimizers

```swift
// Models and optimizers conform to Evaluatable
eval(model)
eval(optimizer)
eval(model, optimizer)

// Inner state for custom types
let arrays = model.innerState()
eval(arrays)
```

## Compilation

### compile - JIT Compilation

```swift
// Compile a function for faster execution
let f: (MLXArray, MLXArray) -> MLXArray = { a, b in
    let x = a + b
    let y = x * x
    return sum(y)
}

let compiledF = compile(f)

// First call compiles, subsequent calls are fast
let result = compiledF(a, b)
```

### compile Limitations and Module/Optimizer State

`compile()` traces a function that takes/returns `MLXArray` values (or tuples/arrays
of them) -- it can't take a `Module` or `Optimizer` as a parameter directly. But it
CAN compile closures that capture and mutate a `Module`/`Optimizer`, as long as any
mutable state they touch is declared via `inputs:`/`outputs:`. Both `Module` and
`Optimizer` conform to `Updatable`, so they can be passed there directly:

```swift
// CORRECT: compile with pure MLXArray functions
let compiledOp = compile { (a: MLXArray, b: MLXArray) -> MLXArray in
    let x = a + b
    return sum(x * x)
}

// ALSO CORRECT: compile a full training step by capturing model/optimizer
// and declaring them as inputs/outputs so compile observes their updates
func loss(model: MyModel, x: MLXArray, y: MLXArray) -> MLXArray {
    mseLoss(predictions: model(x), targets: y, reduction: .mean)
}
let lossAndGradFn = valueAndGrad(model: model, loss)

let step = compile(inputs: [model, optimizer], outputs: [model, optimizer]) { x, y in
    let (loss, grads) = lossAndGradFn(model, x, y)
    optimizer.update(model: model, gradients: grads)
    return loss
}

// PITFALL: if you close over a Module but omit it from inputs/outputs,
// compile() bakes in its parameter values at trace time. Later in-place
// updates to the module (e.g. swapping LoRA weights) will be silently
// ignored by the compiled function until it is recompiled for another
// reason (e.g. a shape change).
let compiled = compile { (x: MLXArray) -> MLXArray in model(x) }
_ = compiled(x)
model.update(parameters: newParameters)
eval(model)
_ = compiled(x) // still uses the OLD parameters

// FIX: `inputs:` is what makes compile() re-read current state on every call;
// `outputs:` is only needed when the compiled closure itself mutates the
// state (as `optimizer.update(...)` does in the `step` example above) and
// you need that mutation written back to the real arrays.
let fixed = compile(inputs: [model]) { (x: MLXArray) -> MLXArray in model(x) }
optimizer.update(model: model, gradients: grads)
eval(model, optimizer)
```

### Compilation Options

```swift
// Global toggle to disable compilation (for debugging)
MLX.compile(enable: false)

// Shapeless compilation (recompiles on shape change)
let f = compile(shapeless: true) { ... }
```

## Stop Gradient

### Preventing Gradient Flow

```swift
// Stop gradients from flowing through an expression
let x = MLXArray([1.0, 2.0, 3.0])

let result = x * stopGradient(x)
// Gradient only flows through first x, not second
```

## Custom Gradients

### Defining Custom VJP

```swift
// Define a custom backward pass using result builder syntax
let customOp = CustomFunction {
    Forward { inputs in
        // Forward pass
        let x = inputs[0]
        return [x * x]
    }
    VJP { primals, cotangents in
        // Backward pass
        let x = primals[0]
        let g = cotangents[0]
        return [2 * x * g]
    }
}

let result = customOp([x])
```

