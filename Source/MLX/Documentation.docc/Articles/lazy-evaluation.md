# Lazy Evaluation

Computation in `MLX` is lazy.  Understand when the graph is evaluated.

##

See also [mlx python docs](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

## Why Lazy Evaluation

When you perform operations in MLX, no computation actually happens. Instead a
compute graph is recorded. The actual computation only happens if an
``eval(_:)-(Collection<MLXArray>)`` is performed or an implicit eval is triggered.

MLX uses lazy evaluation because it has some nice features, some of which we
describe below. 

### Transforming Compute Graphs

Lazy evaluation lets us record a compute graph without actually doing any
computations. This is useful for function transformations like `grad` and
`vmap` and graph optimizations like `simplify`.

Currently, MLX does not compile and rerun compute graphs. They are all
generated dynamically. However, lazy evaluation makes it much easier to
integrate compilation for future performance enhancements.

### Only Compute What You Use

In MLX you do not need to worry as much about computing outputs that are never
used. For example:

```swift
func f(_ x: MLXArray) -> (MLXArray, MLXArray) {
    let a = fun1(x)
    let b = expensiveFunction(a)
    return (a, b)
}

let (y, _) = f(x)
```

Here, we never actually compute the output of `expensiveFunction`. Use this
pattern with care though, as the graph of `expensiveFunction` is still built, and
that has some cost associated to it.

Similarly, lazy evaluation can be beneficial for saving memory while keeping
code simple. Say you have a very large model `Model` derived from
`Module`. You can instantiate this model with `model = Model()`.
Typically, this will initialize all of the weights as `float32`, but the
initialization does not actually compute anything until you perform an
``eval(_:)-(Collection<MLXArray>)``. If you update the model with `float16` weights, your maximum
consumed memory will be half that required if eager computation was used
instead.

This pattern is simple to do in MLX thanks to lazy computation:

```swift
let model = Model()

let url = URL(filePath: "weights_fp16.safetensors")
let weights = loadArrays(url: url)

model.update(parameters: weights)
```

## When to Evaluate

A common question is when to use ``eval(_:)-(Collection<MLXArray>)``. The trade-off is between
letting graphs get too large and not batching enough useful work.

For example:

```swift
var a: MLXArray = ...
var b: MLXArray = ...

for _ in 0 ..< 100 {
    a = a + b
    eval(a)
    b = b * 2
    eval(b)
}
```

This is a bad idea because there is some fixed overhead with each graph
evaluation. On the other hand, there is some slight overhead which grows with
the compute graph size, so extremely large graphs (while computationally
correct) can be costly.

Luckily, a wide range of compute graph sizes work pretty well with MLX:
anything from a few tens of operations to many thousands of operations per
evaluation should be okay.

Most numerical computations have an iterative outer loop (e.g. the iteration in
stochastic gradient descent). A natural and usually efficient place to use
``eval(_:)-(Collection<MLXArray>)`` is at each iteration of this outer loop.

Here is a concrete example:

```swift
for batch in dataset {
    // Nothing has been evaluated yet
    let (loss, grad) = valueAndGrad(model, batch)

    // Still nothing has been evaluated
    optimizer.update(model, grad)

    // Evaluate the loss and the new parameters which will
    // run the full gradient computation and optimizer update
    eval(loss, model)
}
```

An important behavior to be aware of is when the graph will be implicitly
evaluated. Anytime you `print` an array, or otherwise access it's memory,
the graph will be evaluated. Saving arrays via ``save(arrays:metadata:url:stream:)`` 
(or any other MLX saving functions) will also evaluate the array.


Calling ``MLXArray/item(_:)`` on a scalar array will also evaluate it. In the
example above, printing the loss (`print(loss)`) or adding the loss scalar to
a list (`losses.append(loss.item(Float.self))`) would cause a graph evaluation. If 
these lines are before `eval(loss, model.parameters())` then this
will be a partial evaluation, computing only the forward pass.

Also, calling ``eval(_:)-(Collection<MLXArray>)`` on an array or set of arrays multiple times is
perfectly fine. This is effectively a no-op.

> Using scalar arrays for control-flow will cause an evaluation.

Here is an example:

```swift
func f(_ x: MLXArray) -> MLXArray {
    let (h, y) = firstLayer(x)

    // note: in python this is just "if y > 0:" which
    // has an implicit item() call in the boolean context
    let z: MLXArray
    if (y > 0).item() {
        z = secondLayerA(h)
    } else {
        z = secondLayerB(h)
    }
    return z
}
```

Using arrays for control flow should be done with care. The above example works
and can even be used with gradient transformations. However, this can be very
inefficient if evaluations are done too frequently.
