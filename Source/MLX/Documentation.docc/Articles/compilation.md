# Compilation

MLX has a ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` function transformation which compiles computation
graphs. Function compilation results in smaller graphs by merging common work
and fusing certain operations. In many cases this can lead to big improvements
in run-time and memory use.

Getting started with ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` is simple, but there are 
some edge cases that are good to be aware of for more complex graphs and advanced usage.

## Basics of Compile

Let's start with a simple example:

```swift
func f(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
    exp(-x) + y
}

let x = MLXArray(1.0)
let y = MLXArray(2.0)

// regular function call, prints array(2.36788, dtype=float32)
print(f(x, y))

// compile the function
let compiled = compile(f)

// call the compiled version, prints array(2.36788, dtype=float32)
print(compiled(x, y))
```

The output of both the regular function and the compiled function is the same
up to numerical precision.
   
The first time you call a compiled function, MLX will build the compute
graph, optimize it, and generate and compile code. This can be relatively
slow. However, MLX will cache compiled functions, so calling a compiled
function multiple times will not initiate a new compilation. This means you
should typically compile functions that you plan to use more than once.

There are some important cases to be aware of that can cause a function to
be recompiled:

* Changing the shape or number of dimensions
* Changing the type of any of the inputs
* Changing the number of inputs to the function

In certain cases only some of the compilation stack will be rerun (for
example when changing the shapes) and in other cases the full compilation
stack will be rerun (for example when changing the types). In general you
should avoid compiling functions too frequently.

Another idiom to watch out for is compiling functions which get created and
destroyed frequently. This can happen, for example, when compiling a
function inside a loop -- better to hoist it to outside the loop.

## Example Speedup

The function `gelu()` is a nonlinear activation function commonly used with
Transformer-based models. The implementation involves several unary and binary
element-wise operations:

```swift
public func gelu(_ x: MLXArray) -> MLXArray {
    x * (1 + erf(x / sqrt(2))) / 2
}
```

If you use this function with small arrays, it will be overhead bound. If you
use it with large arrays it will be memory bandwidth bound.  However, all of
the operations in the `gelu` are fusible into a single kernel with
``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``. This can speedup both cases considerably.

Let's compare the runtime of the regular function versus the compiled
function. We'll use the following timing helper which does a warm up and
handles measures the execution:

```swift
func measure(_ f: (MLXArray) -> MLXArray, _ x: MLXArray) {
    // warm up
    for _ in 0 ..< 10 {
        eval(f(x))
    }
    
    let start = Date.timeIntervalSinceReferenceDate
    let iterations = 100
    for _ in 0 ..< iterations {
        eval(f(x))
    }
    let end = Date.timeIntervalSinceReferenceDate
    
    let timePerIteration = 1000.0 * (end - start) / Double(iterations)
    
    print("Time per iteration \(timePerIteration.formatted()) ms")
}
```

Now make an array, and benchmark both functions:

```swift
let x = MLXRandom.uniform(0 ..< 1, [32, 1000, 4096])

measure(gelu, x)
measure(compile(gelu), x)
```

On an M1 Max the times are 15.5 and 3.1 milliseconds. The compiled `gelu` is
five times faster.

> As of the latest MLX, CPU functions are not fully compiled. Compiling CPU
functions can still be helpful, but won't typically result in as large a
speedup as compiling operations that run on the GPU.

## Debugging

When a compiled function is first called, it is traced with placeholder
inputs. This means you can't evaluate arrays (for example to print their
contents) inside compiled functions.

```swift
func f(_ x: MLXArray) -> MLXArray {
    let z = -x

    // this will crash
    print(z)

    return exp(z)
}

let compiled = compile(f)
_ = compiled(...)
```

For debugging, inspecting arrays can be helpful. One way to do that is to
globally disable compilation using the ``compile(enable:)`` function or
`MLX_DISABLE_COMPILE` environment variable.

## Pure Functions

Compiled functions are intended to be *pure*; that is they should not have side
effects. For example:

```swift
var state = [MLXArray]()

func f(_ x: MLXArray) -> MLXArray {
    let z = x * 8
    state.append(z)
    return exp(z)
}

let compiled = compile(f)
_ = compiled(MLXArray(1.0))

// this will crash
print(state)
```

After calling the compiled version of `f()` the `state` variable will hold
a placeholder array.  The placeholder does not have any data; it is only
used to build the computation graph. Printing such an array results in a crash.

You have two options to deal with this. The first option is to simply return
`state` as an output:

```swift
var state = [MLXArray]()

func f(_ x: MLXArray) -> [MLXArray] {
    let z = x * 8
    state.append(z)
    return [exp(z), state]
}

// note: the arguments would have to be adapted -- using this form
// for example purposes only
let compiled = compile(f)
_ = compiled(MLXArray(1.0))

print(state)
```

In some cases returning updated state can be pretty inconvenient. Hence,
``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` has a parameter to capture implicit state:

```swift
var state = [MLXArray]()

func f(_ x: MLXArray) -> MLXArray {
    let z = x * 8
    state.append(z)
    return exp(z)
}

// capture state the `state` array as a side effect
let compiled = compile(outputs: [state], f)
_ = compiled(MLXArray(1.0))

print(state)
```

Note that `[MLXArray]` implements the ``Updatable`` protocol, as does ``MLXArray``,
`MLXRandom.globalState`, `Optimizer`, and `Module`.

This is particularly useful for compiling a function which includes an update
to a container of arrays, as is commonly done when training the parameters of a
`Module`.

Compiled functions will also treat any inputs not in the parameter list as
constants. For example:

```swift
func f(_ bias: MLXArray) -> MLXArray {
    MLXRandom.uniform(0 ..< 1, [4]) + bias
}

let bias = MLXArray(0)

// without capturing state this won't mutate the random state
let c1 = compile(f)

let c1a = c1(bias)
let c1b = c1(bias)
XCTAssertTrue(allClose(c1a, c1b).item())
```

The random number generation implicitly uses the global random seed, `MLXRandom.seed(_:)`.
Since this is not captured in the state, the value that is seen on the compilation run
is used without seeing any updates or making any changes to state.

To make this work as expected:

```swift
// now capture the random state and the random numbers should change per call
let c2 = compile(inputs: [MLXRandom.globalState], outputs: [MLXRandom.globalState], f)

let c2a = c2(bias)
let c2b = c2(bias)
XCTAssertFalse(allClose(c2a, c2b).item())
```

## Compiling Training Graphs 

This section will step through how to use ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` 
with a simple example of a common setup: training a model with `Module` using an
`Optimizer` with state. We will show how to compile the
full forward, backward, and update with ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``.

Here is the basic scenario:

```swift
class LinearFunctionModel: Module, UnaryLayer {
    let m = MLXRandom.uniform(low: -5.0, high: 5.0)
    let b = MLXRandom.uniform(low: -5.0, high: 5.0)

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        m * x + b
    }
}

func loss(model: LinearFunctionModel, x: MLXArray, y: MLXArray) -> MLXArray {
    mseLoss(predictions: model(x), targets: y, reduction: .mean)
}

let model = LinearFunctionModel()
eval(model)

let lg = valueAndGrad(model: model, loss)

// the optimizer will use the gradients update the model parameters
let optimizer = SGD(learningRate: 1e-1)

// these are the target parameters
let m = 0.25
let b = 7
```

To start, here is the simple example without any compilation:

```swift
for _ in 0 ..< 30 {
    // prepare the training data
    let x = MLXRandom.uniform(low: -5.0, high: 5.0, [10, 1])
    let y = m * x + b
    eval(x, y)

    // evaluate and update parameters
    let (loss, grads) = lg(model, x, y)
    optimizer.update(model: model, gradients: grads)
}
```

To compile the update we can put it all in a function and compile it with the
appropriate input and output captures. Here's the same example but compiled:

```swift
let step = compile(inputs: [model, optimizer], outputs: [model, optimizer]) { x, y in
    let (loss, grads) = lg(model, x, y)
    optimizer.update(model: model, gradients: grads)
    return loss
}

for _ in 0 ..< 30 {
    // prepare the training data
    let x = MLXRandom.uniform(low: -5.0, high: 5.0, [10, 1])
    let y = m * x + b
    eval(x, y)

    let loss = step(x, y)
}
```

> If you are using a module which performs random sampling such as
`Dropout`, make sure you also include `MLXRandom.globalState` in the
`inputs:` and `outputs:`.

## Topics

### Functions

- ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
- ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,(MLXArray)->MLXArray)``
- ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,(MLXArray,MLXArray)->MLXArray)``
- ``compile(enable:)``
