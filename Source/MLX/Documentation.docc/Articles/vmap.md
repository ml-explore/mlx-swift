# Vectorization

Automatic vectorization with ``vmap(_:inAxes:outAxes:)``.

`vmap` transforms a function so that it operates independently over a batch
axis. This is convenient for evaluating a function over many inputs without
writing explicit loops.

## Basics of vmap

Let's start with a simple example:

```swift
func f(_ x: MLXArray) -> MLXArray { x * 2 }

let x = MLXArray(0 ..< 6, [3, 2])

let vf = vmap(f)
let y = vf(x)
```

This is equivalent to calling `f` on each slice of `x` along the first axis
and stacking the results:

```swift
let manual = stacked((0 ..< 3).map { f(x[$0]) }, axis: 0)
```

Both approaches produce the same array.

The `inAxes` parameter controls which axis of each input to map over. Passing
`nil` for an input disables mapping for that value. The `outAxes` parameter
specifies the axis of each output where the batched results are stacked.

```swift
func add(_ x: MLXArray, _ y: MLXArray) -> MLXArray { x + y }
let vf = vmap(add, inAxes: (0, nil))
```

Here `x` is mapped over its first axis while `y` is used as a broadcast value.

## Nested Mapping

You can nest calls to ``vmap(_:inAxes:outAxes:)`` to map over multiple axes.
Each nested `vmap` introduces another batch dimension in the result.

## Topics

### Functions

- ``vmap(_:inAxes:outAxes:)``
