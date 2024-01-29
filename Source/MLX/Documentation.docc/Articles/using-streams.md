# Using Streams

Controlling where your computations are evaluated.

## Specifying the Stream

All operations (including random number generation) take an optional
argument `stream`. The `stream` specifies which
`Stream` the operation should run on. If the stream is unspecified then
the operation is run on the default stream of the default device:
``Stream/defaultStream(_:)``.  The `stream` can also
be a ``Device`` (e.g. `stream: .cpu`) in which case the operation is
run on the default stream of the provided device.

For example:

```swift
// produced on cpu
let a = MLXRandom.uniform([100, 100], stream: .cpu)

// produced on gpu
let b = MLXRandom.uniform([100, 100], stream: .gpu)
```

The parameter is of type ``StreamOrDevice`` and can be initialized with
a stream, device, or some defaults like ``StreamOrDevice/gpu``.

Read more in <doc:unified-memory>.
