# Using Streams

Controlling where your computations are evaluated.

## Specifying the Stream

All operations, including random number generation, take an optional `stream:`
argument. It defaults to ``Stream/defaultStream`` -- the default stream of the
default device.

For example:

```swift
// produced on the CPU
let a = uniform(0 ..< 1, [100, 100], stream: .cpu)

// produced on the GPU
let b = uniform(0 ..< 1, [100, 100], stream: .gpu)
```

The argument is a ``StreamOrDevice``. ``StreamOrDevice/cpu`` and
``StreamOrDevice/gpu`` select whichever stream is currently in effect for that
device, which the scopes below control.

## Scoping the Default Device

To avoid passing `stream:` to every call, change the default device for a
scope. ``Device/withDefaultDevice(_:_:)-17vjl`` does this without creating any
new streams:

```swift
Device.withDefaultDevice(.cpu) {
    // no stream: argument needed -- these run on the CPU
    let a = uniform(0 ..< 1, [100, 100])
    let b = uniform(0 ..< 1, [100, 100])
}
```

Inside the scope, ``Device/defaultDevice()`` returns the CPU device and
``StreamOrDevice/default`` resolves to the CPU stream. On exit, the previous
default is restored.

## Scoping a New Stream

``Stream/withNewDefaultStream(device:_:)-5bwc3`` sets the default device and
creates *new* streams for the scope:

```swift
Stream.withNewDefaultStream(device: .cpu) {
    // a new stream, private to this scope
    let a = uniform(0 ..< 1, [100, 100])
}
```

Use this when work must be independent of the enclosing scope's streams -- for
example so a ``Stream/synchronize()`` waits only on your own work. Use
``Device/withDefaultDevice(_:_:)-17vjl`` when you only want to redirect work to
a different device.

`withNewDefaultStream` replaces *both* the CPU and GPU streams for the scope,
not only the one named by `device:`. Code inside that explicitly asks for the
other device also gets a new stream:

```swift
Stream.withNewDefaultStream(device: .cpu) {
    // a new CPU stream -- and `stream: .gpu` here is also a new
    // stream, not the GPU stream from the enclosing scope
    let a = uniform(0 ..< 1, [100, 100], stream: .gpu)
}
```

This does not affect correctness: MLX inserts dependencies between streams
automatically, as described in <doc:unified-memory>.

Both calls nest arbitrarily, and both have `async` variants.

## Propagating Scopes Across Tasks

These scopes are task-local. They are inherited by `async let` and by child
tasks of a `TaskGroup`:

```swift
await Device.withDefaultDevice(.cpu) {
    await withTaskGroup(of: Void.self) { group in
        group.addTask {
            // still the CPU
            let a = uniform(0 ..< 1, [100, 100])
        }
    }
}
```

They are *not* inherited by `Task.detached` or by a `Thread`, which see the
process-wide default instead. Work started that way must establish its own
scope:

```swift
Task.detached {
    Device.withDefaultDevice(.cpu) {
        // the scope has to be re-established here
    }
}
```

## Holding a Stream Directly

When a stream must outlive a single scope, create a ``Stream`` and pass it
explicitly:

```swift
let stream = Stream(Device(.cpu))
let a = uniform(0 ..< 1, [100, 100], stream: .stream(stream))
```

Streams are pooled: the underlying resources are recycled when the `Stream` is
released, so creating them in a loop does not exhaust system resources. Prefer
the scoped calls above where they fit.

## See Also

- <doc:unified-memory>
- ``Stream``
- ``Device``
- ``StreamOrDevice``
