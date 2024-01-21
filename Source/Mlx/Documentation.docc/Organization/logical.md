# Logical Operators

Evaluating logical operations on MLXArray.

``MLXArray`` has a number of logical operators, instance methods and free functions.  Single or
multiple arrayscan be combined using these operators:

```swift
let r = a > b || !(b < c)
```

These can be used for control flow, though consider <doc:lazy-evaluation> when doing this:

```swift
if (a < b).allTrue() {
    ...
}
```

## Topics

### MLXArray Operators

- ``MLXArray/!(_:)``
- ``MLXArray/==(_:_:)``
- ``MLXArray/!=(_:_:)``
- ``MLXArray/<(_:_:)``
- ``MLXArray/<=(_:_:)``
- ``MLXArray/>(_:_:)``
- ``MLXArray/>=(_:_:)``
- ``MLXArray/&&(_:_:)``
- ``MLXArray/||(_:_:)``

### MLXArray Logical Functions

- ``MLXArray/allTrue(stream:)``
- ``MLXArray/all(axes:keepDims:stream:)``
- ``MLXArray/any(axes:keepDims:stream:)``
- ``MLXArray/allClose(_:rtol:atol:stream:)``
- ``MLXArray/arrayEqual(_:equalNAN:stream:)``

### Logical Free Functions

- ``all(_:axes:keepDims:stream:)``
- ``allClose(_:_:rtol:atol:stream:)``
- ``allTrue(_:stream:)``
- ``any(_:axes:keepDims:stream:)``
- ``arrayEqual(_:_:equalNAN:stream:)``
- ``equal(_:_:stream:)``
- ``greater(_:_:stream:)``
- ``greaterEqual(_:_:stream:)``
- ``less(_:_:stream:)``
- ``lessEqual(_:_:stream:)``
- ``logicalNot(_:stream:)``
- ``notEqual(_:_:stream:)``
- ``where(_:_:_:stream:)``
