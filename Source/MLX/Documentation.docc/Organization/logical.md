# Logical Operators

Evaluating logical operations on MLXArray.

``MLXArray`` has a number of logical operators, instance methods and free functions.  Single or
multiple arrayscan be combined using these operators:

```swift
let r = a > b || !(b < c)
```

These can be used for control flow, though consider <doc:lazy-evaluation> when doing this:

```swift
if (a < b).all().item() {
    ...
}
```

## Topics

### MLXArray Operators

- ``MLXArray/.!(_:)``
- ``MLXArray/.==(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.==(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.!=(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.!=(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.<(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.<(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.<=(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.<=(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.>(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.>(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.>=(_:_:)-(MLXArray,MLXArray)``
- ``MLXArray/.>=(_:_:)-(MLXArray,ScalarOrArray)``
- ``MLXArray/.&&(_:_:)``
- ``MLXArray/.||(_:_:)``

### MLXArray Logical Functions

- ``MLXArray/all(axes:keepDims:stream:)``
- ``MLXArray/any(axes:keepDims:stream:)``
- ``MLXArray/allClose(_:rtol:atol:equalNaN:stream:)``
- ``MLXArray/arrayEqual(_:equalNAN:stream:)``

### Logical Free Functions

- ``all(_:axes:keepDims:stream:)``
- ``allClose(_:_:rtol:atol:equalNaN:stream:)``
- ``any(_:axes:keepDims:stream:)``
- ``arrayEqual(_:_:equalNAN:stream:)``
- ``equal(_:_:stream:)``
- ``greater(_:_:stream:)``
- ``greaterEqual(_:_:stream:)``
- ``isClose(_:_:rtol:atol:equalNaN:stream:)``
- ``less(_:_:stream:)``
- ``lessEqual(_:_:stream:)``
- ``logicalAnd(_:_:stream:)``
- ``logicalNot(_:stream:)``
- ``logicalOr(_:_:stream:)``
- ``notEqual(_:_:stream:)``
- ``where(_:_:_:stream:)``
