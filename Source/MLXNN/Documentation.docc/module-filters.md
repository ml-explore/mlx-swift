# Module Filter and Map Functions

Pre-built filter and map functions in `Module`.

``Module`` provides a number of pre-build filter and map functions for use in:

- ``Module/filterMap(filter:map:isLeaf:)``
- ``Module/apply(filter:map:)``
- ``Module/mapParameters(map:isLeaf:)``

See those methods for more information.

## Examples

The `filterMap()` method has several options for controlling the traversal of
the modules, parameters and other values in the model.  Here is an example
that limits the traversal to just local parameters and produces 
a `NestedDictionary` of the shapes:

```swift
// produces NestedDictionary<String, [Int]> for the parameters attached
// directly to this module
let localParameterShapes = module.filterMap(
    filter: Module.filterLocalParameters,
    map: Module.mapParameters { $0.shape })
```

Applying a map to the entire set of parameters (though some traversal
control is possible through the optional `isLeaf`) is very easy:

```swift
let parameterShapes = module.mapParameters { $0.shape }
```

Finally, `apply()` does both a filter and an ``Module/update(parameters:)``.
This code would convert all floating point parameters to `.float16`.

```swift
layer.apply { array in
    array.dtype.isFloatingPoint ? array.asType(.float16) : array
}
```

## Topics

### Key/Value Filter Functions

Values usable as the `filter:` parameter in ``Module/filterMap(filter:map:isLeaf:)``.

- ``Module/filterAll``
- ``Module/filterLocalParameters``
- ``Module/filterOther``
- ``Module/filterTrainableParameters``
- ``Module/filterValidChild``
- ``Module/filterValidParameters``

### isLeaf Functions

Values usable as the `isLeaf:` parameter in ``Module/filterMap(filter:map:isLeaf:)``.

- ``Module/isLeafDefault``
- ``Module/isLeafModule``
- ``Module/isLeafModuleNoChildren``

### Map Functions

Functions useful for building the `map:` parameter in ``Module/filterMap(filter:map:isLeaf:)``.

- ``Module/mapModule(map:)``
- ``Module/mapOther(map:)``
- ``Module/mapParameters(map:)``
