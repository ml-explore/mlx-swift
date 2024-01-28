# `Module` Filter and Map Functions

Pre-built filter and map functions in `Module`.

``Module`` provides a number of pre-build filter and map functions for use in:

- ``Module/filterMap(filter:map:isLeaf:)``
- ``Module/apply(filter:map:)``
- ``Module/mapParameters(map:isLeaf:)``

See those methods for more information.

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
