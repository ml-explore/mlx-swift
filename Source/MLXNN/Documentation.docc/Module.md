# ``Module``

## Topics

### Parameters

- ``Module/apply(filter:map:)``
- ``Module/filterMap(filter:map:isLeaf:)``
- ``Module/mapParameters(map:isLeaf:)``
- ``Module/parameters()``
- ``Module/trainableParameters()``
- ``Module/update(parameters:)``
- ``Module/update(parameters:verify:)``

### Layers (sub-modules)

- ``Module/children()``
- ``Module/filterMap(filter:map:isLeaf:)``
- ``Module/leafModules()``
- ``Module/modules()``
- ``Module/namedModules()``
- ``Module/update(modules:)``
- ``Module/update(modules:verify:)``
- ``Module/visit(modules:)``

### Traversal

- ``Module/children()``
- ``Module/filterMap(filter:map:isLeaf:)``
- ``Module/leafModules()``
- ``Module/items()``
- ``Module/visit(modules:)``

### Module and Parameter Filtering

- <doc:module-filters>

### Training

- ``Module/freeze(recursive:keys:)``
- ``Module/freeze(recursive:keys:strict:)``
- ``Module/train(_:)``
- ``Module/unfreeze(recursive:keys:)``
- ``Module/unfreeze(recursive:keys:strict:)``

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

