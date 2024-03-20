# Converting From Python

Common patterns from python and mapping `mlx` function names.

## Indexing

``MLXArray`` supports many types of indexes (see <doc:indexing>) -- some
of the python `mx.array` indexing modes are supported directly, though written
differenty.  Some of them require different calls.

Here is a mapping of some observed calls:

Python code | Swift Code
--- | ---
`array[10]` | `array[10]`
`array[-1]` | `array[-1]` -- this works on ``MLXArray`` but not swift arrays in general.
`array.shape[0]` | `array.dim(0)` or `array.shape[0]`
`array.shape[-1]` | `array.dim(-1)`
`array[1, 2, 3]` | `array[1, 2, 3]`
`array[2:8]` | `array[2 ..< 8]`
`array[:, :8, 8:]` | `array[0..., ..<8, 8...]`
`array[array2]` | `array[array2]`
`array[array2, array3]` | `array[array2, array3]` -- numpy style advanced indexing
`array[None]` | `array.expandedDimensions(axis: 0)`
`array[:, None]` | `array.expandedDimensions(axis: 1)`
`array[..., None]` | `array.expandedDimensions(axis: -1)`
`array[:, -1, :]` | `array[-1, axis: 1]`
`array[..., ::2]` | `array[stride: 2, axis: -1]`
`array[::-1]` | `array[stride: -1, axis: 0]` -- reverse first dimension of array
`array[..., ::-1]` | `array[stride: -1, axis: -1]` or `array[stride: -1]` -- reverse last dimension of array
`[array] * 4` | `Array(repeating: array, count: 4)`
`array.shape[:-1]` | `array.shape.dropLast()`

## Name Mapping

### Swift Naming

Note that the element-wise logical operations such as:

- ``MLXArray/.==(_:_:)-56m0a``
- ``MLXArray/.==(_:_:)-79hbc``

are named using the Swift convention for SIMD operations, e.g. `.==`, `.<`, etc.  These
operators produce a new ``MLXArray`` with `true`/`false` values for the elementwise comparison.

Functions and method are typically named in a similar fashion changing `snake_case` 
to `camelCase`.  A few exceptions to that rule follow swift naming for functions that have
no side effects.  For example:

- `flatten()` becomes ``flattened(_:start:end:stream:)``
- `reshape()` becomes ``reshaped(_:_:stream:)-5x3y0``
- `moveaxis()` becomes ``movedAxis(_:source:destination:stream:)``

and so on.

### mx.array methods

Here is a mapping of python `mx.array` methods to their ``MLXArray`` counterparts.  
Note: some of the symbols are not linkable.

`mx.array` method | ``MLXArray`` method
--- | ---
`__init__` | see <doc:initialization>
`__repr__` | ``MLXArray/description``
`__eq__` | ``MLXArray/.==(_:_:)-56m0a``
`size` | ``MLXArray/size``
`ndim` | ``MLXArray/ndim``
`itemsize` | ``MLXArray/itemSize``
`nbytes` | ``MLXArray/nbytes``
`shape` | ``MLXArray/shape``
`dtype` | ``MLXArray/dtype``
`item` | ``MLXArray/item(_:)``
`tolist` | ``MLXArray/asArray(_:)``
`astype` | ``MLXArray/asType(_:stream:)-4eqoc`` or ``MLXArray/asType(_:stream:)-6d44y``
`__getitem__ [1]` | ``MLXArray/subscript(_:stream:)-od5g``
`"__getitem__ [1, 2]"` | ``MLXArray/subscript(_:stream:)-7n5nw``
`__getitem__ [1:3]` | ``MLXArray/subscript(_:stream:)-4z56f``
`__getitem__ [mlx.array]` | ``MLXArray/subscript(_:stream:)-82jwt``
`__len__` | ``MLXArray/count``
`__iter__` | implements `Sequence`
`__add__` | ``MLXArray/+(_:_:)-1rv98``
`__sub__` | `-`
`__mul__` | ``MLXArray/*(_:_:)-1z2ck``
`__truediv__` | `/`
`__div__` | `/`
`__floordiv__` | ``MLXArray/floorDivide(_:stream:)``
`__mod__` | ``MLXArray/%(_:_:)-3ubwd``
`__eq__` | ``MLXArray/.==(_:_:)-56m0a``
`__lt__` | ``MLXArray/.<(_:_:)-9rzup``
`__le__` | ``MLXArray/.<=(_:_:)-2a0s9``
`__gt__` | ``MLXArray/.>(_:_:)-fwi1``
`__ge__` | ``MLXArray/.>=(_:_:)-2gqml``
`__ne__` | ``MLXArray/.!=(_:_:)-mbw0``
`__neg__` | ``MLXArray/-(_:)``
`__bool__` | ``MLXArray/all(keepDims:stream:)`` + ``MLXArray/item()``
`__repr__` | ``MLXArray/description``
`__matmul__` | ``MLXArray/matmul(_:stream:)``
`__pow__` | ``MLXArray/**(_:_:)-8xxt3``
`abs` | ``MLXArray/abs(stream:)``
`all` | ``MLXArray/all(axes:keepDims:stream:)``
`any` | ``MLXArray/any(axes:keepDims:stream:)``
`argmax` | ``MLXArray/argMax(axis:keepDims:stream:)``
`argmin` | ``MLXArray/argMin(axis:keepDims:stream:)``
`cos` | ``MLXArray/cos(stream:)``
`cummax` | ``MLXArray/cummax(axis:reverse:inclusive:stream:)``
`cummin` | ``MLXArray/cummin(axis:reverse:inclusive:stream:)``
`cumprod` | ``MLXArray/cumprod(axis:reverse:inclusive:stream:)``
`cumsum` | ``MLXArray/cumsum(axis:reverse:inclusive:stream:)``
`exp` | ``MLXArray/exp(stream:)``
`flatten` | ``MLXArray/flattened(start:end:stream:)``
`log` | ``MLXArray/log(stream:)``
`log10` | ``MLXArray/log10(stream:)``
`log1p` | ``MLXArray/log1p(stream:)``
`log2` | ``MLXArray/log2(stream:)``
`logsumexp` | ``MLXArray/logSumExp(axes:keepDims:stream:)``
`max` | ``MLXArray/max(axes:keepDims:stream:)``
`mean` | ``MLXArray/mean(axes:keepDims:stream:)``
`min` | ``MLXArray/min(axes:keepDims:stream:)``
`moveaxis` | ``MLXArray/movedAxis(source:destination:stream:)``
`prod` | ``MLXArray/product(axes:keepDims:stream:)``
`reciprocal` | ``MLXArray/reciprocal(stream:)``
`reshape` | ``MLXArray/reshaped(_:stream:)-67a89``
`round` | ``MLXArray/round(decimals:stream:)``
`rsqrt` | ``MLXArray/rsqrt(stream:)``
`sin` | ``MLXArray/sin(stream:)``
`split` | ``MLXArray/split(parts:axis:stream:)``
`sqrt` | ``MLXArray/sqrt(stream:)``
`square` | ``MLXArray/square(stream:)``
`squeeze` | ``MLXArray/squeezed(axes:stream:)``
`sum` | ``MLXArray/sum(axes:keepDims:stream:)``
`swapaxes` | ``MLXArray/swappedAxes(_:_:stream:)``
`T` | ``MLXArray/T``
`transpose` | ``MLXArray/transposed(_:stream:)``
`var` | ``MLXArray/variance(axes:keepDims:ddof:stream:)``

### mx free functions

This is a mapping of `mx` free functions to their ``MLX`` counterparts.

`mx.array` free function | ``MLX`` free function
--- | ---
`abs` | ``MLX/abs(_:stream:)``
`add` | ``MLX/add(_:_:stream:)``
`all` | ``MLX/all(_:axes:keepDims:stream:)``
`allclose` | ``MLX/allClose(_:_:rtol:atol:equalNaN:stream:)``
`any` | ``MLX/any(_:axes:keepDims:stream:)``
`arange` | ``MLXArray/init(_:_:)-4n0or``
`arccos` | ``MLX/acos(_:stream:)``
`arccosh` | ``MLX/acosh(_:stream:)``
`arcsin` | ``MLX/asin(_:stream:)``
`arcsinh` | ``MLX/asinh(_:stream:)``
`arctan` | ``MLX/atan(_:stream:)``
`arctanh` | ``MLX/atanh(_:stream:)``
`argmax` | ``MLX/argMax(_:axis:keepDims:stream:)``
`argmin` | ``MLX/argMin(_:axis:keepDims:stream:)``
`argpartition` | ``MLX/argPartition(_:kth:axis:stream:)``
`argsort` | ``MLX/argSort(_:axis:stream:)``
`array_equal` | ``MLX/arrayEqual(_:_:equalNAN:stream:)``
`as_strided` | ``MLX/asStrided(_:_:strides:offset:stream:)``
`broadcast_to` | ``MLX/broadcast(_:to:stream:)``
`ceil` | ``MLX/ceil(_:stream:)``
`clip` | ``MLX/clip(_:min:max:stream:)``
`concatenate` | ``MLX/concatenated(_:axis:stream:)``
`conv1d` | ``MLX/conv1d(_:_:stride:padding:dilation:groups:stream:)``
`conv2d` | ``MLX/conv2d(_:_:stride:padding:dilation:groups:stream:)``
`convolve` | ``MLX/convolve(_:_:mode:stream:)``
`cos` | ``MLX/cos(_:stream:)``
`cosh` | ``MLX/cosh(_:stream:)``
`cummax` | ``MLX/cummax(_:axis:reverse:inclusive:stream:)``
`cummin` | ``MLX/cummin(_:axis:reverse:inclusive:stream:)``
`cumprod` | ``MLX/cumprod(_:axis:reverse:inclusive:stream:)``
`cumsum` | ``MLX/cumsum(_:axis:reverse:inclusive:stream:)``
`dequantize` | ``MLX/dequantized(_:scales:biases:groupSize:bits:stream:)``
`divide` | ``MLX/divide(_:_:stream:)``
`equal` | ``MLX/equal(_:_:stream:)``
`erf` | ``MLX/erf(_:stream:)``
`erfinv` | ``MLX/erfInverse(_:stream:)``
`exp` | ``MLX/exp(_:stream:)``
`expand_dims` | ``MLX/expandedDimensions(_:axes:stream:)``
`eye` | ``MLXArray/eye(_:m:k:type:stream:)``
`flatten` | ``MLX/flattened(_:start:end:stream:)``
`floor` | ``MLX/floor(_:stream:)``
`floor_divide` | ``MLX/floorDivide(_:_:stream:)``
`full` | ``MLXArray/full(_:values:type:stream:)``
`greater` | ``MLX/greater(_:_:stream:)``
`greater_equal` | ``MLX/greaterEqual(_:_:stream:)``
`identity` | ``MLXArray/identity(_:type:stream:)``
`less` | ``MLX/less(_:_:stream:)``
`less_equal` | ``MLX/lessEqual(_:_:stream:)``
`linspace` | ``MLXArray/linspace(_:_:count:stream:)-92x6l``
`load` | ``MLX/loadArray(url:stream:)`` and ``MLX/loadArrays(url:stream:)``
`log` | ``MLX/log(_:stream:)``
`log10` | ``MLX/log10(_:stream:)``
`log1p` | ``MLX/log1p(_:stream:)``
`log2` | ``MLX/log2(_:stream:)``
`logaddexp` | ``MLX/logAddExp(_:_:stream:)``
`logical_not` | ``MLX/logicalNot(_:stream:)``
`logsumexp` | ``MLX/logSumExp(_:axes:keepDims:stream:)``
`matmul` | ``MLX/matmul(_:_:stream:)``
`max` | ``MLX/max(_:axes:keepDims:stream:)``
`maximum` | ``MLX/maximum(_:_:stream:)``
`mean` | ``MLX/mean(_:axes:keepDims:stream:)``
`min` | ``MLX/min(_:axes:keepDims:stream:)``
`minimum` | ``MLX/minimum(_:_:stream:)``
`moveaxis` | ``MLX/movedAxis(_:source:destination:stream:)``
`multiply` | ``MLX/multiply(_:_:stream:)``
`negative` | ``MLX/negative(_:stream:)``
`not_equal` | ``MLX/notEqual(_:_:stream:)``
`ones` | ``MLXArray/ones(_:type:stream:)``
`ones_like` | ``MLXArray/ones(like:stream:)``
`pad` | ``MLX/padded(_:widths:value:stream:)``
`partition` | ``MLX/partitioned(_:kth:axis:stream:)``
`power` | ``MLX/pow(_:_:stream:)-8ie9c``
`prod` | ``MLX/product(_:axes:keepDims:stream:)``
`quantize` | ``MLX/quantized(_:groupSize:bits:stream:)``
`quantized_matmul` | ``MLX/quantizedMatmul(_:_:scales:biases:transpose:groupSize:bits:stream:)``
`reciprocal` | ``MLX/reciprocal(_:stream:)``
`remainder` | ``MLX/remainder(_:_:stream:)``
`repeat` | ``MLX/repeated(_:count:axis:stream:)``
`reshape` | ``MLX/reshaped(_:_:stream:)-5x3y0``
`round` | ``MLX/round(_:decimals:stream:)``
`rsqrt` | ``MLX/rsqrt(_:stream:)``
`save` | ``MLX/save(array:url:stream:)`` and ``MLX/save(arrays:metadata:url:stream:)``
`save_safetensors` | ``MLX/save(arrays:metadata:url:stream:)``
`savez` | not supported
`savez_compressed` | not supported
`sigmoid` | ``MLX/sigmoid(_:stream:)``
`sign` | ``MLX/sign(_:stream:)``
`sin` | ``MLX/sin(_:stream:)``
`sinh` | ``MLX/sinh(_:stream:)``
`softmax` | ``MLX/softMax(_:axes:stream:)``
`sort` | ``MLX/sorted(_:axis:stream:)``
`split` | ``MLX/split(_:parts:axis:stream:)``
`sqrt` | ``MLX/sqrt(_:stream:)``
`square` | ``MLX/square(_:stream:)``
`squeeze` | ``MLX/squeezed(_:axes:stream:)``
`stack` | ``MLX/stacked(_:axis:stream:)``
`stop_gradient` | ``MLX/stopGradient(_:stream:)``
`subtract` | ``MLX/subtract(_:_:stream:)``
`sum` | ``MLX/sum(_:axes:keepDims:stream:)``
`swapaxes` | ``MLX/swappedAxes(_:_:_:stream:)``
`take` | ``MLX/take(_:_:axis:stream:)``
`take_along_axis` | ``MLX/takeAlong(_:_:axis:stream:)``
`tan` | ``MLX/tan(_:stream:)``
`tanh` | ``MLX/tanh(_:stream:)``
`topk` | ``MLX/top(_:k:axis:stream:)``
`transpose` | ``MLX/transposed(_:axes:stream:)``
`tri` | ``MLXArray/tri(_:m:k:type:stream:)``
`tril` | ``MLX/tril(_:k:stream:)``
`triu` | ``MLX/triu(_:k:stream:)``
`var` | ``MLX/variance(_:axes:keepDims:ddof:stream:)``
`where` | ``MLX/which(_:_:_:stream:)``
`zeros` | ``MLXArray/zeros(_:type:stream:)``
`zeros_like` | ``MLXArray/zeros(like:stream:)``
