# Free Functions

Free functions in MLX.

MLX has a wide variety of free functions, categorized below.  ``MLXArray`` has some identical
operations as methods for convenience.

## Topics

### Element-wise Arithmetic Free Functions

- ``abs(_:stream:)``
- ``acos(_:stream:)``
- ``acosh(_:stream:)``
- ``add(_:_:stream:)``
- ``asin(_:stream:)``
- ``asinh(_:stream:)``
- ``atan(_:stream:)``
- ``atanh(_:stream:)``
- ``ceil(_:stream:)``
- ``clip(_:min:max:stream:)``
- ``clip(_:max:stream:)``
- ``cos(_:stream:)``
- ``cosh(_:stream:)``
- ``divide(_:_:stream:)``
- ``erf(_:stream:)``
- ``erfInverse(_:stream:)``
- ``exp(_:stream:)``
- ``expm1(_:stream:)``
- ``floor(_:stream:)``
- ``floorDivide(_:_:stream:)``
- ``log(_:stream:)``
- ``log10(_:stream:)``
- ``log1p(_:stream:)``
- ``log2(_:stream:)``
- ``logAddExp(_:_:stream:)``
- ``logicalNot(_:stream:)``
- ``matmul(_:_:stream:)``
- ``maximum(_:_:stream:)``
- ``minimum(_:_:stream:)``
- ``multiply(_:_:stream:)``
- ``negative(_:stream:)``
- ``notEqual(_:_:stream:)``
- ``pow(_:_:stream:)-(MLXArray,ScalarOrArray,_)``
- ``pow(_:_:stream:)-(ScalarOrArray,MLXArray,_)``
- ``pow(_:_:stream:)-(MLXArray,MLXArray,_)``
- ``reciprocal(_:stream:)``
- ``remainder(_:_:stream:)``
- ``round(_:decimals:stream:)``
- ``rsqrt(_:stream:)``
- ``sigmoid(_:stream:)``
- ``sign(_:stream:)``
- ``sin(_:stream:)``
- ``sinh(_:stream:)``
- ``softmax(_:precise:stream:)``
- ``softmax(_:axis:precise:stream:)``
- ``softmax(_:axes:precise:stream:)``
- ``sqrt(_:stream:)``
- ``square(_:stream:)``
- ``subtract(_:_:stream:)``
- ``tan(_:stream:)``
- ``tanh(_:stream:)``
- ``which(_:_:_:stream:)``

### Convolution

- ``conv1d(_:_:stride:padding:dilation:groups:stream:)``
- ``conv2d(_:_:stride:padding:dilation:groups:stream:)``
- ``convolve(_:_:mode:stream:)``

### Cumulative

- ``cummax(_:axis:reverse:inclusive:stream:)``
- ``cummax(_:reverse:inclusive:stream:)``
- ``cummin(_:axis:reverse:inclusive:stream:)``
- ``cummin(_:reverse:inclusive:stream:)``
- ``cumprod(_:axis:reverse:inclusive:stream:)``
- ``cumprod(_:reverse:inclusive:stream:)``
- ``cumsum(_:axis:reverse:inclusive:stream:)``
- ``cumsum(_:reverse:inclusive:stream:)``

### Indexes

- ``argMax(_:axis:keepDims:stream:)``
- ``argMax(_:keepDims:stream:)``
- ``argMin(_:axis:keepDims:stream:)``
- ``argMin(_:keepDims:stream:)``
- ``argPartition(_:kth:axis:stream:)``
- ``argPartition(_:kth:stream:)``
- ``argSort(_:axis:stream:)``
- ``argSort(_:stream:)``
- ``takeAlong(_:_:axis:stream:)``
- ``takeAlong(_:_:stream:)``
- ``take(_:_:stream:)``
- ``take(_:_:axis:stream:)``
- ``top(_:k:stream:)``
- ``top(_:k:axis:stream:)``

### Factory

- ``MLX/zeros(_:type:stream:)``
- ``MLX/zeros(like:stream:)``
- ``MLX/ones(_:type:stream:)``
- ``MLX/ones(like:stream:)``
- ``MLX/eye(_:m:k:type:stream:)``
- ``MLX/full(_:values:type:stream:)``
- ``MLX/full(_:values:stream:)``
- ``MLX/identity(_:type:stream:)``
- ``MLX/linspace(_:_:count:stream:)-6w959``
- ``MLX/repeated(_:count:axis:stream:)``
- ``MLX/repeated(_:count:stream:)``
- ``MLX/repeat(_:count:axis:stream:)``
- ``MLX/repeat(_:count:stream:)``
- ``MLX/tri(_:m:k:type:stream:)``
- ``tril(_:k:stream:)``
- ``triu(_:k:stream:)``

### I/O

- ``loadArray(url:stream:)``
- ``loadArrays(url:stream:)``
- ``loadArraysAndMetadata(url:stream:)``
- ``save(array:url:stream:)``
- ``save(arrays:metadata:url:stream:)``

### Logical

- ``all(_:axes:keepDims:stream:)``
- ``all(_:keepDims:stream:)``
- ``all(_:axis:keepDims:stream:)``
- ``allClose(_:_:rtol:atol:equalNaN:stream:)``
- ``any(_:axes:keepDims:stream:)``
- ``any(_:keepDims:stream:)``
- ``any(_:axis:keepDims:stream:)``
- ``arrayEqual(_:_:equalNAN:stream:)``
- ``equal(_:_:stream:)``
- ``greater(_:_:stream:)``
- ``greaterEqual(_:_:stream:)``
- ``less(_:_:stream:)``
- ``lessEqual(_:_:stream:)``
- ``logicalNot(_:stream:)``
- ``notEqual(_:_:stream:)``
- ``where(_:_:_:stream:)``

### Logical Reduction

- ``all(_:axes:keepDims:stream:)``
- ``any(_:axes:keepDims:stream:)``

### Aggregating Reduction

- ``logSumExp(_:axes:keepDims:stream:)``
- ``product(_:axis:keepDims:stream:)``
- ``max(_:axes:keepDims:stream:)``
- ``mean(_:axes:keepDims:stream:)``
- ``min(_:axes:keepDims:stream:)``
- ``sum(_:axes:keepDims:stream:)``
- ``variance(_:axes:keepDims:ddof:stream:)``

### Shapes

- ``asStrided(_:_:strides:offset:stream:)``
- ``broadcast(_:to:stream:)``
- ``concatenated(_:axis:stream:)``
- ``expandedDimensions(_:axes:stream:)``
- ``expandedDimensions(_:axis:stream:)``
- ``movedAxis(_:source:destination:stream:)``
- ``padded(_:width:mode:value:stream:)``
- ``padded(_:widths:mode:value:stream:)``
- ``reshaped(_:_:stream:)-(MLXArray,Collection<Int>,StreamOrDevice)``
- ``reshaped(_:_:stream:)-(MLXArray,Int...,StreamOrDevice)``
- ``split(_:indices:axis:stream:)``
- ``split(_:parts:axis:stream:)``
- ``split(_:axis:stream:)``
- ``squeezed(_:stream:)``
- ``squeezed(_:axis:stream:)``
- ``squeezed(_:axes:stream:)``
- ``stacked(_:axis:stream:)``
- ``swappedAxes(_:_:_:stream:)``
- ``transposed(_:stream:)``
- ``transposed(_:axis:stream:)``
- ``transposed(_:axes:stream:)``
- ``transposed(_:_:stream:)``
- ``T(_:stream:)``

### Sorting

- ``argSort(_:axis:stream:)``
- ``argPartition(_:kth:axis:stream:)``
- ``sorted(_:stream:)``
- ``sorted(_:axis:stream:)``
- ``partitioned(_:kth:stream:)``
- ``partitioned(_:kth:axis:stream:)``

### Quantization

- ``quantizedQuantizedMM(_:_:scales:groupSize:bits:mode:stream:)``
- ``quantized(_:groupSize:bits:mode:stream:)``
- ``quantizedMatmul(_:_:scales:biases:transpose:groupSize:bits:mode:stream:)``
- ``dequantized(_:scales:biases:groupSize:bits:mode:dtype:stream:)``

### Evaluation and Transformation

- ``eval(_:)-(MLXArray...)``
- ``eval(_:)-(Collection<MLXArray>)``
- ``eval(_:)-(Any...)``
- ``eval(_:)-(Sequence<Any>)``
- ``asyncEval(_:)-(Collection<MLXArray>)``
- ``asyncEval(_:)-(Any...)``
- ``asyncEval(_:)-(Sequence<Any>)``
- ``grad(_:)-r8dv``
- ``grad(_:)-7z6i``
- ``grad(_:argumentNumbers:)-(([MLXArray])->[MLXArray],Collection<Int>)``
- ``grad(_:argumentNumbers:)-(([MLXArray])->MLXArray,Collection<Int>)``
- ``valueAndGrad(_:)``
- ``valueAndGrad(_:argumentNumbers:)``
- ``stopGradient(_:stream:)``
- ``jvp(_:primals:tangents:)``
- ``vjp(_:primals:cotangents:)``

### Device

- ``using(device:fn:)``

### Other

- ``depends(input:dependencies:)``
- ``depends(inputs:dependencies:)``
- ``diag(_:k:stream:)``
- ``diagonal(_:offset:axis1:axis2:stream:)``
- ``einsum(_:operands:stream:)``
- ``einsum(_:_:stream:)``
- ``hadamardTransform(_:scale:stream:)``
- ``view(_:dtype:stream:)``
- ``setErrorHandler(_:data:dtor:)``
- ``fatalErrorHandler``
