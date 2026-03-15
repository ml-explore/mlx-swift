# Distributed NN Layers API Reference

Complete API reference for distributed linear layers and the `sumGradients` helper.

## Architecture: Column-Parallel vs Row-Parallel Sharding

```
Column-Parallel (AllToSharded):
┌─────────────────────────────────┐
│         Input (full)            │  ← All ranks have same input
│        [batch, inDims]          │
└─────────┬───────────────────────┘
          │ sumGradients (identity fwd, allSum bwd)
          ▼
┌─────────────────────────────────┐
│  weight[outDims/N, inDims]      │  ← Each rank has slice of output features
│  matmul + bias[outDims/N]       │
└─────────┬───────────────────────┘
          ▼
┌─────────────────────────────────┐
│     Output (sharded)            │  ← Each rank has its portion
│    [batch, outDims/N]           │
└─────────────────────────────────┘

Row-Parallel (ShardedToAll):
┌─────────────────────────────────┐
│       Input (sharded)           │  ← Each rank has its portion
│      [batch, inDims/N]          │
└─────────┬───────────────────────┘
          │ matmul
          ▼
┌─────────────────────────────────┐
│  weight[outDims, inDims/N]      │  ← Each rank has slice of input features
│  partial result                 │
└─────────┬───────────────────────┘
          │ allSum (aggregate across ranks)
          ▼
┌─────────────────────────────────┐
│      Output (full)              │  ← All ranks have same output
│     [batch, outDims]            │
│     + bias[outDims]             │
└─────────────────────────────────┘
```

**Typical usage pattern:** Pair `AllToShardedLinear` with `ShardedToAllLinear` in alternating layers for tensor-parallel inference.

---

## AllToShardedLinear

Each member of the group applies part of the affine transformation such that the result is sharded across the group. Gradients are automatically aggregated from each member via `sumGradients`.

```swift
open class AllToShardedLinear: Module, UnaryLayer
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `weight` | `MLXArray` | Weight matrix of shape `[outputDimensions/N, inputDimensions]` |
| `bias` | `MLXArray?` | Bias vector of shape `[outputDimensions/N]`, or `nil` |
| `group` | `DistributedGroup` | The distributed group (excluded from `parameters()`) |

### init(inputDimensions:outputDimensions:bias:group:)

```swift
public init(
    inputDimensions: Int,
    outputDimensions: Int,
    bias: Bool = true,
    group: DistributedGroup? = nil
)
```

**Parameters:**
- `inputDimensions`: Number of input dimensions.
- `outputDimensions`: Number of output dimensions. **Must be divisible by group size.**
- `bias`: If `true`, apply a bias. Default is `true`.
- `group`: The distributed group. If `nil`, uses `MLXDistributed.init()`.

**Precondition:** `outputDimensions % group.size == 0`

Weight initialization: uniform in `[-scale, scale]` where `scale = sqrt(1.0 / inputDimensions)`.

### fromLinear(_:segments:group:)

Create an `AllToShardedLinear` from an existing `Linear` layer.

```swift
public class func fromLinear(
    _ linear: Linear, segments: Int = 1, group: DistributedGroup? = nil
) -> AllToShardedLinear
```

**Parameters:**
- `linear`: The linear layer to convert.
- `segments`: Number of segments for fused weights (e.g., 3 for QKV). Default is `1`.
- `group`: The distributed group.

**Returns:** A new `AllToShardedLinear` with sharded weights.

For a size-1 group, the sharded weights are identical to the original.

### callAsFunction(_:)

```swift
open func callAsFunction(_ x: MLXArray) -> MLXArray
```

Forward pass:
1. Apply `sumGradients(group:)` to input (identity forward, allSum backward).
2. Compute `addMM(bias, x, weight.T)` if bias exists, or `matmul(x, weight.T)` otherwise.

**Input shape:** `[batch, inputDimensions]`
**Output shape:** `[batch, outputDimensions / N]`

---

## ShardedToAllLinear

Each member of the group applies part of the affine transformation and then aggregates the results via `allSum`. All nodes will have the same exact result.

```swift
open class ShardedToAllLinear: Module, UnaryLayer
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `weight` | `MLXArray` | Weight matrix of shape `[outputDimensions, inputDimensions/N]` |
| `bias` | `MLXArray?` | Bias vector of shape `[outputDimensions]`, or `nil` |
| `group` | `DistributedGroup` | The distributed group (excluded from `parameters()`) |

### init(inputDimensions:outputDimensions:bias:group:)

```swift
public init(
    inputDimensions: Int,
    outputDimensions: Int,
    bias: Bool = true,
    group: DistributedGroup? = nil
)
```

**Parameters:**
- `inputDimensions`: Number of input dimensions. **Must be divisible by group size.**
- `outputDimensions`: Number of output dimensions.
- `bias`: If `true`, apply a bias. Default is `true`.
- `group`: The distributed group. If `nil`, uses `MLXDistributed.init()`.

**Precondition:** `inputDimensions % group.size == 0`

### fromLinear(_:segments:group:)

```swift
public class func fromLinear(
    _ linear: Linear, segments: Int = 1, group: DistributedGroup? = nil
) -> ShardedToAllLinear
```

**Parameters:**
- `linear`: The linear layer to convert.
- `segments`: Number of segments for fused weights (e.g., 3 for QKV). Default is `1`.
- `group`: The distributed group.

**Returns:** A new `ShardedToAllLinear` with sharded weights.

### callAsFunction(_:)

```swift
open func callAsFunction(_ x: MLXArray) -> MLXArray
```

Forward pass:
1. Compute `matmul(x, weight.T)`.
2. Apply `MLXDistributed.allSum(x, group: group)` to aggregate across ranks.
3. Add bias if present.

**Input shape:** `[batch, inputDimensions / N]`
**Output shape:** `[batch, outputDimensions]`

---

## QuantizedAllToShardedLinear

Quantized equivalent of `AllToShardedLinear`. Parameters are frozen and excluded from gradient computation. Conforms to `Quantized` protocol.

```swift
open class QuantizedAllToShardedLinear: Module, UnaryLayer, Quantized
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `weight` | `MLXArray` | Quantized weight matrix |
| `scales` | `MLXArray` | Quantization scale factors |
| `biases` | `MLXArray?` | Quantization bias factors (for affine mode) |
| `bias` | `MLXArray?` | Layer bias of shape `[outputDimensions/N]`, or `nil` |
| `groupSize` | `Int` | Group size for quantization |
| `bits` | `Int` | Bit width for quantization |
| `mode` | `QuantizationMode` | Quantization mode |
| `group` | `DistributedGroup` | The distributed group |

### init(inputDimensions:outputDimensions:bias:groupSize:bits:mode:group:)

```swift
public init(
    inputDimensions: Int,
    outputDimensions: Int,
    bias: Bool = true,
    groupSize: Int = 64,
    bits: Int = 4,
    mode: QuantizationMode = .affine,
    group: DistributedGroup? = nil
)
```

**Parameters:**
- `inputDimensions`: Number of input dimensions.
- `outputDimensions`: Number of output dimensions. **Must be divisible by group size.**
- `bias`: If `true`, apply a bias. Default is `true`.
- `groupSize`: The group size used for quantization. Default is `64`.
- `bits`: The bit width used for quantization. Default is `4`.
- `mode`: The quantization mode. Default is `.affine`.
- `group`: The distributed group.

**Precondition:** `outputDimensions % group.size == 0`

The layer is automatically frozen after initialization.

### fromQuantizedLinear(_:segments:group:)

```swift
public class func fromQuantizedLinear(
    _ quantizedLinear: QuantizedLinear, segments: Int = 1,
    group: DistributedGroup? = nil
) -> QuantizedAllToShardedLinear
```

### callAsFunction(_:)

Forward pass:
1. Apply `sumGradients(group:)` to input.
2. Compute `quantizedMM(x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits, mode: mode)`.
3. Add bias if present.

### unfreeze(recursive:keys:strict:)

Override that re-freezes the layer's own parameters after unfreezing. Quantized parameters cannot be trained.

```swift
public override func unfreeze(
    recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
) throws
```

---

## QuantizedShardedToAllLinear

Quantized equivalent of `ShardedToAllLinear`. Parameters are frozen and excluded from gradient computation. Conforms to `Quantized` protocol.

```swift
open class QuantizedShardedToAllLinear: Module, UnaryLayer, Quantized
```

### Properties

Same as `QuantizedAllToShardedLinear` except bias shape is `[outputDimensions]` (not sharded).

### init(inputDimensions:outputDimensions:bias:groupSize:bits:mode:group:)

```swift
public init(
    inputDimensions: Int,
    outputDimensions: Int,
    bias: Bool = true,
    groupSize: Int = 64,
    bits: Int = 4,
    mode: QuantizationMode = .affine,
    group: DistributedGroup? = nil
)
```

**Precondition:** `inputDimensions % group.size == 0`

### fromQuantizedLinear(_:segments:group:)

```swift
public class func fromQuantizedLinear(
    _ quantizedLinear: QuantizedLinear, segments: Int = 1,
    group: DistributedGroup? = nil
) -> QuantizedShardedToAllLinear
```

### callAsFunction(_:)

Forward pass:
1. Compute `quantizedMM(x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits, mode: mode)`.
2. Apply `MLXDistributed.allSum(x, group: group)`.
3. Add bias if present.

---

## sumGradients(group:)

Returns a closure that is the identity in the forward pass but performs `allSum` on the cotangents during the backward pass.

```swift
public func sumGradients(group: DistributedGroup) -> (MLXArray) -> MLXArray
```

**Parameters:**
- `group`: The distributed group to aggregate gradients over.

**Returns:** A closure `(MLXArray) -> MLXArray` that is identity forward, allSum backward.

The result is cached per group instance using `ObjectIdentifier`. On a size-1 group, returns a pure identity closure (optimization).

Internally uses `CustomFunction` with:
- `Forward { inputs in inputs }` — identity pass-through
- `VJP { _, cotangents in cotangents.map { MLXDistributed.allSum($0, group: group) } }` — sum cotangents across group

```swift
let fn = sumGradients(group: group)
let output = fn(input)  // Forward: output == input
// Backward: gradient of output is allSum'd across group
```

## Module Protocol Compliance

All four distributed layer types:
- Inherit from `Module`
- Conform to `UnaryLayer`
- Store `group` as a plain property (excluded from `parameters()` and `children()`)
- Return `weight` and optionally `bias` from `parameters()`
- Return empty `children()` (no sub-modules)
- Support `freeze()` / `unfreeze()` (quantized variants re-freeze after unfreeze)
- Support `update(parameters:)` for weight updates
