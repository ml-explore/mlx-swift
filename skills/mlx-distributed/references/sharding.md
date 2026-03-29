# Sharding Utilities API Reference

Complete API reference for `shardLinear`, `shardInPlace`, and `ShardingType`.

## ShardingType

Describes the type of sharding for distributed linear layers.

```swift
public enum ShardingType {
    case allToSharded
    case shardedToAll
}
```

| Case | Description | Input | Output |
|------|-------------|-------|--------|
| `.allToSharded` | Column-parallel: replicated input → sharded output | Full `[batch, inDims]` | Sharded `[batch, outDims/N]` |
| `.shardedToAll` | Row-parallel: sharded input → replicated output | Sharded `[batch, inDims/N]` | Full `[batch, outDims]` |

---

## shardLinear(module:sharding:segments:group:)

Create a new distributed linear layer from an existing `Linear` or `QuantizedLinear`.

```swift
public func shardLinear(
    module: Module,
    sharding: ShardingType,
    segments: Int = 1,
    group: DistributedGroup? = nil
) -> Module
```

**Parameters:**
- `module`: The `Linear` or `QuantizedLinear` layer to shard.
- `sharding`: The type of sharding (`.allToSharded` or `.shardedToAll`).
- `segments`: Number of segments for fused weights (e.g., 3 for QKV). Default is `1`.
- `group`: The distributed group. If `nil`, uses `MLXDistributed.init()`.

**Returns:** A new distributed `Module` with sharded parameters.

**Precondition:** `module` must be a `Linear` or `QuantizedLinear`. Other module types cause a `preconditionFailure`.

### Type Dispatch

`QuantizedLinear` is checked before `Linear` because `QuantizedLinear` is a subclass of `Linear` and would otherwise match the `Linear` case.

| Sharding | Input Type | Output Type |
|----------|-----------|-------------|
| `.allToSharded` | `QuantizedLinear` | `QuantizedAllToShardedLinear` |
| `.allToSharded` | `Linear` | `AllToShardedLinear` |
| `.shardedToAll` | `QuantizedLinear` | `QuantizedShardedToAllLinear` |
| `.shardedToAll` | `Linear` | `ShardedToAllLinear` |

### Example

```swift
let group = MLXDistributed.`init`()!

// Standard Linear → AllToShardedLinear
let linear = Linear(1024, 1024, bias: true)
eval(linear)
let sharded = shardLinear(module: linear, sharding: .allToSharded, group: group)
// sharded is AllToShardedLinear

// QuantizedLinear → QuantizedShardedToAllLinear
let quantized = QuantizedLinear(linear, groupSize: 64, bits: 4)
eval(quantized)
let shardedQ = shardLinear(module: quantized, sharding: .shardedToAll, group: group)
// shardedQ is QuantizedShardedToAllLinear
```

---

## shardInPlace(module:sharding:segments:group:)

Shard a module's parameters in-place using `Module.update(parameters:)`.

```swift
public func shardInPlace(
    module: Module,
    sharding: ShardingType,
    segments: Int = 1,
    group: DistributedGroup? = nil
)
```

**Parameters:**
- `module`: The module whose parameters will be sharded in-place.
- `sharding`: The type of sharding (`.allToSharded` or `.shardedToAll`).
- `segments`: Number of segments for fused weights (e.g., 3 for QKV). Default is `1`.
- `group`: The distributed group. If `nil`, uses `MLXDistributed.init()`.

Unlike `shardLinear`, this function modifies the parameters of the existing module without changing its type. The module itself must natively support distributed communication for the collective ops to take effect.

### Example

```swift
let linear = Linear(64, 32, bias: true)
eval(linear)

// Parameters are sharded in-place; module type remains Linear
shardInPlace(module: linear, sharding: .allToSharded, group: group)
// linear.weight.shape is now [32/N, 64] for a group of size N
```

---

## Segments Parameter

The `segments` parameter allows sharding of fused weight matrices. This is critical for architectures that fuse multiple projections into a single weight (e.g., fused QKV in transformers).

### How It Works

1. The weight is split into `segments` equal parts along the sharding axis.
2. Each segment is independently split across the `N` processes in the group.
3. The rank-local parts from each segment are concatenated back together.

### Example: Fused QKV (segments=3)

```
Original weight: [3*hidden, hidden] = [3072, 1024]
                  ├── Q: [1024, 1024]
                  ├── K: [1024, 1024]
                  └── V: [1024, 1024]

With N=2, segments=3, allToSharded:
  1. Split into 3 segments: Q[1024, 1024], K[1024, 1024], V[1024, 1024]
  2. Each segment split by N=2: Q[512, 1024], K[512, 1024], V[512, 1024]
  3. Rank 0 gets first half of each, rank 1 gets second half
  4. Concatenated: rank 0 = [1536, 1024], rank 1 = [1536, 1024]
```

```swift
// Fused QKV linear: weight shape [3*1024, 1024]
let fusedQKV = Linear(1024, 3072, bias: true)
eval(fusedQKV)

let sharded = shardLinear(
    module: fusedQKV, sharding: .allToSharded, segments: 3, group: group)
```

---

## Internal Sharding Predicates

The sharding logic uses internal predicate functions to determine how each parameter should be sharded.

### allToShardedPredicate

- **Bias:** Shard along last axis (`axis: -1`).
- **Weight:** Shard along axis 0 (`max(ndim - 2, 0)` for 2D weights).

### shardedToAllPredicate

- **Bias:** Don't shard (return `nil`). Bias is replicated across all ranks.
- **Weight:** Shard along last axis (`axis: -1`).

### shardParameterTree (internal)

Applies the predicate to each parameter in a flattened parameter tree:

1. Flatten the `ModuleParameters` to `[(path, MLXArray)]` pairs.
2. For each parameter, check the predicate to get the sharding axis and segments.
3. Split into segments along the axis, then split each segment across the group.
4. Take the rank-local part and concatenate back.
5. Unflatten back to `ModuleParameters`.

---

## Sharding a Full Model

```swift
import MLX
import MLXNN

let group = MLXDistributed.`init`()!

// Example: Shard a 4-layer model for tensor parallelism
// Alternating allToSharded / shardedToAll for proper data flow
let model = Sequential(
    layers:
        Linear(1024, 1024, bias: true),
        Linear(1024, 1024, bias: true),
        Linear(1024, 1024, bias: true),
        Linear(1024, 1024, bias: true)
)
eval(model)

let shardedModel = Sequential(
    layers:
        shardLinear(module: model.layers[0], sharding: .allToSharded, group: group) as! UnaryLayer,
        shardLinear(module: model.layers[1], sharding: .shardedToAll, group: group) as! UnaryLayer,
        shardLinear(module: model.layers[2], sharding: .allToSharded, group: group) as! UnaryLayer,
        shardLinear(module: model.layers[3], sharding: .shardedToAll, group: group) as! UnaryLayer
)
eval(shardedModel)

// Forward pass
let input = MLXRandom.uniform(0 ..< 1, [4, 1024])
let output = shardedModel(input)
// output shape: [4, 1024] (ShardedToAll aggregates back to full)
```
