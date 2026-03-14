---
name: swift-nn-worker
description: Worker for MLXNN distributed layer features - distributed linear layers, sharding utilities, and tests
---

# Swift NN Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for features that involve:
- Distributed NN layer implementations (AllToShardedLinear, ShardedToAllLinear, etc.)
- Quantized distributed layer implementations
- Sharding utility functions (shardLinear, shardInPlace, averageGradients)
- CustomFunction/VJP-based helpers (sumGradients)
- NN layer tests

## Work Procedure

### 1. Read Context

- Read `skills/mlx-swift/SKILL.md` and references: `neural-networks.md`, `custom-layers.md`, `transforms.md`
- Read the feature description, preconditions, expectedBehavior, and verificationSteps carefully
- Read `.factory/library/architecture.md` for distributed layer design patterns
- Read existing implementations for patterns:
  - `Source/MLXNN/Linear.swift` -- base Linear layer
  - `Source/MLXNN/Quantized.swift` -- QuantizedLinear, Quantized protocol
  - `Source/MLXNN/Module.swift` -- Module base class, @ModuleInfo, @ParameterInfo
  - `Source/MLX/MLXCustomFunction.swift` -- CustomFunction with VJP support
  - `Source/MLX/Distributed.swift` -- MLXDistributed API (must exist from prior feature)

### 2. Write Tests First (TDD)

Before implementing:
- Create `Tests/MLXTests/DistributedNNTests.swift` (or add to existing)
- Write test cases matching expectedBehavior:
  - Init tests: check weight.shape, bias.shape, dtype, frozen state
  - Forward tests: check output shape for various batch sizes
  - Module protocol tests: parameters(), children(), freeze/unfreeze, update
  - Conversion tests: shardLinear return types and weight shapes
- Follow patterns from existing tests (e.g., `Tests/MLXTests/ModuleTests.swift`)
- Run tests to confirm they fail (red)

### 3. Implement

**For distributed linear layers:**
- Subclass `Module` directly (not `Linear`)
- Store `group` as a plain property (NOT `@ModuleInfo` or `@ParameterInfo`) -- it must NOT appear in parameters() or children()
- Use `@ParameterInfo` only for `weight` and optional `bias`
- Validate divisibility in init (output_dims % N == 0 for AllToSharded, input_dims % N == 0 for ShardedToAll)
- `callAsFunction(_: MLXArray) -> MLXArray` following Python logic exactly

**For quantized distributed layers:**
- Store `groupSize: Int`, `bits: Int`, `mode: QuantizationMode`
- Conform to `Quantized` protocol
- Call `self.freeze()` after init
- Override `unfreeze` to re-freeze own params: `super.unfreeze(); freeze(recurse: false)`
- Use `quantizedMatmul` (maps to Python's `mx.quantized_matmul`)

**For sumGradients helper:**
- Use `CustomFunction` with `Forward` (identity) and `VJP` (allSum on gradients)
- Cache per group (use dictionary keyed by group identity)

**For shardLinear/shardInPlace:**
- Accept sharding type as enum (`.allToSharded`, `.shardedToAll`)
- Use `split` and `concatenate` for weight sharding
- Support `segments` parameter (default 1) for fused QKV matrices
- Call `contiguous()` on sharded results

### 4. Verify

- Run `xcodebuild build -scheme mlx-swift-Package -destination 'platform=macOS'` (must succeed)
- Run `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS'` (all tests must pass)
- Verify NN layer tests specifically:
  - Shapes are correct for size-1 group
  - ShardedToAllLinear output matches standard Linear (within atol=1e-5)
  - Module protocol methods work correctly
  - Quantized layers are frozen after init

### 5. Manual Verification

- Compare each layer's `callAsFunction` against the Python implementation
- Verify weight initialization matches Python (scale = sqrt(1/inputDims), uniform distribution)
- Check that `group` does NOT appear in parameters() or children() output
- For quantized layers: verify trainableParameters() is empty after init

## Example Handoff

```json
{
  "salientSummary": "Implemented AllToShardedLinear and ShardedToAllLinear with sumGradients helper. Both use CustomFunction VJP for gradient aggregation. Wrote 18 test cases covering init shapes, forward pass, bias/no-bias, Module protocol compliance, and comparison with standard Linear. xcodebuild test: 540 tests, 0 failures.",
  "whatWasImplemented": "Source/MLXNN/Distributed.swift: AllToShardedLinear (weight [outDims/N, inDims], forward: sumGradients(x) then addMM), ShardedToAllLinear (weight [outDims, inDims/N], forward: matmul then allSum then add bias). sumGradients helper using CustomFunction with identity forward and allSum VJP, cached per group.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "xcodebuild build -scheme mlx-swift-Package -destination 'platform=macOS'", "exitCode": 0, "observation": "BUILD SUCCEEDED"},
      {"command": "xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests", "exitCode": 0, "observation": "540 tests, 0 failures (18 new)"}
    ],
    "interactiveChecks": [
      {"action": "Compared AllToShardedLinear.callAsFunction against Python distributed.py", "observed": "Logic matches: sum_gradients(x) -> addMM(bias, x, weight.T)"},
      {"action": "Verified group not in parameters()", "observed": "parameters() returns only weight and bias, no group"},
      {"action": "Tested ShardedToAllLinear output vs Linear with same weights", "observed": "allClose within atol=1e-5 on size-1 group"}
    ]
  },
  "tests": {
    "added": [
      {"file": "Tests/MLXTests/DistributedNNTests.swift", "cases": [
        {"name": "testAllToShardedLinearInit", "verifies": "Weight shape [outDims, inDims], bias shape [outDims] for size-1 group"},
        {"name": "testAllToShardedLinearForward", "verifies": "Output shape [batch, outDims] for various batch sizes"},
        {"name": "testShardedToAllVsLinear", "verifies": "Output matches standard Linear within tolerance"},
        {"name": "testModuleProtocolCompliance", "verifies": "parameters, children, freeze/unfreeze work correctly"},
        {"name": "testNoBias", "verifies": "Layers work with bias=false"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- `Source/MLX/Distributed.swift` doesn't exist yet (prerequisite feature not done)
- `CustomFunction` VJP doesn't work as expected
- Module reflection doesn't handle `group` property correctly (appears in parameters when it shouldn't)
- Quantized protocol conformance requires changes to existing Quantized.swift
- Weight sharding logic is unclear for edge cases
