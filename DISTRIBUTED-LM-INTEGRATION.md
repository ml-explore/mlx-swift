# Distributed Inference Integration Guide for mlx-swift-lm

This document specifies the changes needed in [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) to support distributed inference across multiple Apple Silicon nodes. The distributed primitives in [mlx-swift](https://github.com/ml-explore/mlx-swift) are complete — this guide covers the integration layer.

Reference implementation: [Python mlx-lm distributed](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py) (`sharded_load()`, per-model `shard()` methods).

## 1. Architecture

### Tensor Parallelism (implement first)

Tensor parallelism splits individual weight matrices across devices. Each device holds a slice of every layer and processes the full sequence, communicating intermediate results via collective operations (allSum, allGather).

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                        App Layer                            │
                    │   ModelContainer.loadDistributed() → generate()             │
                    └───────────────────────────┬─────────────────────────────────┘
                                                │
                    ┌───────────────────────────▼─────────────────────────────────┐
                    │                    mlx-swift-lm                              │
                    │   ShardableModel protocol                                   │
                    │   shardedLoad() — lazy load + shard + materialize            │
                    │   Per-model shard() — replaces Linear with sharded variants  │
                    └───────────────────────────┬─────────────────────────────────┘
                                                │  calls shardLinear()
                    ┌───────────────────────────▼─────────────────────────────────┐
                    │                      mlx-swift                               │
                    │   MLXNN: AllToShardedLinear, ShardedToAllLinear              │
                    │   MLXNN: shardLinear(), shardInPlace(), averageGradients()   │
                    │   MLX:  MLXDistributed (allSum, allGather, send, recv, ...)  │
                    │   MLX:  DistributedGroup (rank, size)                        │
                    └───────────────────────────┬─────────────────────────────────┘
                                                │
                    ┌───────────────────────────▼─────────────────────────────────┐
                    │               MLX-C / C++ Backends                           │
                    │   Ring (TCP/IP) — always available                           │
                    │   JACCL (RDMA/Thunderbolt 5) — macOS 26.2+                  │
                    └─────────────────────────────────────────────────────────────┘
```

**Why tensor parallelism first:** It is simpler, requires no changes to the generation pipeline or KV cache, and covers the primary use case (running models too large for a single device). Pipeline parallelism can be added later for very large models.

**Key insight:** The generation pipeline (`TokenIterator`, `generate()`) and KV cache need **no fundamental changes**. Sharded linear layers handle all inter-node communication internally during the forward pass. After sharding, `n_heads` is divided by the group size, so KV cache dimensions are automatically correct.

### Pipeline Parallelism (future)

Pipeline parallelism assigns different layers to different devices. Device 0 runs layers 0-15, device 1 runs layers 16-31, etc. This requires:
- Layer assignment logic (`pipeline()` method on models)
- Selective weight file downloading (only download files for local layers)
- Inter-device activation passing via `send`/`recv`

This is out of scope for the initial implementation.

## 2. mlx-swift Distributed API Quick Reference

All APIs below are already implemented in mlx-swift. This is what you will call from mlx-swift-lm.

> **Critical:** All distributed operations are CPU-only. Wrap distributed code in `Device.withDefaultDevice(.cpu) { ... }` or pass `stream: .cpu`.

### Group Management

```swift
// Check if any distributed backend is available
MLXDistributed.isAvailable() -> Bool

// Initialize a distributed group (returns nil if no backend, or if strict and init fails)
MLXDistributed.`init`(strict: Bool = false) -> DistributedGroup?

// Group properties
group.rank  -> Int   // This process's rank (0-indexed)
group.size  -> Int   // Total number of processes in the group
```

### Sharding Utilities (the main API you will use)

```swift
// Replace a Linear or QuantizedLinear with its distributed variant.
// Automatically detects the module type and returns the appropriate sharded layer.
// segments: for fused QKV weights (e.g., 3). Default is 1.
public func shardLinear(
    module: Module, sharding: ShardingType, segments: Int = 1,
    group: DistributedGroup? = nil
) -> Module

// Shard a module's parameters in-place (modifies the module's weight arrays directly).
public func shardInPlace(
    module: Module, sharding: ShardingType, segments: Int = 1,
    group: DistributedGroup? = nil
)

public enum ShardingType {
    case allToSharded   // Column-parallel: full input → sharded output (for Q, K, V, gate, up)
    case shardedToAll   // Row-parallel: sharded input → full output (for O, down)
}
```

### Gradient Averaging (for distributed training)

```swift
public func averageGradients(
    gradients: ModuleParameters,
    group: DistributedGroup? = nil,
    allReduceSize: Int = 32 * 1024 * 1024,
    communicationType: DType? = nil,
    communicationStream: StreamOrDevice? = nil
) -> ModuleParameters
```

### Collective Operations (lower level, rarely needed directly)

```swift
MLXDistributed.allSum(_ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray
MLXDistributed.allGather(_ array: MLXArray, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray
MLXDistributed.send(_ array: MLXArray, to dst: Int, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray
MLXDistributed.recv(shape: [Int], dtype: DType, from src: Int, group: DistributedGroup, stream: StreamOrDevice = .default) -> MLXArray
```

## 3. Changes to MLXLMCommon

### 3.1 ShardableModel Protocol

**File:** `Sources/MLXLMCommon/LanguageModel.swift` (or a new file `Sources/MLXLMCommon/ShardableModel.swift`)

```swift
import MLX
import MLXNN

/// A language model that supports tensor-parallel sharding across a distributed group.
///
/// Models conforming to this protocol can replace their linear layers with distributed
/// variants, enabling inference across multiple devices. After calling `shard()`, the
/// model's forward pass automatically communicates across the group — no changes to
/// the generation pipeline are needed.
public protocol ShardableModel: LanguageModel {
    /// Replace linear layers with distributed sharded variants.
    ///
    /// This method walks the model's transformer layers and replaces:
    /// - Attention Q/K/V projections with `AllToShardedLinear` (column-parallel)
    /// - Attention O projection with `ShardedToAllLinear` (row-parallel)
    /// - MLP gate/up projections with `AllToShardedLinear` (column-parallel)
    /// - MLP down projection with `ShardedToAllLinear` (row-parallel)
    ///
    /// It also divides `n_heads` and `n_kv_heads` by `group.size`.
    ///
    /// - Parameter group: The distributed group. Defaults to the global group.
    mutating func shard(group: DistributedGroup?)
}

extension ShardableModel {
    public mutating func shard() {
        shard(group: nil)
    }
}
```

### 3.2 Distributed Model Loading

**File:** `Sources/MLXLMCommon/Load.swift` (extend existing file)

Add a new function alongside the existing `loadWeights()`:

```swift
/// Load a model with distributed tensor-parallel sharding.
///
/// This function:
/// 1. Creates the model from configuration (weights are lazy/uninitialized)
/// 2. Loads weights from safetensors files
/// 3. Calls `model.shard(group:)` to replace linear layers with distributed variants
/// 4. Materializes all parameters with `eval(model)`
/// 5. Performs a barrier sync to ensure all ranks are ready
///
/// - Parameters:
///   - hub: The HuggingFace Hub API instance
///   - configuration: Model configuration (repo ID, quantization, etc.)
///   - group: Distributed group for tensor parallelism
///   - progressHandler: Progress callback for download/loading
/// - Returns: A fully loaded and sharded ModelContext
public func shardedLoad(
    hub: HubApi = HubApi(),
    configuration: ModelConfiguration,
    group: DistributedGroup,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> ModelContext {
    // Step 1: Download model files (all ranks download — or use shared filesystem)
    let modelDirectory = try await downloadModel(
        hub: hub, configuration: configuration,
        progressHandler: progressHandler
    )

    // Step 2: Create model from config
    let config = try loadConfiguration(url: modelDirectory)
    var model = try createModel(configuration: configuration, rawConfig: config)

    // Step 3: Load weights (standard loading — all weights on each rank)
    try loadWeights(
        modelDirectory: modelDirectory, model: model,
        quantization: configuration.quantization
    )

    // Step 4: Shard the model (replace Linear layers with distributed variants)
    guard var shardableModel = model as? (any ShardableModel) else {
        throw DistributedError.modelNotShardable(
            "\(type(of: model)) does not conform to ShardableModel"
        )
    }
    shardableModel.shard(group: group)
    model = shardableModel as! any LanguageModel

    // Step 5: Materialize sharded weights
    eval(model)

    // Step 6: Barrier sync — ensures all ranks have finished loading
    let barrier = MLXDistributed.allSum(
        MLXArray(Float(1.0)), group: group, stream: .cpu
    )
    eval(barrier)

    // Step 7: Load tokenizer (same on all ranks)
    let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)

    return ModelContext(
        configuration: configuration,
        model: model,
        tokenizer: tokenizer
    )
}

public enum DistributedError: Error, LocalizedError {
    case modelNotShardable(String)
    case distributedNotAvailable

    public var errorDescription: String? {
        switch self {
        case .modelNotShardable(let msg): return "Model is not shardable: \(msg)"
        case .distributedNotAvailable: return "No distributed backend available"
        }
    }
}
```

**Important implementation notes:**
- The exact function names `downloadModel()`, `loadConfiguration()`, `createModel()`, `loadWeights()`, and `loadTokenizer()` should match whatever the current mlx-swift-lm codebase uses. Check `Load.swift` for the actual names.
- The `ModelContext` struct may have a different initializer — adapt accordingly.
- If mlx-swift-lm uses a `ModelFactory` pattern, add a convenience method there too (see 3.3).

### 3.3 ModelFactory Extension

**File:** `Sources/MLXLMCommon/ModelFactory.swift` or `Sources/MLXLLM/LLMModelFactory.swift`

```swift
extension LLMModelFactory {
    /// Load a distributed model into a thread-safe container.
    ///
    /// This is the primary entry point for distributed inference.
    ///
    /// - Parameters:
    ///   - configuration: Model configuration
    ///   - group: Distributed group for tensor parallelism
    ///   - progressHandler: Progress callback
    /// - Returns: A ModelContainer ready for generation
    public func loadDistributedContainer(
        configuration: ModelConfiguration,
        group: DistributedGroup,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let context = try await shardedLoad(
            configuration: configuration,
            group: group,
            progressHandler: progressHandler
        )
        return ModelContainer(context: context)
    }
}
```

### 3.4 Generation Pipeline — Rank-Aware Output

**File:** `Sources/MLXLMCommon/Evaluate.swift`

The generate functions need only one change: only rank 0 should emit tokens to the caller. All ranks must still run the full generation loop (because forward passes require collective communication), but non-zero ranks discard the output.

Option A — Add rank parameter to generate:

```swift
/// Generate text from a prompt with distributed support.
///
/// All ranks execute the generation loop (required for collective ops in forward pass),
/// but only rank 0 yields tokens through the stream.
///
/// - Parameter rank: This process's rank. Pass `group.rank`. If nil, all output is emitted.
public func generate(
    input: LMInput,
    parameters: GenerateParameters,
    context: ModelContext,
    rank: Int? = nil,
    // ... other existing parameters
) -> AsyncStream<Generation> {
    AsyncStream { continuation in
        Task {
            // ... existing generation logic ...

            for try await token in tokenIterator {
                // Only rank 0 emits output
                if let rank, rank != 0 { continue }

                continuation.yield(.token(token))
            }

            continuation.finish()
        }
    }
}
```

Option B — Let the caller handle rank filtering (simpler, less invasive):

```swift
// In the app layer:
let group = MLXDistributed.`init`()!

for await generation in generate(input: input, parameters: params, context: context) {
    if group.rank == 0 {
        // Process output
        print(generation.text, terminator: "")
    }
}
```

**Recommendation:** Option B is simpler and avoids changing the generate() signature. The app layer already knows the rank.

### 3.5 KV Cache — No Changes Needed

After `shard()` divides `n_heads` and `n_kv_heads` by `group.size`, each rank's attention layer operates on fewer heads. The KV cache is created based on the model's head count, so dimensions are automatically correct:

- Rank 0 with 8 heads (of original 32) → KV cache stores 8 heads
- Rank 1 with 8 heads (of original 32) → KV cache stores 8 heads

No code changes to KV cache classes.

## 4. Changes to MLXLLM — Per-Model Sharding

### 4.1 General Pattern

Every transformer model follows the same sharding pattern:

```
Attention Layer:
  Q projection:  allToSharded (column-parallel — output is sharded)
  K projection:  allToSharded
  V projection:  allToSharded
  O projection:  shardedToAll (row-parallel — gathers results back)
  n_heads:       ÷= group.size
  n_kv_heads:    ÷= group.size

MLP Layer:
  gate projection: allToSharded (column-parallel)
  up projection:   allToSharded (column-parallel)
  down projection: shardedToAll (row-parallel — gathers results back)
```

The rule is:
- **First linear in a pair:** `allToSharded` — splits the computation across devices
- **Last linear in a pair:** `shardedToAll` — gathers results back to full size

### 4.2 Llama Model (Reference Implementation)

**File:** `Sources/MLXLLM/Models/Llama.swift` (or wherever Llama is defined)

First, examine the existing Llama model structure to find the exact property names. The model will have something like:

```swift
class LlamaModel {
    var layers: [TransformerBlock]
    // ...
}

class TransformerBlock {
    var selfAttn: Attention    // or attention
    var mlp: MLP
    // ...
}

class Attention {
    var qProj: Linear    // or q_proj — check actual naming
    var kProj: Linear
    var vProj: Linear
    var oProj: Linear
    var nHeads: Int
    var nKVHeads: Int
    // ...
}

class MLP {
    var gateProj: Linear
    var upProj: Linear
    var downProj: Linear
    // ...
}
```

> **Important:** Check the exact property names in the Swift source. They may use camelCase (`qProj`) or snake_case (`q_proj`) depending on the model. Some models use `@ModuleInfo` wrappers. Adapt the code below accordingly.

```swift
extension LlamaModel: ShardableModel {
    mutating func shard(group: DistributedGroup? = nil) {
        let group = group ?? MLXDistributed.`init`()!
        let N = group.size

        for i in model.layers.indices {
            // Attention projections
            model.layers[i].selfAttn.qProj = shardLinear(
                module: model.layers[i].selfAttn.qProj,
                sharding: .allToSharded, group: group
            )
            model.layers[i].selfAttn.kProj = shardLinear(
                module: model.layers[i].selfAttn.kProj,
                sharding: .allToSharded, group: group
            )
            model.layers[i].selfAttn.vProj = shardLinear(
                module: model.layers[i].selfAttn.vProj,
                sharding: .allToSharded, group: group
            )
            model.layers[i].selfAttn.oProj = shardLinear(
                module: model.layers[i].selfAttn.oProj,
                sharding: .shardedToAll, group: group
            )

            // Divide head counts
            model.layers[i].selfAttn.nHeads /= N
            model.layers[i].selfAttn.nKVHeads /= N

            // MLP projections
            model.layers[i].mlp.gateProj = shardLinear(
                module: model.layers[i].mlp.gateProj,
                sharding: .allToSharded, group: group
            )
            model.layers[i].mlp.upProj = shardLinear(
                module: model.layers[i].mlp.upProj,
                sharding: .allToSharded, group: group
            )
            model.layers[i].mlp.downProj = shardLinear(
                module: model.layers[i].mlp.downProj,
                sharding: .shardedToAll, group: group
            )
        }
    }
}
```

**Why this works:**
- `shardLinear()` automatically detects whether the input is `Linear` or `QuantizedLinear` and returns the appropriate distributed variant (`AllToShardedLinear`, `QuantizedAllToShardedLinear`, etc.)
- The returned module conforms to `UnaryLayer`, so the rest of the model's forward pass works unchanged
- The sharded layers' `callAsFunction(_:)` handles communication (allSum/allGather) internally

### 4.3 Fused QKV Models

Some models fuse Q, K, V into a single linear layer. Use the `segments` parameter:

```swift
// If the model has a fused qkv_proj instead of separate q/k/v:
model.layers[i].selfAttn.qkvProj = shardLinear(
    module: model.layers[i].selfAttn.qkvProj,
    sharding: .allToSharded,
    segments: 3,    // Q, K, V are 3 segments in the fused weight
    group: group
)
```

### 4.4 Model Catalog

Each model in `Sources/MLXLLM/Models/` needs a `shard()` implementation. The pattern is identical for standard transformer architectures — only property names differ.

| Model | Attention projections | MLP projections | Notes |
|-------|----------------------|-----------------|-------|
| **Llama** | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj | Reference implementation |
| **Qwen2** | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj | Same as Llama |
| **Gemma** | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj | Same as Llama |
| **Phi** | q_proj, k_proj, v_proj, dense | fc1, fc2 | Different naming; fc1→allToSharded, fc2→shardedToAll |
| **Mistral** | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj | Same as Llama |
| **Starcoder2** | q_proj, k_proj, v_proj, o_proj | c_fc, c_proj | c_fc→allToSharded, c_proj→shardedToAll |
| **Cohere** | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj | Same as Llama |

> **For each model:** Read the actual Swift source file in mlx-swift-lm to find the exact property names and types. The table above is based on Python mlx-lm and common naming; Swift names may differ.

### 4.5 MoE (Mixture of Experts) Models

MoE models (DeepSeek-V3, Qwen3.5-MoE) need special handling:
- The router/gate layer should NOT be sharded (it's shared across all devices)
- Individual expert MLP layers follow the standard gate/up/down pattern
- The expert dispatch logic may need coordination across ranks

**Recommendation:** Defer MoE support to a follow-up. Standard dense models cover the majority of use cases.

## 5. Multi-Process Setup

### 5.1 Environment Variables

The MLX-C ring backend reads these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `MLX_RANK` | This process's rank (0-indexed) | `0` |
| `MLX_HOSTFILE` | Path to JSON hostfile | `/tmp/hostfile.json` |

### 5.2 Hostfile Format

A JSON array of arrays. Each inner array contains one `"ip:port"` string per rank:

**2-node cluster (e.g., two Mac Studios on Ethernet):**
```json
[
    ["192.168.1.10:12345"],
    ["192.168.1.11:12345"]
]
```

**4-node cluster:**
```json
[
    ["192.168.1.10:12345"],
    ["192.168.1.11:12345"],
    ["192.168.1.12:12345"],
    ["192.168.1.13:12345"]
]
```

**Local testing (2 processes on same machine):**
```json
[
    ["127.0.0.1:12345"],
    ["127.0.0.1:12346"]
]
```

### 5.3 Shell Script Launcher

Unlike Python's `mlx.launch`, Swift has no built-in launcher. Use a shell script:

```bash
#!/bin/bash
# launch_distributed.sh — Launch N workers for distributed inference
# Usage: ./launch_distributed.sh <hostfile> <executable> [args...]

HOSTFILE=$1
EXECUTABLE=$2
shift 2

# Count ranks from hostfile
NUM_RANKS=$(python3 -c "import json; print(len(json.load(open('$HOSTFILE'))))")

echo "Launching $NUM_RANKS ranks..."

PIDS=()
for ((rank=0; rank<NUM_RANKS; rank++)); do
    MLX_RANK=$rank MLX_HOSTFILE=$HOSTFILE $EXECUTABLE "$@" &
    PIDS+=($!)
    sleep 0.5  # Stagger launches to avoid port contention
done

echo "Waiting for all ranks to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done
```

### 5.4 Swift-Based Launcher

For a more integrated solution, use Foundation.Process:

```swift
import Foundation

func launchDistributed(
    executable: URL,
    hostfile: URL,
    ranks: Int,
    arguments: [String] = []
) throws {
    var processes: [Process] = []

    for rank in 0..<ranks {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = ProcessInfo.processInfo.environment.merging([
            "MLX_RANK": "\(rank)",
            "MLX_HOSTFILE": hostfile.path,
        ]) { _, new in new }

        try process.run()
        processes.append(process)

        // Stagger launches
        if rank < ranks - 1 {
            Thread.sleep(forTimeInterval: 0.5)
        }
    }

    // Wait for all processes
    for process in processes {
        process.waitUntilExit()
    }
}
```

### 5.5 JACCL Backend (Thunderbolt 5)

JACCL is tried automatically before ring if the hardware is available:
- Requires macOS 26.2 or later
- Requires Thunderbolt 5 connection between nodes
- Provides RDMA (Remote Direct Memory Access) — significantly lower latency than TCP
- No configuration needed — MLX-C detects and uses it automatically

If JACCL is not available, MLX-C falls back to the ring (TCP) backend transparently.

## 6. Complete Example: Distributed Inference App

```swift
// main.swift — Minimal distributed inference app
import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM

@main
struct DistributedInferenceApp {
    static func main() async throws {
        // All distributed ops are CPU-only
        Device.withDefaultDevice(.cpu) {
            try await run()
        }
    }

    static func run() async throws {
        // Step 1: Initialize distributed group
        guard MLXDistributed.isAvailable() else {
            fatalError("No distributed backend available. Set MLX_RANK and MLX_HOSTFILE.")
        }

        guard let group = MLXDistributed.`init`(strict: true) else {
            fatalError("Failed to initialize distributed group")
        }

        let isRankZero = group.rank == 0

        if isRankZero {
            print("Distributed group initialized: rank \(group.rank)/\(group.size)")
        }

        // Step 2: Load model with sharding
        let configuration = ModelConfiguration(id: "mlx-community/Llama-3.2-3B-Instruct-4bit")

        let factory = LLMModelFactory.shared
        let container = try await factory.loadDistributedContainer(
            configuration: configuration,
            group: group
        ) { progress in
            if isRankZero {
                print("Loading: \(Int(progress.fractionCompleted * 100))%")
            }
        }

        if isRankZero {
            print("Model loaded and sharded across \(group.size) devices")
        }

        // Step 3: Generate text
        let prompt = "Explain tensor parallelism in one paragraph."

        try await container.perform { context in
            let input = try await context.processor.prepare(prompt: prompt)
            let parameters = GenerateParameters(temperature: 0.7)

            for await generation in generate(
                input: input, parameters: parameters, context: context
            ) {
                // Only rank 0 prints output
                if isRankZero {
                    switch generation {
                    case .token(let token):
                        print(token.text, terminator: "")
                    case .done:
                        print()  // Final newline
                    }
                }
            }
        }

        // Step 4: Clean up
        // All ranks must reach this point before exiting.
        // Use _exit(0) to avoid ring backend destructor hang.
        if isRankZero {
            print("Done.")
        }
        _exit(0)
    }
}
```

**Launch with:**
```bash
# Create hostfile for 2 local ranks
echo '[["127.0.0.1:12345"],["127.0.0.1:12346"]]' > /tmp/hostfile.json

# Launch both ranks
MLX_RANK=0 MLX_HOSTFILE=/tmp/hostfile.json .build/release/DistributedInferenceApp &
sleep 0.5
MLX_RANK=1 MLX_HOSTFILE=/tmp/hostfile.json .build/release/DistributedInferenceApp &
wait
```

## 7. Testing Strategy

### 7.1 Unit Tests — Single-Process (No Distributed Backend Needed)

These tests verify sharding logic without requiring multi-process setup:

```swift
import XCTest
import MLX
import MLXNN

class ShardingTests: XCTestCase {

    /// Verify that shard() replaces Linear layers with distributed variants.
    func testShardReplacesLinearLayers() {
        let model = createTestLlamaModel()  // Small test model

        // Before sharding: all projections are Linear
        XCTAssertTrue(model.layers[0].selfAttn.qProj is Linear)
        XCTAssertTrue(model.layers[0].mlp.gateProj is Linear)

        // Create a singleton group (size 1)
        let group = MLXDistributed.`init`()!
        model.shard(group: group)

        // After sharding: projections are distributed variants
        // On a size-1 group, shardLinear still returns distributed types
        // (they just behave as identity in the communication path)
        XCTAssertTrue(
            model.layers[0].selfAttn.qProj is AllToShardedLinear
            || model.layers[0].selfAttn.qProj is QuantizedAllToShardedLinear
        )
        XCTAssertTrue(
            model.layers[0].selfAttn.oProj is ShardedToAllLinear
            || model.layers[0].selfAttn.oProj is QuantizedShardedToAllLinear
        )
    }

    /// Verify head counts are divided by group size.
    func testShardDividesHeadCounts() {
        let model = createTestLlamaModel(nHeads: 32, nKVHeads: 8)
        let group = MLXDistributed.`init`()!  // size 1

        let originalHeads = model.layers[0].selfAttn.nHeads
        let originalKVHeads = model.layers[0].selfAttn.nKVHeads

        model.shard(group: group)

        // With size-1 group, counts stay the same (÷1)
        XCTAssertEqual(model.layers[0].selfAttn.nHeads, originalHeads / group.size)
        XCTAssertEqual(model.layers[0].selfAttn.nKVHeads, originalKVHeads / group.size)
    }

    /// Verify forward pass produces same output on size-1 group.
    func testShardedForwardMatchesOriginal() {
        let model = createTestLlamaModel()
        eval(model)

        let input = MLXArray.ones([1, 10], dtype: .int32)  // batch=1, seq=10
        let originalOutput = model(input)
        eval(originalOutput)

        let group = MLXDistributed.`init`()!
        model.shard(group: group)
        eval(model)

        let shardedOutput = model(input)
        eval(shardedOutput)

        // On size-1 group, sharded output should match original
        XCTAssertTrue(allClose(originalOutput, shardedOutput, atol: 1e-5).item())
    }

    /// Verify weight dimensions are divisible by group size.
    func testWeightDivisibility() {
        // Models require dimensions divisible by group.size
        // Test with known dimensions
        let linear = Linear(512, 256)
        eval(linear)

        let sharded = shardLinear(
            module: linear, sharding: .allToSharded
        )
        // Output dim should be 256 / group.size
        // Input dim should remain 512
    }
}
```

### 7.2 Multi-Process Tests

For testing actual distributed communication, follow the pattern established in `mlx-swift/Tests/MLXTests/DistributedTests.swift`:

1. Build a test helper executable that loads a small model, shards it, runs a forward pass, and outputs JSON results
2. Spawn 2 processes with different ranks using `Foundation.Process`
3. Verify both ranks produce consistent results

```swift
func testDistributedForwardPass() {
    // Spawn 2 worker processes that each:
    // 1. Init distributed group
    // 2. Load a small test model
    // 3. Shard it
    // 4. Run forward pass on same input
    // 5. Output logits shape and values as JSON

    guard let results = runMultiProcessTest(operation: "shardedForward") else {
        XCTFail("Multi-process test failed to launch")
        return
    }

    // Both ranks should produce identical output (sharded layers gather results)
    let rank0Logits = results[0]["logitsShape"] as! [Int]
    let rank1Logits = results[1]["logitsShape"] as! [Int]
    XCTAssertEqual(rank0Logits, rank1Logits)
}
```

### 7.3 Integration Tests

Test the full pipeline: load → shard → generate → verify output:

```swift
func testDistributedGeneration() async throws {
    // This test requires 2 processes — run as multi-process test
    // Each rank:
    // 1. Loads a small model (e.g., a tiny Llama with 2 layers)
    // 2. Shards across 2 ranks
    // 3. Generates 10 tokens from a fixed prompt with temperature=0 (deterministic)
    // 4. Rank 0 outputs the generated text

    // Verify: output is coherent text (not garbage)
    // Verify: both ranks completed without error
    // Verify: generation took less time than single-device (for large enough model)
}
```

### 7.4 What to Verify

| Test | What it checks |
|------|---------------|
| Layer type replacement | `shard()` converts Linear → AllToShardedLinear / ShardedToAllLinear |
| Head count division | `n_heads` and `n_kv_heads` ÷= `group.size` |
| Weight dimensions | Sharded weight shapes = original shapes ÷ group.size on the split axis |
| Forward pass consistency | Same input → same output across all ranks (after gathering) |
| KV cache dimensions | Cache created after sharding has correct (reduced) head dimensions |
| Quantized model support | `shard()` works on quantized models (QuantizedLinear → QuantizedAllToShardedLinear) |
| Generation determinism | Same prompt + seed → same output in distributed and single-device modes |
| Barrier sync | All ranks reach completion (no hangs or deadlocks) |

## 8. Implementation Priority

Implement in this order. Each step produces a testable, shippable increment:

### Phase 1: Core Infrastructure (MVP)
1. **ShardableModel protocol** — Define the protocol in MLXLMCommon
2. **Llama shard()** — Implement for the most common architecture
3. **shardedLoad()** — Distributed model loading function
4. **Rank-aware output** — Document the pattern (app-layer responsibility)
5. **Test with Llama-3.2-3B-Instruct-4bit** across 2 devices

### Phase 2: Model Coverage
6. **Qwen2 shard()** — Same pattern as Llama
7. **Gemma shard()** — Same pattern as Llama
8. **Mistral shard()** — Same pattern as Llama
9. **Phi shard()** — Different MLP naming (fc1/fc2)
10. **Starcoder2 shard()** — Different MLP naming (c_fc/c_proj)

### Phase 3: Polish
11. **ModelFactory convenience** — `loadDistributedContainer()` method
12. **Error handling** — Graceful failure when dimensions aren't divisible by group size
13. **Documentation** — Usage guide and examples
14. **Launcher utility** — Swift-based multi-process launcher

### Future
- Pipeline parallelism for very large models
- MoE model support
- Distributed KV cache sharing for prompt caching across restarts

## 9. Known Limitations and Upstream Gaps

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| All distributed ops are CPU-only | Must use `Device.withDefaultDevice(.cpu)` | Wrap model loading and generation in CPU scope |
| MLX-C has no backend selection parameter | Cannot programmatically choose ring vs JACCL | MLX-C tries JACCL first, then ring — usually correct |
| `mlx_distributed_group_free()` not in public C API | Group deallocation relies on C++ shared_ptr | No action needed — works via ref counting |
| `group.split()` unsupported by ring/JACCL | Cannot create subgroups | Not needed for tensor parallelism |
| `sumScatter` not implemented in ring backend | Cannot use reduce-scatter collective | Use allSum instead (slightly more bandwidth) |
| No Swift equivalent of Python's `mlx.launch` | Must use shell scripts or Foundation.Process for multi-process | See section 5.3 and 5.4 |
| Ring backend destructor can hang on exit | Process may not exit cleanly | Use `_exit(0)` instead of normal return |
| Head counts must be divisible by group size | Not all models work with all group sizes | Validate divisibility in `shard()` and fail with clear error |

## 10. Dependency Requirements

### mlx-swift-lm Package.swift

Update the mlx-swift dependency to the version containing distributed support:

```swift
dependencies: [
    .package(
        url: "https://github.com/ml-explore/mlx-swift",
        from: "X.Y.Z"  // Version with distributed support
    ),
    // ... other dependencies
]
```

Ensure all targets that need distributed have both `MLX` and `MLXNN` as dependencies:

```swift
.target(
    name: "MLXLMCommon",
    dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        // ... other dependencies
    ]
),
.target(
    name: "MLXLLM",
    dependencies: [
        "MLXLMCommon",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        // ... other dependencies
    ]
),
```

### Minimum Platform Requirements

- macOS 14.0+ (for MLX framework)
- macOS 26.2+ (for JACCL/Thunderbolt 5 backend — optional, ring works on any macOS)
- Swift 5.9+
- Xcode 15.0+
