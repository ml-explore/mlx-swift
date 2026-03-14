# Architecture

Architectural decisions, patterns discovered, and design notes.

---

## MLX-Swift Module Architecture

```
MLXOptimizers (Adam, AdamW, SGD)
       |
MLXNN (Layers, Modules, Losses)
       |
MLX (Arrays, Ops, Transforms, FFT, Linalg, Random, Distributed)
       |
Cmlx (C/C++ vendored MLX + MLX-C)
```

## Distributed Architecture

### Layer Structure
- `Cmlx` target compiles: MLX C++ distributed core + ring backend + JACCL backend + MLX-C wrappers
- `MLX` target: `Distributed.swift` with `DistributedGroup` class + `MLXDistributed` enum
- `MLXNN` target: `Distributed.swift` with distributed NN layers

### C Interop Pattern
```
Swift (MLXDistributed.allSum) -> C (mlx_distributed_all_sum) -> C++ (mlx::core::distributed::all_sum)
```

### Handle Lifecycle
`DistributedGroup` wraps `mlx_distributed_group` (opaque `void* ctx`).
- Created by `mlx_distributed_init(strict)` or `mlx_distributed_group_split(group, color, key)`
- `deinit` must call appropriate free function
- Split children are independent of parent (own reference-counted C++ object)

### Backend Selection
MLX-C `init(strict)` uses implicit `bk="any"` which tries backends in order.
When both ring and JACCL are compiled:
- JACCL is tried first (but only available on macOS 26.2+ with TB5 + RDMA)
- Ring is fallback (available unconditionally with TCP sockets)

### Distributed NN Layer Design
- `AllToShardedLinear`: identity forward for input, all_sum backward for gradients (via CustomFunction VJP)
- `ShardedToAllLinear`: all_sum in forward pass after matmul
- Quantized variants use `quantizedMatmul` instead of standard matmul
- `group` stored as plain property (NOT `@ModuleInfo` / `@ParameterInfo`) to exclude from parameter tree

### GPU Limitation
Distributed operations (AllReduce, AllGather, Send, Recv) have **no GPU implementation** -- they must run on CPU. For multi-process distributed code, set `MLX.Device.setDefault(.cpu)`. Single-process tests on size-1 groups work on GPU because identity operations don't actually invoke the distributed primitives. The NN layers must handle this: data may need CPU transfer for collective ops then back to GPU.

### MLX-C Gaps
1. `mlx_distributed_init()` has no backend parameter (C++ has `bk` string). Filed as issue on ml-explore/mlx-c. Workaround: compile desired backends; `"any"` picks first available.
2. `mlx_distributed_group_free()` is not publicly exposed in MLX-C v0.5.0. The private inline helper exists in `mlx/c/private/distributed_group.h` but is C++-only. Groups are singleton-like and long-lived, so practical impact is minimal. Should file upstream issue.
