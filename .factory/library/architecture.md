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
- Public MLX-C v0.5.0 does not expose `mlx_distributed_group_free()`, so Swift wrappers cannot currently release group handles through the public C API
- Split children are independent of parent (own reference-counted C++ object)

### Backend Selection
MLX-C `init(strict)` uses implicit `bk="any"` which tries backends in order.
When both ring and JACCL are compiled:
- JACCL is tried first (but only available on macOS 26.2+ with TB5 + RDMA)
- Ring is fallback (available unconditionally with TCP sockets)

### Distributed NN Layer Design
- `AllToShardedLinear`: identity forward for input, all_sum backward for gradients (via CustomFunction VJP)
- `ShardedToAllLinear`: all_sum in forward pass after matmul
- Quantized variants use `quantizedMM` instead of standard matmul (`quantizedMatmul` is the deprecated alias in this repo)
- `QuantizedLinear` subclasses `Linear`, so type-based dispatch must check `QuantizedLinear` before `Linear` in helpers like `shardLinear`
- `group` stored as plain property (NOT `@ModuleInfo` / `@ParameterInfo`) to exclude from parameter tree

### MLXNN Parameter Discovery
- Plain stored `MLXArray` properties are already discovered by `Module.parameters()`; `@ParameterInfo` is only needed when a parameter needs custom metadata/renaming rather than for ordinary weight/bias storage.

### GPU Limitation
Distributed operations (AllReduce, AllGather, Send, Recv) have **no GPU implementation** -- they must run on CPU. For multi-process distributed code, set `MLX.Device.setDefault(.cpu)`. Single-process tests on size-1 groups work on GPU because identity operations don't actually invoke the distributed primitives. The NN layers must handle this: data may need CPU transfer for collective ops then back to GPU.

### Singleton Group Behavior
- On a size-1 group, `allSum`, `allGather`, `allMax`, `allMin`, and `sumScatter` behave like identity operations.
- `send`, `recv`, and `recvLike` do not have a successful singleton-group path in the current backend; cover those APIs via `withErrorHandler` in single-process tests and use multi-process tests for success-path validation.
- `split` currently has no successful path in any compiled MLX backend (`ring`, `jaccl`, `nccl`) regardless of group size. Tests can validate error surfacing and parent-group recovery after a failed split attempt, but they cannot validate split-child success semantics until upstream backend support exists.
- The localhost `ring` backend used by this repo's multi-process tests does **not** currently implement multi-process `ReduceScatter` / `sumScatter`. Tests can validate graceful error surfacing for that path, but they cannot prove the scattered result until upstream backend support lands.
- `averageGradients(...)` returns immediately when `group.size == 1`, so singleton-group tests only validate the identity fast path. Coverage for `communicationType`, mixed-dtype fallback, or batching behavior must use a multi-rank setup (or other instrumentation) that bypasses the early return.

### JACCL Testing Limitations

JACCL (Joint Accelerator Communication Library) cannot be tested in CI or on most developer machines because it requires all of the following:
- **macOS 26.2 or later** (JACCL APIs were introduced in this version)
- **Thunderbolt 5 hardware** with RDMA-capable network interfaces (currently only Apple M4 Mac mini/MacBook Pro with TB5 ports connected to TB5 peers)
- **RDMA explicitly enabled** in Recovery Mode via `csrutil enable --rdma` (disabled by default)

When these requirements are not met, `MLXDistributed.isAvailable()` still returns `true` because the ring backend (TCP sockets) is always available as a fallback. There is no public MLX-C API to query which specific backend was selected, so tests cannot distinguish "ring is available" from "JACCL is available."

**Testing strategy:**
- `testJACCLAvailability` verifies `isAvailable()` returns `true` (ring backend) without crashing, and documents that JACCL requires the hardware/software prerequisites above.
- All multi-process tests use the ring backend on localhost. JACCL multi-process tests would require two TB5-connected Macs.
- Full JACCL validation requires a manual test lab with TB5-connected hardware running macOS 26.2+.

### MLX-C Gaps
1. `mlx_distributed_init()` has no backend parameter (C++ has `bk` string). Filed as issue on ml-explore/mlx-c. Workaround: compile desired backends; `"any"` picks first available.
2. `mlx_distributed_group_free()` is not publicly exposed in MLX-C v0.5.0. The private inline helper exists in `mlx/c/private/distributed_group.h` but is C++-only. Groups are singleton-like and long-lived, so practical impact is minimal. Should file upstream issue.

### Multi-Process Test Harness Notes

- The ring backend can finish the distributed operation, emit valid JSON, and then hang during socket/C++ destructor cleanup while the child process exits.
- The current test harness mitigates that by draining stdout/stderr asynchronously, accepting timed-out workers as success when they already emitted valid JSON, and flushing output before the worker terminates with `_exit(0)`.
- Deterministic high-port allocation, launch staggering, brief socket cleanup delays, and retry-on-timeout are the current anti-flake patterns for localhost multi-process tests in this repo.
