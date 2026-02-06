# Wired Memory Management Reference

MLX Swift includes a process-wide wired-memory coordinator for concurrent GPU workloads.

## Core Types

- `WiredMemoryManager`: coordinates wired-limit updates and admission control.
- `WiredMemoryTicket`: a handle representing a memory demand.
- `WiredMemoryPolicy`: computes desired limits and optional admission gating.
- `WiredMemoryEvent`: debug event stream emitted by the manager (DEBUG builds).

## Active vs Reservation Tickets

Use two ticket kinds:

- `.active`: raises wired memory while work is running.
- `.reservation`: participates in admission and sizing but does not keep wired memory elevated while idle.

```swift
import MLX

let policy = WiredSumPolicy()

// Reserve model weights.
let weights = policy.ticket(size: weightsBytes, kind: .reservation)
_ = await weights.start()

// Raise wired memory only while inference is active.
let request = policy.ticket(size: kvCacheBytes, kind: .active)
try await request.withWiredLimit {
    // run inference
}

_ = await weights.end()
```

## Policy Grouping and Built-in Policies

Policies are grouped by `id`. The manager computes one limit per policy group, then applies the maximum across groups.

Built-ins:

- `WiredSumPolicy`: baseline + sum(active ticket sizes)
- `WiredMaxPolicy`: baseline + max(active ticket size)

Custom policy example:

```swift
struct CappedPolicy: WiredMemoryPolicy, Hashable {
    let capDelta: Int

    func limit(baseline: Int, activeSizes: [Int]) -> Int {
        baseline + activeSizes.reduce(0, +)
    }

    func canAdmit(baseline: Int, activeSizes: [Int], newSize: Int) -> Bool {
        activeSizes.reduce(0, +) + max(0, newSize) <= capDelta
    }
}
```

## Hysteresis and Configuration

`WiredMemoryManagerConfiguration` controls shrink behavior:

- `shrinkThresholdRatio`: minimum fractional drop before shrinking.
- `shrinkCooldown`: minimum time between shrink attempts.
- `policyOnlyWhenUnsupported`: keep policy/admission logic active on unsupported backends.
- `baselineOverride`: manual baseline for policy-only mode.
- `useRecommendedWorkingSetWhenUnsupported`: use `GPU.maxRecommendedWorkingSetBytes()` as baseline when possible.

```swift
await WiredMemoryManager.shared.updateConfiguration { config in
    config.shrinkThresholdRatio = 0.25
    config.shrinkCooldown = 1.0
}
```

## Debug Event Stream

In DEBUG builds, inspect coordination behavior:

```swift
let stream = await WiredMemoryManager.shared.events()
for await event in stream {
    print(event.kind, event.appliedLimit ?? -1)
}
```

In release builds, `events()` returns an empty stream.

## CPU and Unsupported Backends

When wired memory control is unavailable, the manager can still enforce policy admission and budgeting:

```swift
await WiredMemoryManager.shared.updateConfiguration { config in
    config.policyOnlyWhenUnsupported = true
    config.baselineOverride = 8 * 1024 * 1024 * 1024
}
```

Baseline resolution on unsupported backends:

1. `baselineOverride` (if set)
2. `GPU.maxRecommendedWorkingSetBytes()` (if available)
3. `0`

## Migration From Deprecated APIs

Prefer ticket-based APIs now:

```swift
// OLD (deprecated)
try await Memory.withWiredLimit(bytes) {
    try await runInference()
}

// NEW
let ticket = WiredMemoryTicket(size: bytes, policy: WiredSumPolicy(), kind: .active)
try await ticket.withWiredLimit {
    try await runInference()
}
```

Notes:

- `GPU.withWiredLimit(...)` is deprecated.
- `Memory.withWiredLimit(...)` is deprecated.
- The synchronous `Memory.withWiredLimit` overload is a deprecated no-op.
