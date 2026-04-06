# Multi-Process Distributed Execution Guide

Guide for setting up multi-process distributed execution with MLX Swift, including the ring backend, JACCL requirements, hostfile format, environment variables, and worker process lifecycle.

## Backends

MLX Swift commonly uses two distributed backends on Apple Silicon: ring and JACCL. When you let MLX choose automatically with `.any`, it follows upstream backend selection order; on typical Apple Silicon setups that means ring is attempted before JACCL unless you explicitly request `.jaccl`.

### Ring Backend (TCP/IP)

The ring backend uses TCP sockets for communication. It is always compiled in and available.

**Requirements:**
- Network connectivity between processes (localhost or LAN)
- A JSON hostfile describing the topology
- Environment variables: `MLX_RANK`, `MLX_HOSTFILE`

### JACCL Backend (RDMA/Thunderbolt 5)

JACCL (Joint Accelerator Communication Library) uses RDMA over Thunderbolt 5 for high-bandwidth, low-latency communication.

**Requirements:**
- macOS 26.2 or later
- Thunderbolt 5 hardware with RDMA-capable NICs
- RDMA explicitly enabled in Recovery Mode (`csrutil`)
- Physical Thunderbolt 5 cable between nodes

> **Note:** You can select a specific backend using the `backend` parameter (e.g., `DistributedGroup(backend: .jaccl)`). Use `DistributedGroup()` or `DistributedGroup(backend: .any)` to let MLX choose automatically.

---

## Hostfile Format

The ring backend reads a JSON hostfile to discover peers. The file contains an array of arrays, where each inner array contains a single `host:port` string.

```json
[
    ["127.0.0.1:15000"],
    ["127.0.0.1:15001"]
]
```

For a multi-machine setup:
```json
[
    ["192.168.1.10:15000"],
    ["192.168.1.11:15000"]
]
```

The rank of each process corresponds to its index in the outer array (rank 0 is index 0, rank 1 is index 1, etc.).

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MLX_RANK` | The rank of this process (0-based) | `0`, `1` |
| `MLX_HOSTFILE` | Path to the JSON hostfile | `/tmp/hostfile.json` |

These must be set before calling `try DistributedGroup(strict: .ring)` for ring-backend execution.

```swift
guard let rankStr = ProcessInfo.processInfo.environment["MLX_RANK"],
      let rank = Int(rankStr) else {
    fputs("ERROR: MLX_RANK not set\n", stderr)
    exit(1)
}

guard ProcessInfo.processInfo.environment["MLX_HOSTFILE"] != nil else {
    fputs("ERROR: MLX_HOSTFILE not set\n", stderr)
    exit(1)
}
```

---

## Worker Process Lifecycle

### 1. Read Environment Variables

```swift
let rank = Int(ProcessInfo.processInfo.environment["MLX_RANK"]!)!
```

### 2. Set CPU Device

Distributed operations only have CPU implementations.

```swift
Device.withDefaultDevice(.cpu) {
    runWorker(rank: rank)
}
```

### 3. Initialize Distributed Group (strict)

```swift
let group: DistributedGroup
do {
    group = try DistributedGroup(strict: .ring)
    guard group.rank == rank else {
        fputs("ERROR: rank mismatch\n", stderr)
        exit(1)
    }
} catch {
    fputs("ERROR: Failed to initialize distributed group: \(error)\n", stderr)
    exit(1)
}
```

### 4. Perform Distributed Operations

```swift
let localData = MLXArray(converting: rank == 0 ? [1.0, 2.0, 3.0] : [4.0, 5.0, 6.0])
let result = group.allSum(localData)
eval(result)
```

### 5. Flush Output and Exit with _exit(0)

```swift
fflush(stdout)
fflush(stderr)

// CRITICAL: Use _exit(0) instead of exit(0)
// The ring backend's TCP sockets can block in their destructor waiting for
// peer socket closure, causing exit(0) (which runs atexit handlers and C++
// destructors) to hang indefinitely.
_exit(0)
```

### Complete Worker Example

```swift
import Foundation
import MLX
import MLXNN

@main
struct DistributedWorker {
    static func main() {
        guard let rankStr = ProcessInfo.processInfo.environment["MLX_RANK"],
              let rank = Int(rankStr) else {
            fputs("ERROR: MLX_RANK not set\n", stderr)
            exit(1)
        }

        guard ProcessInfo.processInfo.environment["MLX_HOSTFILE"] != nil else {
            fputs("ERROR: MLX_HOSTFILE not set\n", stderr)
            exit(1)
        }

        Device.withDefaultDevice(.cpu) {
            do {
                let group = try DistributedGroup(strict: .ring)

                // Perform work...
                let data = MLXArray(converting: [Float(rank + 1)])
                let sum = group.allSum(data)
                eval(sum)

                print("Rank \(rank): sum = \(sum.asArray(Float.self))")

                fflush(stdout)
                fflush(stderr)
                _exit(0)
            } catch {
                fputs("ERROR: Failed to initialize: \(error)\n", stderr)
                exit(1)
            }
        }
    }
}
```

---

## Error Handling

Use normal Swift `try` / `catch` for call-time failures such as strict init,
`split`, `send`, `recv`, and `recvLike`. Use `withError { ... }` plus
`checkedEval(...)` for lazy evaluation-time failures:

```swift
do {
    try withError {
        let result = try group.sumScatter(data)
        try checkedEval(result)
    }
} catch {
    print("Distributed error: \(error)")
}
```

This is essential for:
- `sumScatter` on ring backend (lazy eval-time failure)
- `group.split()` on ring/JACCL backends (call-time throw)
- `send`/`recv` on singleton groups or invalid ranks (call-time throw)
