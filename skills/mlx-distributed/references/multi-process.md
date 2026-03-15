# Multi-Process Distributed Execution Guide

Guide for setting up multi-process distributed execution with MLX Swift, including the ring backend, JACCL requirements, hostfile format, environment variables, worker process lifecycle, and testing patterns.

## Backends

MLX-C supports two distributed backends. The C layer tries backends in priority order: JACCL first, then ring.

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

> **Note:** MLX-C does not expose a backend selection parameter. You cannot force one backend over the other. If JACCL hardware is present, it will be preferred.

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

These must be set before calling `MLXDistributed.init(strict: true)`.

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
guard let group = MLXDistributed.`init`(strict: true) else {
    fputs("ERROR: Failed to initialize distributed group\n", stderr)
    exit(1)
}

guard group.rank == rank else {
    fputs("ERROR: rank mismatch\n", stderr)
    exit(1)
}
```

### 4. Perform Distributed Operations

```swift
let localData = MLXArray(converting: rank == 0 ? [1.0, 2.0, 3.0] : [4.0, 5.0, 6.0])
let result = MLXDistributed.allSum(localData, group: group)
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
            guard let group = MLXDistributed.`init`(strict: true) else {
                fputs("ERROR: Failed to initialize\n", stderr)
                exit(1)
            }

            // Perform work...
            let data = MLXArray(converting: [Float(rank + 1)])
            let sum = MLXDistributed.allSum(data, group: group)
            eval(sum)

            print("Rank \(rank): sum = \(sum.asArray(Float.self))")

            fflush(stdout)
            fflush(stderr)
            _exit(0)
        }
    }
}
```

---

## Testing Patterns

### Port Allocation

Avoid ephemeral port collisions by using a sequential counter with a random base:

```swift
class DistributedTests: XCTestCase {
    // Random base avoids TIME_WAIT conflicts across test runs
    // Range 15000-28999 avoids well-known ports and macOS ephemeral range (49152-65535)
    private static var nextPort: Int = 15000 + Int.random(in: 0 ..< 7000) * 2

    private func nextAvailablePort() -> Int {
        while true {
            let port = Self.nextPort
            Self.nextPort += 1
            if isPortAvailable(port) {
                return port
            }
        }
    }

    private func isPortAvailable(_ port: Int) -> Bool {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return false }
        defer { close(sock) }

        var reuse: Int32 = 1
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse,
                   socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = UInt32(INADDR_LOOPBACK).bigEndian

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(sock, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bindResult == 0
    }
}
```

### Hostfile Creation

```swift
private func createHostfile(port1: Int, port2: Int) throws -> URL {
    let hostfile = [
        ["127.0.0.1:\(port1)"],
        ["127.0.0.1:\(port2)"],
    ]
    let jsonData = try JSONSerialization.data(
        withJSONObject: hostfile, options: [.prettyPrinted])
    let jsonString = String(data: jsonData, encoding: .utf8)!

    let tempDir = FileManager.default.temporaryDirectory
    let hostfilePath = tempDir.appendingPathComponent(
        "mlx_test_hostfile_\(UUID().uuidString).json")
    try jsonString.write(to: hostfilePath, atomically: true, encoding: .utf8)

    return hostfilePath
}
```

### Process Spawning

Key patterns for spawning worker processes:

1. **Stagger launches**: Rank 0 must start `accept()` before rank 1 calls `connect()`. Add a ~1 second delay.
2. **Async pipe reading**: Read stdout/stderr asynchronously to prevent deadlocks from buffer overflow.
3. **Timeout handling**: Use 30-second timeouts with retry logic for ring backend TCP races.
4. **Cleanup in tearDown**: Track spawned processes and kill orphans.
5. **JSON output**: Workers print results as JSON to stdout for test verification.

```swift
private func spawnWorker(
    workerBinary: URL, rank: Int, hostfilePath: URL,
    operation: String, timeout: TimeInterval
) -> (exitCode: Int32, stdout: String, stderr: String) {
    let process = Process()
    process.executableURL = workerBinary
    process.environment = [
        "MLX_RANK": "\(rank)",
        "MLX_HOSTFILE": hostfilePath.path,
        "MLX_TEST_OP": operation,
        "PATH": ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin",
        "HOME": ProcessInfo.processInfo.environment["HOME"] ?? "/tmp",
        "DYLD_LIBRARY_PATH":
            ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"] ?? "",
        "DYLD_FRAMEWORK_PATH":
            ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] ?? "",
    ]

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    // Read pipe data asynchronously to prevent deadlocks
    var stdoutData = Data()
    var stderrData = Data()
    let dataLock = NSLock()

    stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if !data.isEmpty {
            dataLock.lock()
            stdoutData.append(data)
            dataLock.unlock()
        }
    }

    try! process.run()
    // ... wait with timeout, handle results
}
```

### Socket Cleanup Between Tests

Add a delay in `tearDown` for TCP socket TIME_WAIT cleanup:

```swift
override func tearDown() {
    for process in spawnedProcesses where process.isRunning {
        process.terminate()
        Thread.sleep(forTimeInterval: 0.5)
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
        }
    }
    spawnedProcesses.removeAll()

    // Allow socket cleanup between tests
    Thread.sleep(forTimeInterval: 1.0)
    super.tearDown()
}
```

### Timeout Tolerance

The ring backend can cause timeouts due to TCP socket cleanup blocking `exit()`. If a worker produced valid JSON output before the timeout, treat it as success:

```swift
// If worker produced valid JSON before timeout, treat as success
let trimmedStdout = stdoutStr.trimmingCharacters(in: .whitespacesAndNewlines)
if !trimmedStdout.isEmpty,
   let jsonData = trimmedStdout.data(using: .utf8),
   (try? JSONSerialization.jsonObject(with: jsonData)) != nil {
    return (0, stdoutStr, stderrStr)  // Success despite timeout
}
```

### Port Range Separation

Use different port ranges for different test classes to avoid cross-class collisions:

| Test Class | Port Range |
|------------|------------|
| `DistributedTests` | 15000–28999 |
| `DistributedNNTests` | 35000–48999 |

---

## Error Handling

Use `withErrorHandler` to catch C++ errors from the distributed backend gracefully:

```swift
let errorCaught = BoolBox()
withErrorHandler({ errMsg in
    print("Distributed error: \(errMsg)")
    errorCaught.value = true
}) {
    let result = MLXDistributed.sumScatter(data, group: group)
    eval(result)
}
```

This is essential for:
- `sumScatter` on ring backend (not implemented)
- `group.split()` on ring/JACCL backends (not supported)
- `send`/`recv` on singleton groups (requires size ≥ 2)
