# User Testing

Testing surface, resource cost classification, and validation approach.

---

## Validation Surface

This is a **library** project with no GUI, CLI, or web interface. The user-facing surface is:
- **Build**: `xcodebuild build -scheme mlx-swift-Package -destination 'platform=macOS'`
- **Tests**: `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS'`

All validation is through automated tests (XCTest) and build success verification.

**No agent-browser or interactive testing needed.**

## Validation Concurrency

- **Machine**: Apple M1 Max, 32GB RAM, 10 cores
- **Build time**: ~1 minute
- **Test time**: ~30 seconds (507 tests)
- **Max concurrent validators**: 1 (xcodebuild locks DerivedData)

Since xcodebuild uses exclusive access to DerivedData and the test suite is fast (~30s), running validators sequentially is efficient. No parallelization needed.

## Test Patterns

- XCTest with `XCTestCase` subclasses
- `setDefaultDevice()` in `override class func setUp()`
- Custom `assertEqual(_:_:rtol:atol:)` for float comparisons
- `@testable import MLX` and `@testable import MLXNN`

## Multi-Process Test Infrastructure

Multi-process tests (VAL-DIST-012/013/014) require:
1. A compiled helper binary that imports MLX and performs distributed operations
2. Foundation `Process` to spawn children with env vars
3. Temp hostfile for ring backend: `[["127.0.0.1:port1"], ["127.0.0.1:port2"]]`
4. 30-second timeout with process termination on timeout
5. Port selection must avoid conflicts (use ephemeral ports or fixed high ports)
