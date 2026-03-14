---
name: swift-library-worker
description: Worker for Swift library features - compilation changes, C interop bindings, and tests
---

# Swift Library Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for features that involve:
- Package.swift modifications (exclude list changes)
- Swift bindings wrapping MLX-C functions
- Single-process and multi-process test development
- Build verification features

## Work Procedure

### 1. Read Context

- Read `skills/mlx-swift/SKILL.md` and relevant reference files under `skills/mlx-swift/references/`
- Read the feature description, preconditions, expectedBehavior, and verificationSteps carefully
- Read `.factory/library/architecture.md` for architectural patterns
- Read `.factory/library/environment.md` for environment details
- Identify the MLX-C headers you need: `Source/Cmlx/include/mlx/c/distributed.h` and `distributed_group.h`

### 2. Write Tests First (TDD)

Before implementing anything:
- Create the test file (e.g., `Tests/MLXTests/DistributedTests.swift`)
- Write test cases that match the feature's expectedBehavior
- Follow existing test patterns: `XCTestCase` subclass, `setDefaultDevice()` in setUp
- Use `assertEqual` or `XCTAssertEqual` for comparisons
- Run `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests` to confirm tests fail (red)

### 3. Implement

- Follow the enum namespace pattern for `MLXDistributed` (like `MLXRandom` in `Source/MLX/Random.swift`)
- Follow the C handle wrapping pattern (like `Device` in `Source/MLX/Device.swift`)
- Every C function call follows:
  ```swift
  var result = mlx_array_new()
  mlx_distributed_all_sum(&result, array.ctx, group.ctx, stream.ctx)
  return MLXArray(result)
  ```
- Match the file header style from existing files
- Use `StreamOrDevice = .default` as last parameter

### 4. For Package.swift Changes

- ONLY modify the exclude list -- do not change targets, products, or dependencies
- When un-excluding a file, also exclude its stub (e.g., un-exclude `ring.cpp`, exclude `no_ring.cpp`)
- Keep `no_mpi.cpp` and `no_nccl.cpp` compiled (MPI and NCCL stay disabled)
- After changes, run full build AND full test suite to verify no regressions

### 5. Verify

- Run `xcodebuild build -scheme mlx-swift-Package -destination 'platform=macOS'` (must succeed)
- Run `xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS'` (all tests must pass)
- Verify new tests are green
- Check for compiler warnings in new code

### 6. Manual Verification

- For binding features: verify each Swift function signature matches the MLX-C header
- For compilation features: verify the build output shows no duplicate symbols
- For multi-process tests: verify both processes complete and produce correct results

## Example Handoff

```json
{
  "salientSummary": "Created DistributedGroup class and MLXDistributed enum with all 8 collective operations wrapping MLX-C distributed API. Wrote 15 test cases covering lifecycle, single-process identity ops, dtype handling, and stream parameter. xcodebuild test passes with 522 tests (15 new), 0 failures.",
  "whatWasImplemented": "Source/MLX/Distributed.swift: DistributedGroup class (init, deinit, rank, size, split) + MLXDistributed enum (isAvailable, init, allSum, allGather, allMax, allMin, sumScatter, send, recv, recvLike). All functions follow the mlx_array_new() + mlx_distributed_* + MLXArray(result) pattern with StreamOrDevice parameter.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "xcodebuild build -scheme mlx-swift-Package -destination 'platform=macOS'", "exitCode": 0, "observation": "BUILD SUCCEEDED, no warnings in Distributed.swift"},
      {"command": "xcodebuild test -scheme mlx-swift-Package -destination 'platform=macOS' -only-testing:MLXTests", "exitCode": 0, "observation": "522 tests, 0 failures (15 new distributed tests)"}
    ],
    "interactiveChecks": [
      {"action": "Compared each Swift function signature against MLX-C distributed.h", "observed": "All 8 collective ops + 5 group management functions have matching Swift wrappers"},
      {"action": "Verified DistributedGroup.deinit calls correct free function", "observed": "deinit calls mlx_free(ctx) matching Device.swift pattern"}
    ]
  },
  "tests": {
    "added": [
      {"file": "Tests/MLXTests/DistributedTests.swift", "cases": [
        {"name": "testGroupLifecycle", "verifies": "Create group, access rank/size, deinit without crash"},
        {"name": "testIsAvailable", "verifies": "isAvailable returns true with ring backend"},
        {"name": "testInitSingletonGroup", "verifies": "init returns rank=0, size=1"},
        {"name": "testAllSumIdentity", "verifies": "allSum on size-1 group returns input"},
        {"name": "testAllGatherIdentity", "verifies": "allGather on size-1 group returns input"},
        {"name": "testMultipleDtypes", "verifies": "allSum with float16 and int32 preserves dtype"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- MLX-C header is missing a function you need
- Build fails due to C++ compilation errors in submodule code (cannot modify)
- Existing tests start failing for unclear reasons
- Multi-process test infrastructure design needs architectural decisions
