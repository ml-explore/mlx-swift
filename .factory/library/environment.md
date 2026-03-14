# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Build Environment

- **Xcode 26.3** (Build 17C529), Swift 6.2.4
- **macOS 26.3**, Apple M1 Max, 32GB RAM, 10 cores
- Metal shaders require xcodebuild (swift test cannot compile them)
- The active macOS SDK includes `usr/include/infiniband/verbs.h`, so the vendored JACCL sources compile without installing extra RDMA headers on this machine

## Git Submodules

- `Source/Cmlx/mlx` -> `https://github.com/ml-explore/mlx` (tag v0.30.6)
- `Source/Cmlx/mlx-c` -> `https://github.com/ml-explore/mlx-c` (tag v0.5.0)
- Files inside submodules are READ-ONLY

## Distributed Backend Environment Variables (Runtime)

The ring backend uses these env vars:
- `MLX_RANK` -- integer rank of this process
- `MLX_HOSTFILE` -- path to JSON file with host addresses
- `MLX_RING_VERBOSE` -- enable verbose logging

The JACCL backend uses:
- `MLX_RANK` -- integer rank
- `MLX_JACCL_COORDINATOR` -- IP:port of coordinator
- `MLX_IBV_DEVICES` -- JSON device connectivity file
- Requires macOS 26.2+ and Thunderbolt 5 hardware with RDMA enabled
