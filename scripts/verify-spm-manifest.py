#!/usr/bin/env python3

import json
import os
import platform
import subprocess


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


# On macOS, /usr/bin/python3 is an xcrun shim that injects SDKROOT into the
# environment (possibly pointing at the CommandLineTools SDK).  That SDK may
# not match the active Xcode's swift compiler, so drop it and let swift pick
# the SDK matching its own toolchain -- same as running from the shell.
env = dict(os.environ)
env.pop("SDKROOT", None)

# Capture only stdout (the JSON); let stderr flow to the console so that
# SwiftPM errors are visible in CI logs.
result = subprocess.run(
    ["swift", "package", "dump-package"],
    stdout=subprocess.PIPE,
    text=True,
    env=env,
)
if result.returncode != 0:
    raise SystemExit(
        f"`swift package dump-package` failed with exit code {result.returncode}"
    )
manifest = json.loads(result.stdout)

targets = {target["name"]: target for target in manifest["targets"]}
dependencies = {
    dependency["sourceControl"][0]["identity"]
    for dependency in manifest["dependencies"]
    if "sourceControl" in dependency
}

cuda_targets = {"CudaBuild", "encuda"}
cuda_plugin_usages = targets["Cmlx"].get("pluginUsages") or []

if platform.system() == "Linux":
    require(
        cuda_targets <= targets.keys(), "Linux manifest is missing CUDA build targets"
    )
    require(
        "swift-argument-parser" in dependencies,
        "Linux manifest is missing the CUDA helper dependency",
    )
    require(
        cuda_plugin_usages,
        "Linux Cmlx target is missing the CUDA build plugin",
    )
else:
    require(
        cuda_targets.isdisjoint(targets),
        "non-Linux manifest unexpectedly contains CUDA build targets",
    )
    require(
        "swift-argument-parser" not in dependencies,
        "non-Linux manifest unexpectedly contains the CUDA helper dependency",
    )
    require(
        not cuda_plugin_usages,
        "non-Linux Cmlx target unexpectedly uses the CUDA build plugin",
    )

print(f"SwiftPM manifest CUDA containment passed for {platform.system()}.")
