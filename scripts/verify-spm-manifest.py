#!/usr/bin/env python3

import json
import platform
import subprocess


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


manifest = json.loads(
    subprocess.run(
        ["swift", "package", "dump-package"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
)

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
