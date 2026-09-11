# Copyright © 2026 Apple Inc.

"""
Compare default argument values between python `mlx` and Swift `MLX`.

A default that differs is a bug: the same call with no optional arguments should
produce the same result in both languages.  Both free functions (`mx.*`) and the
initializers of `nn` layers and optimizers are compared.  This found three so far
-- `nanToNum`'s `posInf`/`negInf` (`0` vs the dtype max/min), `tensordot`'s `axes`
(`1` vs `2`) and `Lion`'s `betas` (`(0.9, 0.999)` vs python's `(0.9, 0.99)`).

    python3 tools/audit_defaults.py [--json rows.json] [--verbose]

Exits non-zero when a mismatch is reported, so it can be wired into CI.

`rows.json` is the output of `tools/audit_integration_coverage.py --json`, which
supplies the python -> Swift name mapping; it is generated automatically when not
supplied.  Needs the python mlx stubs on `sys.path`; nothing is imported from
mlx, so no GPU is required.

Differences that are only spelling are normalized (`None`/`nil`, `1e-08`/`1e-8`,
`0.0`/`0`, `'affine'`/`.affine`).  Differences that are real but intentional are
listed in `ACKNOWLEDGED` with a reason, so the default output is empty when the
API agrees.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# python module -> stub file
MODULES = {
    "mx": "core/__init__.pyi",
    "mx.fft": "core/fft.pyi",
    "mx.linalg": "core/linalg.pyi",
    "mx.random": "core/random.pyi",
    "mx.fast": "core/fast.pyi",
}

# python parameter name -> Swift label, where they differ by more than case
PARAMETER_ALIASES = {
    "keepdims": "keepDims",
    "group_size": "groupSize",
    "posinf": "posInf",
    "neginf": "negInf",
    "equal_nan": "equalNaN",
    "num_samples": "count",
    "allow_col_major": "allowColMajor",
    "output_padding": "outputPadding",
    "kernel_dilation": "kernelDilation",
    "input_dilation": "inputDilation",
    "label_smoothing": "labelSmoothing",
    "with_logits": "withLogits",
}

# python class name -> Swift class name, where they differ
CLASS_ALIASES = {
    "Adagrad": "AdaGrad",
    "ConvTranspose1d": "ConvTransposed1d",
    "ConvTranspose2d": "ConvTransposed2d",
    "ConvTranspose3d": "ConvTransposed3d",
    "ReLU2": "ReLUSquared",
    "Hardswish": "HardSwish",
}

# python parameter -> Swift label for class initializers
CLASS_PARAMETER_ALIASES = {
    "negative_slope": "negativeSlope",
    "num_parameters": "count",
    "init": "value",
    "lambd": "lambda",
    "min_val": "min",
    "max_val": "max",
    "num_groups": "groupCount",
    "num_features": "featureCount",
    "num_embeddings": "embeddingCount",
    "dims": "dimensions",
    "min_freq": "minFrequency",
    "max_freq": "maxFrequency",
    "cos_first": "cosineFirst",
    "full_turns": "fullTurns",
    "pytorch_compatible": "pytorchCompatible",
    "track_running_stats": "trackRunningStats",
    "kernel_size": "kernelSize",
    "in_channels": "inputChannels",
    "out_channels": "outputChannels",
    "output_padding": "outputPadding",
    "num_heads": "numHeads",
    "mlp_dims": "mlpDimensions",
    "input_size": "inputSize",
    "hidden_size": "hiddenSize",
    "scale_factor": "scaleFactor",
    "align_corners": "alignCorners",
    "weight_decay": "weightDecay",
    "decay_rate": "decayRate",
    "clip_threshold": "clipThreshold",
    "beta_1": "beta1",
    "scale_parameter": "scaleParameter",
    "relative_step": "relativeStep",
    "warmup_init": "warmupInit",
    "ns_steps": "nsSteps",
    "bias_correction": "biasCorrection",
}

# (python class, python parameter) -> why the difference is intentional
ACKNOWLEDGED_CLASSES = {
    ("QuantizedLinear", "bits"): "python passes None through to the C++ default (4)",
    ("QuantizedEmbedding", "bits"): "python passes None through to the C++ default (4)",
    ("QuantizedLinear", "group_size"): "python passes None through to the C++ default (64)",
    ("QuantizedEmbedding", "group_size"): "python passes None through to the C++ default (64)",
    ("Transformer", "activation"): "python takes the relu function, Swift a ReLU() layer",
    ("Transformer", "norm_first"): "checked in the layer tests",
    ("Upsample", "mode"): "Swift folds align_corners into the mode enum",
    ("Upsample", "align_corners"): "Swift folds align_corners into the mode enum",
}


# (python function, python parameter) -> why the difference is intentional
ACKNOWLEDGED = {
    ("normal", "loc"): "python None means 0",
    ("normal", "scale"): "python None means 1",
    ("linspace", "dtype"): "Swift nil resolves to float32 (or the bounds' float type)",
    ("zeros", "dtype"): "Swift nil resolves to float32",
    ("ones", "dtype"): "Swift nil resolves to float32",
    ("full", "dtype"): "Swift nil resolves from the fill value",
    ("uniform", "shape"): "both mean 'scalar unless a shape is given'",
    ("randint", "shape"): "both mean 'scalar unless a shape is given'",
    ("bernoulli", "shape"): "both mean 'scalar unless a shape is given'",
    ("truncated_normal", "shape"): "both mean 'scalar unless a shape is given'",
    ("vmap", "in_axes"): "Swift takes an array of axes",
    ("vmap", "out_axes"): "Swift takes an array of axes",
    ("compile", "inputs"): "Swift uses an empty array rather than None",
    ("compile", "outputs"): "Swift uses an empty array rather than None",
    ("load", "stream"): "IO defaults to the CPU in Swift",
}

OPTIONAL_SENTINELS = {"nil", "[Int]?.none", "MLXArray?.none", "[Int]()"}


def find_mlx_python() -> pathlib.Path:
    for entry in sys.path:
        candidate = pathlib.Path(entry) / "mlx" / "core" / "__init__.pyi"
        if candidate.exists():
            return candidate.parent.parent
    sys.exit("could not find the installed mlx python package on sys.path")


def split_parameters(text: str) -> list[str]:
    out, depth, current = [], 0, ""
    for character in text:
        if character in "[({<":
            depth += 1
        if character in "])}>":
            depth -= 1
        if character == "," and depth == 0:
            out.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        out.append(current.strip())
    return out


def defaults_from(text: str) -> dict[str, str]:
    out = {}
    for parameter in split_parameters(text):
        if parameter in ("/", "*") or parameter.startswith("*") or "=" not in parameter:
            continue
        name = parameter.split(":")[0].split("=")[0].strip()
        # `_ label: Type = value` -> the label is what callers write
        name = name.split(" ")[-1] if name.startswith("_ ") else name.split(" ")[0]
        out[name] = parameter.split("=", 1)[1].strip()
    return out


def python_signature(name: str, module: str, mlx: pathlib.Path) -> str | None:
    text = (mlx / module).read_text()
    match = re.search(rf"^def {re.escape(name)}\(([^\n]*?)\)\s*->", text, re.M)
    return match.group(1) if match else None


def swift_signatures() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted((ROOT / "Source").rglob("*.swift")):
        if any(skip in str(path) for skip in ("/Cmlx/", "/Examples/", "/Encuda/")):
            continue
        lines = path.read_text().split("\n")
        for i, line in enumerate(lines):
            match = re.match(r"^(?:public|open)\s+func ([A-Za-z_]\w*)(.*)$", line)
            if not match:
                continue
            signature, j = match.group(2), i
            while "{" not in signature and j + 1 < len(lines) and len(signature) < 600:
                j += 1
                signature += " " + lines[j].strip()
            signature = re.sub(r"\s+", " ", signature.split("{")[0])
            out.setdefault(match.group(1), []).append(signature)
    return out


def python_class_defaults(mlx: pathlib.Path) -> dict[str, dict[str, str]]:
    """class -> {parameter: default}, following single inheritance for __init__"""
    classes: dict[str, dict[str, str]] = {}
    bases: dict[str, list[str]] = {}

    for path in list((mlx / "optimizers").rglob("*.py")) + list((mlx / "nn").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
            for item in node.body:
                if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
                    continue
                names = [a.arg for a in item.args.args][1:]
                defaults = [ast.unparse(d) for d in item.args.defaults]
                mapping = dict(zip(names[len(names) - len(defaults) :], defaults))
                for argument, default in zip(item.args.kwonlyargs, item.args.kw_defaults):
                    if default is not None:
                        mapping[argument.arg] = ast.unparse(default)
                classes[node.name] = mapping

    def resolve(name: str, seen: tuple[str, ...] = ()) -> dict[str, str]:
        if name in seen:
            return {}
        mapping = dict(classes.get(name, {}))
        if not mapping:
            for base in bases.get(name, []):
                mapping.update(resolve(base, seen + (name,)))
        return mapping

    return {name: resolve(name) for name in classes}


def swift_class_defaults() -> dict[str, dict[str, str]]:
    """class -> {label: default} from the first public initializer"""
    classes: dict[str, dict[str, str]] = {}

    for directory in ("Source/MLXOptimizers", "Source/MLXNN"):
        for path in sorted((ROOT / directory).rglob("*.swift")):
            lines = path.read_text().split("\n")
            current: str | None = None
            for i, line in enumerate(lines):
                match = re.match(
                    r"^\s*(?:open|public|final public|public final)\s+class (\w+)", line
                )
                if match:
                    current = match.group(1)
                if current is None or not re.match(r"^\s+public init\(", line):
                    continue
                # accumulate until the parentheses balance: a signature like
                # `betas: (Float, Float) = (0.9, 0.999)` closes several times
                signature, j = line, i
                while j + 1 < len(lines) and len(signature) < 800:
                    body = signature.split("init(", 1)[1]
                    if body.count("(") + 1 == body.count(")"):
                        break
                    j += 1
                    signature += " " + lines[j].strip()

                body = signature.split("init(", 1)[1]
                depth, end = 1, None
                for index, character in enumerate(body):
                    if character == "(":
                        depth += 1
                    elif character == ")":
                        depth -= 1
                        if depth == 0:
                            end = index
                            break
                if end is None:
                    continue
                classes.setdefault(current, defaults_from(body[:end]))

    return classes


def check_classes(mlx: pathlib.Path, verbose: bool) -> tuple[list[str], list[str]]:
    """compare initializer defaults of nn layers and optimizers"""
    python = python_class_defaults(mlx)
    swift = swift_class_defaults()

    mismatches, acknowledged = [], []
    for name, parameters in sorted(python.items()):
        swift_name = CLASS_ALIASES.get(name, name)
        swift_parameters = swift.get(swift_name)
        if swift_parameters is None:
            continue
        for parameter, value in parameters.items():
            label = CLASS_PARAMETER_ALIASES.get(parameter, camel(parameter))
            if label not in swift_parameters:
                continue
            if equivalent(value, swift_parameters[label]):
                continue
            report = (
                f"{name}.{parameter}: python={value} swift={swift_parameters[label]}"
            )
            reason = ACKNOWLEDGED_CLASSES.get((name, parameter))
            (acknowledged if reason else mismatches).append(
                f"{report} ({reason})" if reason else report
            )
    return mismatches, acknowledged


def normalize(value: str) -> str:
    value = value.strip().strip("'\"")
    value = {"None": "nil", "True": "true", "False": "false"}.get(value, value)
    value = re.sub(r"^mx\.", "", value).lstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value
    return repr(number)


def _sequence(value: str) -> list[str] | None:
    text = value.strip()
    if not (text.startswith(("(", "[")) and text.endswith((")", "]"))):
        return None
    return [normalize(part) for part in text[1:-1].split(",") if part.strip()]


def equivalent(python_value: str, swift_value: str) -> bool:
    if normalize(python_value) == normalize(swift_value):
        return True
    if normalize(python_value) == "nil" and swift_value.strip() in OPTIONAL_SENTINELS:
        return True
    # `[0.9, 0.999]` in python vs `(0.9, 0.999)` in Swift
    left, right = _sequence(python_value), _sequence(swift_value)
    return left is not None and left == right


def camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def main() -> int:
    verbose = "--verbose" in sys.argv
    mlx = find_mlx_python()

    if "--json" in sys.argv:
        rows_path = pathlib.Path(sys.argv[sys.argv.index("--json") + 1])
    else:
        rows_path = pathlib.Path("/tmp/mlx-swift-coverage-rows.json")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/audit_integration_coverage.py"),
                "--json", str(rows_path),
                "--out", "/dev/null",
            ],
            check=True,
            capture_output=True,
        )
    rows = json.load(open(rows_path))
    swift = swift_signatures()

    mismatches: list[str] = []
    acknowledged: list[str] = []
    for row in rows:
        if row["group"] not in MODULES or row["kind"] != "free" or not row["swift"]:
            continue
        signature = python_signature(row["py"], MODULES[row["group"]], mlx)
        if signature is None:
            continue
        python_defaults = defaults_from(signature)
        if not python_defaults:
            continue

        for swift_signature in swift.get(row["swift"], [])[:4]:
            if "(" not in swift_signature:
                continue
            inner = swift_signature[swift_signature.index("(") + 1 : swift_signature.rindex(")")]
            swift_defaults = defaults_from(inner)
            differences = []
            for name, value in python_defaults.items():
                if name == "stream" and (row["py"], name) not in ACKNOWLEDGED:
                    continue
                label = PARAMETER_ALIASES.get(name, camel(name))
                if label not in swift_defaults:
                    continue
                if equivalent(value, swift_defaults[label]):
                    continue
                report = (
                    f"{row['py']} -> {row['swift']}: {name}: "
                    f"python={value} swift={swift_defaults[label]}"
                )
                reason = ACKNOWLEDGED.get((row["py"], name))
                (acknowledged if reason else differences).append(
                    f"{report} ({reason})" if reason else report
                )
            if differences:
                mismatches += differences
                break

    class_mismatches, class_acknowledged = check_classes(mlx, verbose)
    mismatches += class_mismatches
    acknowledged += class_acknowledged

    for line in acknowledged if verbose else []:
        print(f"acknowledged: {line}")
    for line in mismatches:
        print(f"MISMATCH {line}")

    if mismatches:
        print(
            f"\n{len(mismatches)} default(s) differ.  Either fix the Swift default or, "
            "if the difference is intentional, add it to ACKNOWLEDGED with a reason."
        )
        return 1

    print(
        f"defaults agree ({len(acknowledged)} acknowledged difference(s); "
        "pass --verbose to list them)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
