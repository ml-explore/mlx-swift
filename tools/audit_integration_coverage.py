# Copyright © 2026 Apple Inc.

"""
Audit integration-test coverage of the MLX Swift API.

This compares three things:

1. the python `mlx` API (read from the installed type stubs / python sources --
   *not* imported, so this runs on machines without a GPU)
2. the public Swift API in `Source/` (parsed, not compiled)
3. the symbols covered by the generator's case tables
   (`tools/integration_tests/cases.py` and `modules.py`, which produce
   `Tests/MLXIntegrationTests/Generated/`) and the symbols referenced by the hand
   written tests in `Tests/`

and writes a markdown report: `tools/integration-coverage-report.md`.

Coverage state per symbol:

- `integration` -- covered by a generated case (has a python-vs-swift value check)
- `unit-only`   -- referenced by some other test, but no python comparison
- `none`        -- not referenced by any test
- `no-swift`    -- exists in python, no obvious Swift counterpart (API gap, not a test gap)

Usage:

    python3 tools/audit_integration_coverage.py [--mlx-python PATH] [--out PATH]

`--mlx-python` should point at the installed `mlx` package directory (the one
containing `core/__init__.pyi`).  It defaults to the `mlx` found on `sys.path`.

Caveats: the mapping from python names to Swift names is heuristic (normalized
name match + the table in `Source/MLX/Documentation.docc/Articles/converting-python.md`
+ a manual alias table below).  Treat the report as a work list, not gospel; the
`MANUAL_ALIASES` table is the place to fix bad matches.
"""

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

SKIP_SOURCE = ("/Cmlx/", "/Examples/", "/Encuda/", "/CompileLockRepro/")

# generated integration tests and the hand written support beside them
GENERATED_DIRECTORY = "Tests/MLXIntegrationTests/Generated"
SUPPORT_FILE = "Tests/MLXIntegrationTests/IntegrationSupport.swift"

# python name -> Swift name, for cases the automatic match cannot find.
# (the automatic match handles snake_case -> camelCase and the table in
# converting-python.md)
MANUAL_ALIASES = {
    # ops
    "arctan2": "atan2",
    "conj": "conjugate",
    "concat": "concatenated",
    "concatenate": "concatenated",
    "tile": "tiled",
    "flip": "flipped",
    "unstack": "unstacked",
    "real": "realPart",
    "imag": "imaginaryPart",
    "put_along_axis": "putAlong",
    "take_along_axis": "takeAlong",
    "unflatten": "unflattened",
    "conv_transpose1d": "convTransposed1d",
    "conv_transpose2d": "convTransposed2d",
    "conv_transpose3d": "convTransposed3d",
    "conv_general": "convGeneral",
    "gather_qmm": "gatherQuantizedMM",
    "gather_mm": "gatherMM",
    "block_masked_mm": "blockMaskedMM",
    "segmented_mm": "segmentedMM",
    "addmm": "addMM",
    "meshgrid": "meshGrid",
    "nan_to_num": "nanToNum",
    "isnan": "isNaN",
    "isinf": "isInf",
    "isfinite": "isFinite",
    "isposinf": "isPosInf",
    "isneginf": "isNegInf",
    "isclose": "isClose",
    "allclose": "allClose",
    "array_equal": "arrayEqual",
    "broadcast_to": "broadcast",
    "expand_dims": "expandedDimensions",
    "atleast_1d": "atLeast1D",
    "atleast_2d": "atLeast2D",
    "atleast_3d": "atLeast3D",
    "left_shift": "leftShift",
    "right_shift": "rightShift",
    "bitwise_and": "bitwiseAnd",
    "bitwise_or": "bitwiseOr",
    "bitwise_xor": "bitwiseXOr",
    "bitwise_invert": "bitwiseInvert",
    "logical_and": "logicalAnd",
    "logical_or": "logicalOr",
    "logical_not": "logicalNot",
    "logaddexp": "logAddExp",
    "logcumsumexp": "logCumsumExp",
    "logsumexp": "logSumExp",
    "stop_gradient": "stopGradient",
    "value_and_grad": "valueAndGrad",
    "custom_function": "customFunction",
    "hadamard_transform": "hadamardTransform",
    "sort": "sorted",
    "argsort": "argSort",
    "partition": "partitioned",
    "argpartition": "argPartition",
    "topk": "top",
    "where": "which",
    "pad": "padded",
    "squeeze": "squeezed",
    "reshape": "reshaped",
    "transpose": "transposed",
    "swapaxes": "swappedAxes",
    "moveaxis": "movedAxis",
    "flatten": "flattened",
    "stack": "stacked",
    "repeat": "repeated",
    "as_strided": "asStrided",
    "astype": "asType",
    "quantized_matmul": "quantizedMM",
    "quantize": "quantized",
    "dequantize": "dequantized",
    "qqmm": "quantizedQuantizedMM",
    "erfinv": "erfInverse",
    "floor_divide": "floorDivide",
    "greater_equal": "greaterEqual",
    "less_equal": "lessEqual",
    "not_equal": "notEqual",
    "power": "pow",
    "tensordot": "tensordot",
    "zeros_like": "zeros",
    "ones_like": "ones",
    "full_like": "full",
    "number_of_elements": "numberOfElements",
    # random / fast
    "multivariate_normal": "multivariateNormal",
    "truncated_normal": "truncatedNormal",
    "randint": "randInt",
    "scaled_dot_product_attention": "scaledDotProductAttention",
    "layer_norm": "layerNorm",
    "rms_norm": "rmsNorm",
    "rope": "RoPE",
    "metal_kernel": "metalKernel",
    # linalg
    "tri_inv": "triInv",
    "cholesky_inv": "choleskyInv",
    "eigvalsh": "eigValsh",
    "eigvals": "eigVals",
    "lu_factor": "luFactor",
    "solve_triangular": "solveTriangular",
    # nn
    "ConvTranspose1d": "ConvTransposed1d",
    "ConvTranspose2d": "ConvTransposed2d",
    "ConvTranspose3d": "ConvTransposed3d",
    "ReLU2": "ReLUSquared",
    "relu2": "reluSquared",
    "gelu_approx": "geluApproximate",
    "gelu_fast_approx": "geluFastApproximate",
    "log_softmax": "logSoftmax",
    "log_sigmoid": "logSigmoid",
    "hard_shrink": "hardShrink",
    "hard_tanh": "hardTanh",
    "hardswish": "hardSwish",
    "leaky_relu": "leakyRelu",
    "softplus": "softplus",
    "softsign": "softsign",
    "softshrink": "softShrink",
    "upsample_nearest": "Upsample",
    # losses
    "cross_entropy": "crossEntropy",
    "binary_cross_entropy": "binaryCrossEntropy",
    "l1_loss": "l1Loss",
    "mse_loss": "mseLoss",
    "nll_loss": "nllLoss",
    "gaussian_nll_loss": "gaussianNLLLoss",
    "kl_div_loss": "klDivLoss",
    "smooth_l1_loss": "smoothL1Loss",
    "triplet_loss": "tripletLoss",
    "hinge_loss": "hingeLoss",
    "huber_loss": "huberLoss",
    "log_cosh_loss": "logCoshLoss",
    "cosine_similarity_loss": "cosineSimilarityLoss",
    "margin_ranking_loss": "marginRankingLoss",
    # optimizers / schedulers
    "Adagrad": "AdaGrad",
    "clip_grad_norm": "clipGradNorm",
    "cosine_decay": "cosineDecay",
    "exponential_decay": "exponentialDecay",
    "step_decay": "stepDecay",
    "join_schedules": "joinSchedules",
    "linear_schedule": "linearSchedule",
}

# python symbols that are intentionally not part of the Swift API (do not report
# them as gaps)
NOT_APPLICABLE = {
    "from_dlpack",
    "printoptions",
    "set_printoptions",
    "get_printoptions",
    "issubdtype",
    "isdtype",
    "can_cast",
    "result_type",
    "savez",
    "savez_compressed",
    "einsum_path",
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------- python side


def find_mlx_python(explicit: str | None) -> pathlib.Path | None:
    if explicit:
        return pathlib.Path(explicit)
    for entry in sys.path:
        candidate = pathlib.Path(entry) / "mlx" / "core" / "__init__.pyi"
        if candidate.exists():
            return candidate.parent.parent
    return None


def pyi_functions(path: pathlib.Path) -> "collections.OrderedDict[str, str]":
    """top level `def name(...)` in a stub file, name -> first signature seen"""
    out = collections.OrderedDict()
    if not path.exists():
        return out
    for m in re.finditer(r"^def ([a-zA-Z_]\w*)\(([^\n]*)", path.read_text(), re.M):
        out.setdefault(m.group(1), re.sub(r"\s+", " ", m.group(2)).rstrip())
    return out


def py_classes(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    return [
        c
        for c in re.findall(r"^class ([A-Za-z_]\w*)\(", path.read_text(), re.M)
        if not c.startswith("_")
    ]


def py_functions(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    return [
        f
        for f in re.findall(r"^def ([a-z_]\w*)\(", path.read_text(), re.M)
        if not f.startswith("_")
    ]


def read_python_api(mlx: pathlib.Path) -> dict:
    api = {
        "mx": pyi_functions(mlx / "core/__init__.pyi"),
        "mx.fft": pyi_functions(mlx / "core/fft.pyi"),
        "mx.linalg": pyi_functions(mlx / "core/linalg.pyi"),
        "mx.random": pyi_functions(mlx / "core/random.pyi"),
        "mx.fast": pyi_functions(mlx / "core/fast.pyi"),
    }

    layers, functions = collections.OrderedDict(), collections.OrderedDict()
    for p in sorted((mlx / "nn/layers").glob("*.py")):
        for c in py_classes(p):
            layers.setdefault(c, p.name)
        for f in py_functions(p):
            functions.setdefault(f, p.name)

    optimizers = collections.OrderedDict()
    for p in sorted((mlx / "optimizers").glob("*.py")):
        for c in py_classes(p):
            optimizers.setdefault(c, p.name)
        for f in py_functions(p):
            optimizers.setdefault(f, p.name)

    return {
        "modules": api,
        "nn.layer": layers,
        "nn.function": functions,
        "nn.loss": {f: "losses.py" for f in py_functions(mlx / "nn/losses.py")},
        "nn.init": {f: "init.py" for f in py_functions(mlx / "nn/init.py")},
        "optim": optimizers,
    }


# ----------------------------------------------------------------- swift side


def read_swift_api() -> dict:
    free: dict[str, list[list[str]]] = {}
    methods: dict[str, list[list[str]]] = {}
    types: dict[str, str] = {}

    for p in sorted((ROOT / "Source").rglob("*.swift")):
        if any(s in str(p) for s in SKIP_SOURCE):
            continue
        rel = str(p.relative_to(ROOT))
        text = p.read_text()
        lines = text.split("\n")

        for m in re.finditer(
            r"^\s*((?:(?:public|open|final|@\w+(?:\([^)]*\))?)\s+)*)"
            r"(class|struct|enum|actor|protocol)\s+([A-Za-z_]\w*)",
            text,
            re.M,
        ):
            if "public" in m.group(1) or "open" in m.group(1):
                types.setdefault(m.group(3), rel)

        for i, line in enumerate(lines):
            m = re.match(
                r"^(\s*)(?:public|open)\s+"
                r"(?:static\s+|mutating\s+|consuming\s+|borrowing\s+|final\s+)*"
                r"func\s+([A-Za-z_]\w*|[-+*/%<>=!.^~]+)(.*)$",
                line,
            )
            if not m:
                continue
            indent, name, rest = m.group(1), m.group(2), m.group(3)
            signature, j = rest, i
            while "{" not in signature and j + 1 < len(lines) and len(signature) < 500:
                j += 1
                signature += " " + lines[j].strip()
            signature = re.sub(r"\s+", " ", signature.split("{")[0].strip())
            target = free if len(indent) == 0 else methods
            target.setdefault(name, []).append([rel, signature])

    return {"free": free, "methods": methods, "types": types}


def generator_case_expressions() -> str:
    """the **Swift** expressions in the tools/integration_tests case tables

    These case tables are the manifest of what the generator emits, and importing
    them needs no mlx, so coverage can be reported before (or without) generating.
    Only the Swift side is returned: the python expressions would match Swift
    symbols by accident (`mx.addmm` looks like the deprecated Swift `addmm`), and
    the generated Swift itself would match argument labels (every generated test
    says `seed:`, which is not a call to `MLXRandom.seed`).
    """
    directory = ROOT / "tools/integration_tests"
    if not (directory / "cases.py").exists():
        return ""
    sys.path.insert(0, str(directory))
    try:
        import cases  # type: ignore
        import modules  # type: ignore
        import optimizers  # type: ignore
    except Exception:  # noqa: BLE001 -- the audit must not depend on the generator
        return ""
    finally:
        sys.path.remove(str(directory))

    fragments: list[str] = []
    for case in cases.CASES:
        fragments.append(case.swift)
        fragments += [spec.swift() for spec in case.inputs.values()]
    for case in modules.MODULE_CASES:
        fragments += [case.swift, case.call_swift]
        fragments += [spec.swift() for spec in case.inputs.values()]
    for case in optimizers.OPTIMIZER_FUNCTION_CASES:
        fragments.append(case.swift)
        fragments += [spec.swift() for spec in case.inputs.values()]
    for case in optimizers.OPTIMIZER_CASES:
        fragments.append(case.swift)
        if case.schedule:
            fragments.append(case.schedule[1])
        fragments += [spec.swift() for spec in case.parameters.values()]
    for case in optimizers.SCHEDULE_CASES:
        fragments.append(case.swift)
    return "\n".join(fragments)


def doc_aliases() -> tuple[dict, set]:
    """python -> swift mapping from Articles/converting-python.md"""
    path = ROOT / "Source/MLX/Documentation.docc/Articles/converting-python.md"
    aliases, unsupported = {}, set()
    if not path.exists():
        return aliases, unsupported
    doc = path.read_text()
    for m in re.finditer(r"^`(\w+)` \| ``(?:MLX|MLXArray)/([A-Za-z0-9_]+)", doc, re.M):
        aliases.setdefault(m.group(1), m.group(2))
    unsupported.update(re.findall(r"^`(\w+)` \| not supported", doc, re.M))
    return aliases, unsupported


# ------------------------------------------------------------------ coverage


class Coverage:
    def __init__(self):
        self.swift = read_swift_api()
        self.aliases, self.unsupported = doc_aliases()

        self.free_n = {}
        for k in self.swift["free"]:
            self.free_n.setdefault(normalize(k), k)
        self.type_n = {}
        for k in self.swift["types"]:
            self.type_n.setdefault(normalize(k), k)
        self.method_n = {}
        for k in self.swift["methods"]:
            self.method_n.setdefault(normalize(k), k)

        tests = {p: p.read_text() for p in (ROOT / "Tests").rglob("*.swift")}

        # Integration coverage is read from the generator's case tables rather
        # than from its output: the tables are the manifest, they need no mlx, and
        # they do not contain the scaffolding (argument labels, support helpers)
        # that a text search over the generated Swift would false-positive on.
        self.integration = generator_case_expressions()

        # everything else in Tests/ is hand written coverage, except the generated
        # files themselves and the support file beside them
        self.unit = "\n".join(
            text
            for path, text in tests.items()
            if GENERATED_DIRECTORY not in str(path) and not str(path).endswith(SUPPORT_FILE)
        )

    def resolve(self, py: str, prefer: str = "free"):
        candidates = []
        if py in MANUAL_ALIASES:
            candidates.append(MANUAL_ALIASES[py])
        if py in self.aliases:
            candidates.append(self.aliases[py])
        candidates.append(py)
        parts = py.split("_")
        candidates.append(parts[0] + "".join(p.capitalize() for p in parts[1:]))
        for c in candidates:
            # exact spelling wins over a normalized match: mlx-swift keeps
            # deprecated spellings around (e.g. `SoftMax` next to `Softmax`)
            if c in self.swift["types"] and prefer == "type":
                return c, "type", [self.swift["types"][c], ""]
            if c in self.swift["free"] and prefer == "free":
                return c, "free", self.swift["free"][c][0]
        # `glu` (free function) and `GLU` (layer) normalize identically, so the
        # caller says which kind of symbol it is looking for first
        order = (
            ["type", "free", "method"] if prefer == "type" else ["free", "type", "method"]
        )
        for c in candidates:
            n = normalize(c)
            for kind in order:
                if kind == "free" and n in self.free_n:
                    name = self.free_n[n]
                    return name, "free", self.swift["free"][name][0]
                if kind == "type" and n in self.type_n:
                    name = self.type_n[n]
                    return name, "type", [self.swift["types"][name], ""]
                if kind == "method" and n in self.method_n:
                    name = self.method_n[n]
                    return name, "method", self.swift["methods"][name][0]
        return None, "no-swift", ["", ""]

    @staticmethod
    def _used(symbol: str, corpus: str) -> bool:
        return bool(symbol) and (
            re.search(
                r"(?<![A-Za-z0-9_])" + re.escape(symbol) + r"(?![A-Za-z0-9_])", corpus
            )
            is not None
        )

    def row(self, group: str, py: str, detail: str = "", prefer: str = "free") -> dict:
        if py in self.unsupported or py in NOT_APPLICABLE:
            return dict(
                group=group, py=py, swift="", kind="n/a", file="", swift_sig="",
                detail=detail, state="n/a",
            )
        swift, kind, (file, sig) = self.resolve(py, prefer=prefer)
        if kind == "no-swift":
            state = "no-swift"
        elif self._used(swift, self.integration):
            state = "integration"
        elif self._used(swift, self.unit):
            state = "unit-only"
        else:
            state = "none"
        return dict(
            group=group, py=py, swift=swift or "", kind=kind, file=file,
            swift_sig=sig, detail=detail, state=state,
        )


def swift_only_rows(cov: Coverage, rows: list[dict]) -> list[dict]:
    """public Swift free functions that no python symbol mapped onto"""
    mapped = {r["swift"] for r in rows if r["swift"]}
    out = []
    for name, locations in sorted(cov.swift["free"].items()):
        if name in mapped or not re.match(r"^[A-Za-z_]", name):
            continue
        file = locations[0][0]
        if not file.startswith("Source/MLX"):
            continue
        state = (
            "integration"
            if cov._used(name, cov.integration)
            else ("unit-only" if cov._used(name, cov.unit) else "none")
        )
        out.append(
            dict(group="swift-only", py="", swift=name, kind="free", file=file,
                 swift_sig=locations[0][1], detail="", state=state)
        )
    return out


# argument labels / common words that produce false positives in a text search
SWIFT_NOISE = {"from", "with", "into", "result", "values", "value", "count"}


def deprecated_usage(cov: "Coverage") -> list[tuple[str, str, str]]:
    """deprecated Swift declarations the generator's case tables still use

    Only names where *every* public declaration is deprecated are reported: a
    name like `fft` also exists non-deprecated in the `MLX` module, so a call in
    the tests is not necessarily the deprecated one.
    """
    declarations: dict[str, list[tuple[bool, str, str]]] = {}
    for p in sorted((ROOT / "Source").rglob("*.swift")):
        if any(s in str(p) for s in SKIP_SOURCE):
            continue
        rel = str(p.relative_to(ROOT))
        lines = p.read_text().split("\n")
        for i, line in enumerate(lines):
            m = re.match(
                r"^\s*(?:(?:public|open|final|static|@\w+(?:\([^)]*\))?)\s+)*"
                r"(?:func|class|struct|enum|typealias)\s+([A-Za-z_]\w*)",
                line,
            )
            if not m or not re.search(r"\b(public|open)\b", line):
                continue
            note = ""
            for k in range(max(0, i - 5), i):
                if "@available" in lines[k] and "deprecated" in lines[k]:
                    note = lines[k].strip()
            declarations.setdefault(m.group(1), []).append((bool(note), rel, note))

    found = []
    for name, decls in sorted(declarations.items()):
        if len(name) < 4 or name in SWIFT_NOISE or not all(d[0] for d in decls):
            continue
        if not cov._used(name, cov.integration):
            continue
        found.append((name, decls[0][1], decls[0][2]))
    return found


GROUP_ORDER = [
    "mx", "mx.fft", "mx.linalg", "mx.random", "mx.fast",
    "nn.layer", "nn.function", "nn.loss", "nn.init", "optim", "swift-only",
]


def build_rows(cov: Coverage, py_api: dict) -> list[dict]:
    rows = []
    for group, functions in py_api["modules"].items():
        for name, sig in functions.items():
            if name.startswith("_"):
                continue
            rows.append(cov.row(group, name, sig))
    for group in ["nn.layer", "nn.function", "nn.loss", "nn.init", "optim"]:
        for name, detail in py_api[group].items():
            prefer = "type" if name[:1].isupper() else "free"
            rows.append(cov.row(group, name, detail, prefer=prefer))
    rows += swift_only_rows(cov, rows)
    return rows


def write_report(rows: list[dict], out: pathlib.Path, mlx: pathlib.Path, deprecated=()):
    counts = collections.Counter((r["group"], r["state"]) for r in rows)
    totals = collections.Counter(r["group"] for r in rows)

    lines = []
    lines.append("# Integration test coverage report")
    lines.append("")
    lines.append(
        "Generated by `tools/audit_integration_coverage.py` -- do not edit by hand."
    )
    lines.append("")
    lines.append(f"- python `mlx` API read from: `{mlx}`")
    lines.append("- Swift API parsed from: `Source/`")
    lines.append(
        "- `integration` = covered by a case in `tools/integration_tests/cases.py` "
        "or `modules.py`, i.e. it has a python-vs-swift value check in "
        "`Tests/MLXIntegrationTests/Generated/`; `unit-only` = referenced by a hand "
        "written test "
        "file only; `none` = referenced by no test; `no-swift` = no Swift "
        "counterpart found (API gap, not a test gap)."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| area | total | integration | unit-only | none | no-swift | n/a |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for g in GROUP_ORDER:
        if not totals[g]:
            continue
        lines.append(
            f"| {g} | {totals[g]} | {counts[(g,'integration')]} | "
            f"{counts[(g,'unit-only')]} | {counts[(g,'none')]} | "
            f"{counts[(g,'no-swift')]} | {counts[(g,'n/a')]} |"
        )
    lines.append("")

    def table(group, states, title):
        selected = [r for r in rows if r["group"] == group and r["state"] in states]
        if not selected:
            return
        lines.append(f"### {title} ({len(selected)})")
        lines.append("")
        lines.append("| python | swift | state | file | signature |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in selected:
            sig = (r["swift_sig"] or r["detail"] or "").replace("|", "\\|")
            if len(sig) > 120:
                sig = sig[:117] + "..."
            lines.append(
                f"| `{r['py']}` | `{r['swift']}` | {r['state']} | "
                f"{r['file']} | `{sig}` |"
            )
        lines.append("")

    for g in GROUP_ORDER:
        if not totals[g]:
            continue
        lines.append(f"## {g}")
        lines.append("")
        table(g, {"none"}, f"{g}: no test coverage at all")
        table(g, {"unit-only"}, f"{g}: unit tests only (no python comparison)")
        table(g, {"no-swift"}, f"{g}: no Swift counterpart found")
        table(g, {"integration"}, f"{g}: covered by a generated case")

    if deprecated:
        lines.append("## Deprecated symbols used by the case tables")
        lines.append("")
        lines.append(
            "These tests were generated before a rename and should be regenerated "
            "with the current spelling."
        )
        lines.append("")
        lines.append("| symbol | declared in | deprecation |")
        lines.append("| --- | --- | --- |")
        for name, file, note in deprecated:
            lines.append(f"| `{name}` | {file} | `{note.replace('|', chr(92) + '|')}` |")
        lines.append("")

    out.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-python", default=None)
    parser.add_argument("--out", default=str(ROOT / "tools/integration-coverage-report.md"))
    parser.add_argument("--json", default=None, help="also dump raw rows as JSON")
    args = parser.parse_args()

    mlx = find_mlx_python(args.mlx_python)
    if mlx is None:
        sys.exit(
            "could not find the installed mlx python package; pass --mlx-python PATH"
        )

    cov = Coverage()
    rows = build_rows(cov, read_python_api(mlx))
    write_report(rows, pathlib.Path(args.out), mlx, deprecated_usage(cov))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1))

    counts = collections.Counter(r["state"] for r in rows)
    print(f"wrote {args.out}")
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
