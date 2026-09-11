# Copyright © 2026 Apple Inc.

"""
Check case expressions against the python `mlx` API *without running it*.

Generation evaluates the python side, so a typo or a stale keyword only shows up
after a long run -- and it aborts the file it is in.  This parses the installed
mlx sources/stubs instead and reports every problem at once, in a second.

It catches things like:

- `nn.HardTanh(min_val=-0.5)` -- python's `HardTanh` module takes no arguments
  (Swift's `HardTanh(min:max:)` does; use the free function for that)
- `nn.SinusoidalPositionalEncoding(8, cosine_first=True)` -- it is `cos_first`
- `mx.linspace(0, 10, num=6)` -- it is positional in the stub

Only expressions of the form `namespace.name(...)` are checked; methods,
operators and indexing are left alone.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys


def find_mlx_python() -> pathlib.Path | None:
    for entry in sys.path:
        candidate = pathlib.Path(entry) / "mlx" / "core" / "__init__.pyi"
        if candidate.exists():
            return candidate.parent.parent
    return None


def _split_arguments(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for character in text:
        if character in "([{":
            depth += 1
        if character in ")]}":
            depth -= 1
        if character == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += character
    if current.strip():
        parts.append(current)
    return [p for p in parts if p.strip()]


class PythonAPI:
    """parameter names for functions and classes in the installed mlx"""

    def __init__(self, mlx: pathlib.Path):
        self.functions: dict[str, list[str]] = {}
        self.variadic: set[str] = set()
        self.classes: dict[str, dict[str, list[str]]] = {}
        # stub types like `mx.array`, whose call signature is not worth modelling
        self.opaque: set[str] = set()

        # free functions come from the stubs: `def name(a, b, *, stream=None)`
        for module, prefix in [
            ("core/__init__.pyi", "mx"),
            ("core/fft.pyi", "mx.fft"),
            ("core/linalg.pyi", "mx.linalg"),
            ("core/random.pyi", "mx.random"),
            ("core/fast.pyi", "mx.fast"),
        ]:
            path = mlx / module
            if not path.exists():
                continue
            text = path.read_text()
            for match in re.finditer(r"^class (\w+)", text, re.M):
                self.opaque.add(f"{prefix}.{match.group(1)}")
            for match in re.finditer(r"^def (\w+)\(([^\n]*?)\)\s*->", text, re.M):
                names = []
                qualified = f"{prefix}.{match.group(1)}"
                for parameter in _split_arguments(match.group(2)):
                    parameter = parameter.strip()
                    if parameter.startswith("*") and parameter not in ("*",):
                        self.variadic.add(qualified)
                    if parameter in ("/", "*") or parameter.startswith("*"):
                        continue
                    names.append(parameter.split(":")[0].split("=")[0].strip())
                self.functions.setdefault(qualified, names)

        # nn functions and layers come from the sources
        for path in sorted((mlx / "optimizers").rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    self.functions.setdefault(
                        f"optim.{node.name}",
                        [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs],
                    )
                if isinstance(node, ast.ClassDef):
                    methods: dict[str, list[str]] = {}
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            methods[item.name] = [a.arg for a in item.args.args][1:] + [
                                a.arg for a in item.args.kwonlyargs
                            ]
                    # optimizer subclasses inherit __init__ when they do not define one
                    existing = self.classes.get(node.name, {})
                    self.classes[node.name] = methods or existing

        for path in sorted((mlx / "nn").rglob("*.py")):
            tree = ast.parse(path.read_text())
            prefix = "nn.losses" if path.name == "losses.py" else "nn"
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    names = [a.arg for a in node.args.args] + [
                        a.arg for a in node.args.kwonlyargs
                    ]
                    self.functions.setdefault(f"{prefix}.{node.name}", names)
                    if prefix == "nn.losses":
                        # they are re-exported as nn.losses.* only
                        continue
                if isinstance(node, ast.ClassDef):
                    methods: dict[str, list[str]] = {}
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name in (
                            "__init__",
                            "__call__",
                        ):
                            methods[item.name] = [a.arg for a in item.args.args][1:] + [
                                a.arg for a in item.args.kwonlyargs
                            ]
                    self.classes.setdefault(node.name, methods)


def _check_call(api: PythonAPI, label: str, expression: str) -> list[str]:
    match = re.match(
        r"^((?:mx|nn|optim)(?:\.\w+)*)\.(\w+)\((.*)\)$", expression.strip(), re.S
    )
    if not match:
        return []
    namespace, name, arguments = match.groups()
    qualified = f"{namespace}.{name}"

    if qualified in api.opaque:
        return []

    # a static/class method: `nn.MultiHeadAttention.create_additive_causal_mask(...)`
    owner = namespace.split(".")[-1]
    if owner[:1].isupper():
        methods = api.classes.get(owner)
        if methods is None:
            return [f"{label}: python has no class {owner}"]
        if name not in methods:
            # only __init__/__call__ are modelled; anything else is not checked
            return []
        return []

    if name[:1].isupper():
        methods = api.classes.get(name)
        if methods is None:
            return [f"{label}: python has no class {name}"]
        parameters = methods.get("__init__", [])
        described = f"{name}.__init__"
    else:
        parameters = api.functions.get(qualified)
        if parameters is None:
            return [f"{label}: python has no function {qualified}"]
        described = qualified

    problems = []
    positional = 0
    for argument in _split_arguments(arguments):
        keyword = re.match(r"\s*(\w+)\s*=[^=]", argument)
        if keyword:
            if keyword.group(1) not in parameters:
                problems.append(
                    f"{label}: {described} has no parameter '{keyword.group(1)}' "
                    f"(has {parameters or 'none'})"
                )
        else:
            positional += 1
    if positional > len(parameters) and described not in api.variadic:
        problems.append(
            f"{label}: {described} takes {len(parameters)} argument(s), "
            f"the case passes {positional} positionally"
        )
    return problems


def check(
    cases,
    module_cases,
    optimizer_cases=(),
    schedule_cases=(),
    mlx: pathlib.Path | None = None,
) -> list[str]:
    mlx = mlx or find_mlx_python()
    if mlx is None:
        return []
    api = PythonAPI(mlx)

    problems: list[str] = []
    for case in cases:
        label = f"{case.file}/{case.name}"
        if case.py:
            problems += _check_call(api, label, case.py)
        for name, spec in case.inputs.items():
            problems += _check_call(api, f"{label} [{name}]", spec.python())

    for case in module_cases:
        label = f"{case.file}/{case.name}"
        problems += _check_call(api, label, case.py)
        for name, spec in case.inputs.items():
            problems += _check_call(api, f"{label} [{name}]", spec.python())

        # the call: `module(x, offset=2)` against the class' __call__
        match = re.match(r"^nn\.(\w+)\(", case.py)
        if match:
            methods = api.classes.get(match.group(1), {})
            parameters = methods.get("__call__")
            if parameters is not None:
                for argument in _split_arguments(case.call_py[case.call_py.index("(") + 1 : -1]):
                    keyword = re.match(r"\s*(\w+)\s*=[^=]", argument)
                    if keyword and keyword.group(1) not in parameters:
                        problems.append(
                            f"{label}: {match.group(1)}.__call__ has no parameter "
                            f"'{keyword.group(1)}' (has {parameters or 'none'})"
                        )

    for case in optimizer_cases:
        label = f"{case.file}/{case.name}"
        expression = case.py.replace("$SCHEDULE", "schedule")
        problems += _check_call(api, label, expression)
        for name, spec in case.parameters.items():
            problems += _check_call(api, f"{label} [{name}]", spec.python())
        if case.schedule:
            problems += _check_call(api, f"{label} [schedule]", case.schedule[0])

    for case in schedule_cases:
        problems += _check_call(api, f"{case.file}/{case.name}", case.py)

    return problems
