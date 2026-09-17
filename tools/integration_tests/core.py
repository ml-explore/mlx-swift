# Copyright © 2026 Apple Inc.

"""
Core of the integration test generator.

A *case* is a single value comparison between python `mlx` and Swift `MLX`:

    Case(
        "exp",
        file="Elementwise",
        inputs={"a": Normal([4, 3])},
        py="mx.exp(a)",
        swift="exp(a)",
    )

The generator:

- picks a stable seed from the case name (adding cases does not renumber others)
- seeds the *global* python random state, and emits
  `withRandomState(MLXRandom.RandomState(seed:))` on the Swift side so both sides
  draw the same values in the same order
- declares each input in both languages from one spec (see `InputSpec`)
- evaluates the python expression, summarizes the result, and emits the summary
  as a swift-testing `@Test` that recomputes the same summary in Swift

Only cases whose python expression produces a *single* array are supported.
Multiple return values and `Module`/optimizer cases are separate generators.

See `tools/integration_tests/README.md`.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import re
import typing as t
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# bump when the emitted code (not the case list) changes in a way that changes values
GENERATOR_REVISION = 2

# number of evenly spaced elements sampled from each array
SAMPLE_COUNT = 6


# --------------------------------------------------------------------- dtypes

# python dtype name -> (swift DType case, swift concrete type)
DTYPES = {
    "bool_": (".bool", "Bool"),
    "uint8": (".uint8", "UInt8"),
    "uint16": (".uint16", "UInt16"),
    "uint32": (".uint32", "UInt32"),
    "uint64": (".uint64", "UInt64"),
    "int8": (".int8", "Int8"),
    "int16": (".int16", "Int16"),
    "int32": (".int32", "Int32"),
    "int64": (".int64", "Int"),
    "float16": (".float16", "Float16"),
    "float32": (".float32", "Float"),
    "bfloat16": (".bfloat16", "BFloat16"),
    "float64": (".float64", "Double"),
}

EXACT_DTYPES = {"bool_", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"}
LOOSE_DTYPES = {"float16", "bfloat16"}


def dtype_name(dtype) -> str:
    """`mx.float32` -> `float32`"""
    name = str(dtype).split(".")[-1]
    return "bool_" if name in ("bool", "bool_") else name


def swift_dtype(name: str) -> str:
    if name not in DTYPES:
        raise ValueError(f"unsupported dtype: {name}")
    return DTYPES[name][0]


# ---------------------------------------------------------------- input specs


class InputSpec:
    """A value that can be declared identically in python and Swift."""

    def python(self) -> str:
        raise NotImplementedError

    def swift(self) -> str:
        raise NotImplementedError

    @property
    def is_array(self) -> bool:
        return True


def _py_shape(shape: t.Sequence[int]) -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]"


def _shape(shape: t.Sequence[int]) -> str:
    """a Swift shape literal

    `[Int]()` rather than `[]` for a scalar: an empty array literal cannot infer
    its element type in `MLXRandom.normal([])`.
    """
    if len(shape) == 0:
        return "[Int]()"
    return "[" + ", ".join(str(d) for d in shape) + "]"


def _float(value: float) -> str:
    """a float literal that is valid in both languages"""
    if value != value:
        raise ValueError("use Expression() for nan literals")
    if math.isinf(value):
        raise ValueError("use Expression() for infinite literals")
    text = repr(float(value))
    return text


@dataclasses.dataclass
class Normal(InputSpec):
    """`mx.random.normal` / `MLXRandom.normal`"""

    shape: t.Sequence[int] = (4, 3)
    loc: float = 0.0
    scale: float = 1.0
    dtype: str = "float32"

    def python(self) -> str:
        return (
            f"mx.random.normal({_py_shape(self.shape)}, dtype=mx.{self.dtype}, "
            f"loc={_float(self.loc)}, scale={_float(self.scale)})"
        )

    def swift(self) -> str:
        return (
            f"MLXRandom.normal({_shape(self.shape)}, dtype: {swift_dtype(self.dtype)}, "
            f"loc: {_float(self.loc)}, scale: {_float(self.scale)})"
        )


@dataclasses.dataclass
class Uniform(InputSpec):
    """`mx.random.uniform` / `MLXRandom.uniform`"""

    shape: t.Sequence[int] = (4, 3)
    low: float = 0.0
    high: float = 1.0
    dtype: str = "float32"

    def python(self) -> str:
        return (
            f"mx.random.uniform({_float(self.low)}, {_float(self.high)}, "
            f"{_py_shape(self.shape)}, dtype=mx.{self.dtype})"
        )

    def swift(self) -> str:
        return (
            f"MLXRandom.uniform(low: {_float(self.low)}, high: {_float(self.high)}, "
            f"{_shape(self.shape)}, dtype: {swift_dtype(self.dtype)})"
        )


@dataclasses.dataclass
class RandInt(InputSpec):
    """`mx.random.randint` / `MLXRandom.randInt`"""

    low: int
    high: int
    shape: t.Sequence[int] = (4, 3)
    dtype: str = "int32"

    def python(self) -> str:
        return (
            f"mx.random.randint({self.low}, {self.high}, {_py_shape(self.shape)}, "
            f"dtype=mx.{self.dtype})"
        )

    def swift(self) -> str:
        return (
            f"MLXRandom.randInt(low: {self.low}, high: {self.high}, "
            f"{_shape(self.shape)}, type: {DTYPES[self.dtype][1]}.self)"
        )


@dataclasses.dataclass
class Bernoulli(InputSpec):
    """`mx.random.bernoulli` / `MLXRandom.bernoulli` -- a `bool` array"""

    shape: t.Sequence[int] = (4, 3)
    p: float = 0.5

    def python(self) -> str:
        return f"mx.random.bernoulli({_float(self.p)}, {_py_shape(self.shape)})"

    def swift(self) -> str:
        return f"MLXRandom.bernoulli({_float(self.p)}, {_shape(self.shape)})"


@dataclasses.dataclass
class Scalar(InputSpec):
    """a plain number, declared as a `let` in Swift so the call site reads the same"""

    value: float | int

    def python(self) -> str:
        return repr(self.value)

    def swift(self) -> str:
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        if isinstance(self.value, int):
            return str(self.value)
        return f"Float({_float(self.value)})"

    @property
    def is_array(self) -> bool:
        return False


@dataclasses.dataclass
class Expression(InputSpec):
    """escape hatch: an explicit pair of expressions

    Use for anything the specs above cannot describe -- literal arrays, inf/nan
    inputs, derived values, well conditioned matrices, ...

        Expression(
            py="mx.array([1.0, mx.inf, -mx.inf, mx.nan, 0.0])",
            swift="MLXArray([1.0, .infinity, -.infinity, .nan, 0.0])",
        )
    """

    py: str
    swift_: str = ""
    array: bool = True

    def __init__(self, py: str, swift: str, *, array: bool = True):
        self.py = py
        self.swift_ = swift
        self.array = array

    def python(self) -> str:
        return self.py

    def swift(self) -> str:
        return self.swift_

    @property
    def is_array(self) -> bool:
        return self.array


# ---------------------------------------------------------------------- cases


@dataclasses.dataclass
class Case:
    """One python-vs-Swift value comparison.

    - `name`: display name, also the basis for the seed and the Swift function
      name.  Use `/` to group variants, e.g. `sum/axes`.
    - `file`: output file key, e.g. `Elementwise` -> `ElementwiseTests.swift`
    - `inputs`: declared in both languages, *in order*
    - `py`: python expression evaluated with the inputs in scope
    - `swift`: Swift expression using the same names
    - `tolerance`: `None` (derive from the result dtype), or one of
      `exact`, `float32`, `float16`, `loose`
    - `verify_inputs`: also assert the inputs match.  Off by default: random
      number generation is covered once per (spec, dtype) in the `RandomInputs`
      file rather than re-verified in every case.

    An *inputs-only* case leaves `py`/`swift` empty: it declares inputs, verifies
    them, and emits no result.  That is how `RandomInputs` covers the RNG.
    """

    name: str
    file: str
    inputs: dict[str, InputSpec]
    py: str = ""
    swift: str = ""
    tolerance: str | None = None
    verify_inputs: bool = False
    note: str | None = None

    @property
    def inputs_only(self) -> bool:
        return not self.py

    @property
    def seed(self) -> int:
        # stable per case: inserting a case does not change any other seed
        return zlib.crc32(f"{self.file}/{self.name}".encode()) % 100_000

    @property
    def function_name(self) -> str:
        name = re.sub(r"[^A-Za-z0-9]+", "_", self.name).strip("_")
        return f"test_{name}"


def coverage_key(spec: InputSpec) -> tuple[str, str] | None:
    """the (generator, dtype) pair a spec needs RNG coverage for

    `None` for deterministic specs (`Expression`, `Scalar`) -- they carry their
    own values and do not depend on the random state.
    """
    if isinstance(spec, (Normal, Uniform)):
        return (type(spec).__name__, spec.dtype)
    if isinstance(spec, RandInt):
        return ("RandInt", spec.dtype)
    if isinstance(spec, Bernoulli):
        return ("Bernoulli", "bool_")
    return None


@dataclasses.dataclass
class ModuleCase:
    """One python-vs-Swift comparison of an `MLXNN.Module`.

    - `name`: display name, also the basis for the seed and the Swift function name
    - `file`: output file key, e.g. `ModuleLinear` -> `GeneratedModuleLinearTests.swift`
    - `py` / `swift`: expressions constructing the module
    - `inputs`: declared in both languages, *in order*, before the module is built
    - `call_py` / `call_swift`: how the module is called; the default is `module(x)`
      so most cases just name their input `x`
    - `training`: `train(true)` instead of `train(false)`.  Only for layers whose
      forward pass is deterministic in training mode -- anything that draws random
      numbers (Dropout) cannot be compared this way, because python and Swift
      consume a different number of keys while *constructing* the module.

    Every parameter of the module is replaced with a deterministic function of
    its own shape (see `PARAMETER_PY` / `deterministicParameter` in
    `IntegrationSupport.swift`), so the comparison does not depend on the two
    implementations drawing the same random initialization -- which is exactly
    what the old generator relied on.
    """

    name: str
    file: str
    py: str
    swift: str
    inputs: dict[str, InputSpec] = dataclasses.field(default_factory=dict)
    call_py: str = "module(x)"
    call_swift: str = "module(x)"
    training: bool = False
    tolerance: str | None = None
    note: str | None = None

    @property
    def seed(self) -> int:
        return zlib.crc32(f"{self.file}/{self.name}".encode()) % 100_000

    @property
    def function_name(self) -> str:
        name = re.sub(r"[^A-Za-z0-9]+", "_", self.name).strip("_")
        return f"test_{name}"


@dataclasses.dataclass
class OptimizerCase:
    """One python-vs-Swift comparison of an optimizer, over several steps.

    The gradient is the derivative of `sum((parameter - target) ** 2)`, i.e.
    `2 * (parameter - target)`, recomputed from the *current* parameters at every
    step.  That means the trajectory depends on the optimizer's state and step
    count, which is where single-step comparisons (what the retired generator did)
    are blind: Adam's bias correction, Adafactor's step factor and every momentum
    term only show up from the second step onwards.

    - `parameters`: flat names -> initial values.  Two parameters with different
      ranks are worth having: Adafactor factors 2-D parameters and not 1-D ones.
    - `steps`: how many updates to apply
    - `schedule`: optional `(python, swift)` learning rate schedule.  python takes
      it as `learning_rate=`, Swift assigns `optimizer.learningRate` each step --
      comparing the two pins down the step alignment.
    """

    name: str
    file: str
    py: str
    swift: str
    parameters: dict[str, InputSpec]
    steps: int = 3
    schedule: tuple[str, str] | None = None
    tolerance: str | None = None
    note: str | None = None

    def __post_init__(self):
        for key in self.parameters:
            if "." in key:
                raise ValueError(f"{self.file}/{self.name}: nested parameters are not supported")

    @property
    def seed(self) -> int:
        return zlib.crc32(f"{self.file}/{self.name}".encode()) % 100_000

    @property
    def function_name(self) -> str:
        name = re.sub(r"[^A-Za-z0-9]+", "_", self.name).strip("_")
        return f"test_{name}"


@dataclasses.dataclass
class ScheduleCase:
    """One python-vs-Swift comparison of a learning rate schedule.

    The schedule is sampled at `0 ..< steps` and the values are compared as an
    array, so the whole curve is checked rather than one point.
    """

    name: str
    file: str
    py: str
    swift: str
    steps: int = 12
    tolerance: str | None = None
    note: str | None = None

    @property
    def seed(self) -> int:
        return zlib.crc32(f"{self.file}/{self.name}".encode()) % 100_000

    @property
    def function_name(self) -> str:
        name = re.sub(r"[^A-Za-z0-9]+", "_", self.name).strip("_")
        return f"test_{name}"


# how a parameter is replaced, in python; `deterministicParameter` in
# `IntegrationSupport.swift` must match exactly
PARAMETER_PY = (
    "(mx.arange(v.size, dtype=mx.float32).reshape(v.shape) / v.size - 0.5).astype(v.dtype)"
)


# --------------------------------------------------------------- summarizing


@dataclasses.dataclass
class Summary:
    shape: list[int]
    dtype: str
    mean: float
    minimum: float
    maximum: float
    absolute_sum: float
    position_checksum: float
    sample_indices: list[int]
    samples: list[float]


def sample_indices(count: int, wanted: int = SAMPLE_COUNT) -> list[int]:
    if count <= wanted:
        return list(range(count))
    step = (count - 1) / (wanted - 1)
    return sorted({int(round(i * step)) for i in range(wanted)})


def summarize(mx, array) -> Summary:
    """Compute the summary that the Swift side recomputes in `expectSummary`.

    Definitions must match `Tests/MLXIntegrationTests/IntegrationSupport.swift`
    exactly:

        values = array.astype(float32).reshape(-1)
        absolute_sum = sum(|values|)
        position_checksum = sum(|values| * arange(1, n + 1)) / n
        samples = values[evenly spaced indices]
    """
    name = dtype_name(array.dtype)
    if name == "complex64":
        raise ValueError("complex results are not supported yet")

    values = array.astype(mx.float32).reshape(-1)
    count = values.size
    if count == 0:
        raise ValueError("empty results are not supported yet")

    absolute = mx.abs(values)
    weights = mx.arange(1, count + 1, dtype=mx.float32)
    indices = sample_indices(count)

    return Summary(
        shape=list(array.shape),
        dtype=name,
        mean=values.mean().item(),
        minimum=values.min().item(),
        maximum=values.max().item(),
        absolute_sum=absolute.sum().item(),
        position_checksum=(absolute * weights).sum().item() / count,
        sample_indices=indices,
        samples=[values[i].item() for i in indices],
    )


def tolerance_for(summary: Summary, override: str | None) -> str:
    if override:
        return override
    if summary.dtype in EXACT_DTYPES:
        return "exact"
    if summary.dtype in LOOSE_DTYPES:
        return "float16"
    return "float32"


# ------------------------------------------------------------------ emitting


def swift_double(value: float) -> str:
    if value != value:
        return "Double.nan"
    if value == math.inf:
        return "Double.infinity"
    if value == -math.inf:
        return "-Double.infinity"
    return repr(float(value))


def emit_summary(indent: str, name: str, summary: Summary, tolerance: str) -> str:
    samples = ", ".join(swift_double(v) for v in summary.samples)
    indices = ", ".join(str(i) for i in summary.sample_indices)
    return (
        f"{indent}expectSummary(\n"
        f"{indent}    {name},\n"
        f"{indent}    ArraySummary(\n"
        f"{indent}        shape: {_shape(summary.shape)},\n"
        f"{indent}        dtype: {swift_dtype(summary.dtype)},\n"
        f"{indent}        mean: {swift_double(summary.mean)},\n"
        f"{indent}        minimum: {swift_double(summary.minimum)},\n"
        f"{indent}        maximum: {swift_double(summary.maximum)},\n"
        f"{indent}        absoluteSum: {swift_double(summary.absolute_sum)},\n"
        f"{indent}        positionChecksum: {swift_double(summary.position_checksum)},\n"
        f"{indent}        sampleIndices: [{indices}],\n"
        f"{indent}        samples: [{samples}]),\n"
        f"{indent}    tolerance: .{tolerance})\n"
    )


PLACEHOLDER = Summary(
    shape=[1], dtype="float32", mean=0.0, minimum=0.0, maximum=0.0, absolute_sum=0.0,
    position_checksum=0.0, sample_indices=[0], samples=[0.0],
)


def emit_case(mx, case: Case, evaluate: bool = True) -> str:
    """evaluate the python side of a case and emit the Swift test

    With `evaluate=False` no python is run and placeholder values are emitted:
    that is only useful for checking that the generated Swift parses (see
    `generate.py --syntax-only`).
    """
    if not evaluate:
        indent = " " * 12
        body = ""
        for name, spec in case.inputs.items():
            body += f"{indent}let {name} = {spec.swift()}\n"
        if not case.inputs_only:
            body += f"{indent}let result = {case.swift}\n"
            body += emit_summary(indent, "result", PLACEHOLDER, "loose")
        return (
            f'    @Test("{case.name}")\n'
            f"    func {case.function_name}() throws {{\n"
            f"        try withIntegrationState(seed: {case.seed}) {{\n"
            f"{body}"
            f"        }}\n"
            f"    }}\n"
        )

    namespace: dict[str, t.Any] = {"mx": mx}
    if not getattr(mx, "is_fake", False):
        # `nn` and `optim` are available to any case that names them; importing
        # mlx alongside the numpy stand-in used by --self-test aborts the process
        try:
            import mlx.nn as nn  # type: ignore
            import mlx.optimizers as optim  # type: ignore

            namespace["nn"] = nn
            namespace["optim"] = optim
        except Exception:
            pass

    mx.random.seed(case.seed)

    body = ""
    indent = " " * 12
    verify_inputs = case.verify_inputs or case.inputs_only
    for name, spec in case.inputs.items():
        value = eval(spec.python(), namespace)  # noqa: S307 -- generator input
        namespace[name] = value
        body += f"{indent}let {name} = {spec.swift()}\n"
        if not verify_inputs or not spec.is_array:
            continue
        if dtype_name(value.dtype) == "complex64":
            # complex inputs are built from (already verified) real parts
            continue
        summary = summarize(mx, value)
        body += emit_summary(indent, name, summary, tolerance_for(summary, None))

    if not case.inputs_only:
        result = eval(case.py, namespace)  # noqa: S307 -- generator input
        if isinstance(result, (tuple, list)):
            raise ValueError(
                f"{case.file}/{case.name}: expression returns {type(result).__name__}; "
                "this generator only handles single values"
            )

        body += f"{indent}let result = {case.swift}\n"

        if dtype_name(result.dtype) == "complex64":
            # a complex result is verified as two real arrays
            parts = [
                ("result.realPart()", result.real),
                ("result.imaginaryPart()", result.imag),
            ]
        else:
            parts = [("result", result)]

        for expression, value in parts:
            summary = summarize(mx, value)
            body += emit_summary(
                indent, expression, summary, tolerance_for(summary, case.tolerance)
            )

    note = f"        // {case.note}\n" if case.note else ""
    return (
        f'    @Test("{case.name}")\n'
        f"    func {case.function_name}() throws {{\n"
        f"{note}"
        f"        try withIntegrationState(seed: {case.seed}) {{\n"
        f"{body}"
        f"        }}\n"
        f"    }}\n"
    )


def emit_module_case(mx, case: ModuleCase, evaluate: bool = True) -> str:
    """evaluate the python side of a module case and emit the Swift test"""
    indent = " " * 12

    if not evaluate:
        body = "".join(
            f"{indent}let {name} = {spec.swift()}\n" for name, spec in case.inputs.items()
        )
        body += f"{indent}let module = {case.swift}\n"
        body += f"{indent}module.update(parameters: module.mapParameters {{ deterministicParameter($0) }})\n"
        body += f"{indent}module.train({str(case.training).lower()})\n"
        body += f"{indent}expectParameters(module, [])\n"
        body += f"{indent}let result = {case.call_swift}\n"
        body += emit_summary(indent, "result", PLACEHOLDER, "loose")
        return (
            f'    @Test("{case.name}")\n'
            f"    func {case.function_name}() throws {{\n"
            f"        try withIntegrationState(seed: {case.seed}) {{\n"
            f"{body}"
            f"        }}\n"
            f"    }}\n"
        )

    if getattr(mx, "is_fake", False):
        # importing mlx alongside the numpy stand-in aborts the process
        raise ValueError("module cases need the real mlx (not --self-test)")

    import mlx.nn as nn  # type: ignore
    import mlx.optimizers as optim  # type: ignore
    from mlx.utils import tree_flatten, tree_unflatten  # type: ignore

    namespace: dict[str, t.Any] = {"mx": mx, "nn": nn, "optim": optim}
    mx.random.seed(case.seed)

    body = ""
    for name, spec in case.inputs.items():
        namespace[name] = eval(spec.python(), namespace)  # noqa: S307 -- generator input
        body += f"{indent}let {name} = {spec.swift()}\n"

    module = eval(case.py, namespace)  # noqa: S307 -- generator input
    namespace["module"] = module

    # replace every parameter with a deterministic function of its shape
    module.update(
        tree_unflatten(
            [
                (key, eval(PARAMETER_PY, {"mx": mx, "v": value}))  # noqa: S307
                for key, value in tree_flatten(module.parameters())
            ]
        )
    )
    module.train(case.training)

    parameters = sorted(tree_flatten(module.parameters()))
    described = ", ".join(
        f'("{key}", {_shape(list(value.shape))})' for key, value in parameters
    )

    body += f"{indent}let module = {case.swift}\n"
    body += (
        f"{indent}module.update(parameters: module.mapParameters "
        f"{{ deterministicParameter($0) }})\n"
    )
    body += f"{indent}module.train({str(case.training).lower()})\n"
    body += f"{indent}expectParameters(module, [{described}])\n"

    result = eval(case.call_py, namespace)  # noqa: S307 -- generator input
    if isinstance(result, (tuple, list)):
        raise ValueError(
            f"{case.file}/{case.name}: the call returns {type(result).__name__}; "
            "this generator only handles modules with a single output"
        )

    summary = summarize(mx, result)
    body += f"{indent}let result = {case.call_swift}\n"
    body += emit_summary(indent, "result", summary, tolerance_for(summary, case.tolerance))

    note = f"        // {case.note}\n" if case.note else ""
    return (
        f'    @Test("{case.name}")\n'
        f"    func {case.function_name}() throws {{\n"
        f"{note}"
        f"        try withIntegrationState(seed: {case.seed}) {{\n"
        f"{body}"
        f"        }}\n"
        f"    }}\n"
    )


def emit_optimizer_case(mx, case: OptimizerCase, evaluate: bool = True) -> str:
    """run the optimizer for `steps` steps in python and emit the Swift test"""
    indent = " " * 12
    names = list(case.parameters)

    def swift_body(summaries: dict[str, Summary] | None) -> str:
        body = ""
        for name, spec in case.parameters.items():
            body += f"{indent}let {name} = {spec.swift()}\n"
        for name, spec in case.parameters.items():
            body += f"{indent}let {name}Target = {spec.swift()}\n"

        if case.schedule:
            body += f"{indent}let schedule = {case.schedule[1]}\n"
            body += f"{indent}let optimizer = {case.swift.replace('$SCHEDULE', 'schedule(0)')}\n"
        else:
            body += f"{indent}let optimizer = {case.swift}\n"

        pairs = ", ".join(f'("{name}", {name})' for name in names)
        body += f"{indent}var parameters = ModuleParameters.unflattened([{pairs}])\n"
        if case.schedule:
            body += f"{indent}for step in 0 ..< {case.steps} {{\n"
            body += f"{indent}    optimizer.learningRate = schedule(step)\n"
        else:
            body += f"{indent}for _ in 0 ..< {case.steps} {{\n"
        gradients = ", ".join(
            f'("{name}", 2 * (parameters[unwrapping: "{name}"]! - {name}Target))'
            for name in names
        )
        body += f"{indent}    let gradients = ModuleParameters.unflattened([{gradients}])\n"
        body += (
            f"{indent}    parameters = optimizer.apply("
            f"gradients: gradients, modelParameters: parameters)\n"
        )
        body += f"{indent}}}\n"

        for name in names:
            summary = summaries[name] if summaries else PLACEHOLDER
            tolerance = (
                tolerance_for(summary, case.tolerance) if summaries else "loose"
            )
            body += emit_summary(
                indent, f'parameters[unwrapping: "{name}"]!', summary, tolerance
            )
        return body

    if not evaluate:
        body = swift_body(None)
    else:
        import mlx.nn as nn  # type: ignore
        import mlx.optimizers as optim  # type: ignore

        namespace: dict[str, t.Any] = {"mx": mx, "nn": nn, "optim": optim}
        mx.random.seed(case.seed)

        values = {name: eval(spec.python(), namespace) for name, spec in case.parameters.items()}  # noqa: S307
        targets = {name: eval(spec.python(), namespace) for name, spec in case.parameters.items()}  # noqa: S307

        if case.schedule:
            namespace["schedule"] = eval(case.schedule[0], namespace)  # noqa: S307
            optimizer = eval(case.py.replace("$SCHEDULE", "schedule"), namespace)  # noqa: S307
        else:
            optimizer = eval(case.py, namespace)  # noqa: S307

        parameters = dict(values)
        for _ in range(case.steps):
            gradients = {name: 2 * (parameters[name] - targets[name]) for name in names}
            parameters = optimizer.apply_gradients(gradients, parameters)

        body = swift_body({name: summarize(mx, parameters[name]) for name in names})

    note = f"        // {case.note}\n" if case.note else ""
    return (
        f'    @Test("{case.name}")\n'
        f"    func {case.function_name}() throws {{\n"
        f"{note}"
        f"        try withIntegrationState(seed: {case.seed}) {{\n"
        f"{body}"
        f"        }}\n"
        f"    }}\n"
    )


def emit_schedule_case(mx, case: ScheduleCase, evaluate: bool = True) -> str:
    """sample a schedule at 0 ..< steps in python and emit the Swift test"""
    indent = " " * 12

    body = f"{indent}let schedule = {case.swift}\n"
    body += f"{indent}let values = MLXArray((0 ..< {case.steps}).map {{ schedule($0) }})\n"

    if evaluate:
        import mlx.optimizers as optim  # type: ignore

        schedule = eval(case.py, {"mx": mx, "optim": optim})  # noqa: S307 -- generator input
        values = mx.array([float(schedule(step)) for step in range(case.steps)])
        summary = summarize(mx, values)
        tolerance = tolerance_for(summary, case.tolerance)
    else:
        summary, tolerance = PLACEHOLDER, "loose"

    body += emit_summary(indent, "values", summary, tolerance)

    note = f"        // {case.note}\n" if case.note else ""
    return (
        f'    @Test("{case.name}")\n'
        f"    func {case.function_name}() throws {{\n"
        f"{note}"
        f"        try withIntegrationState(seed: {case.seed}) {{\n"
        f"{body}"
        f"        }}\n"
        f"    }}\n"
    )


def matmul_precision() -> str:
    """what `MLX_ENABLE_TF32` was set to when the values were produced

    float32 matmuls run in TF32 on the neural accelerators of M5 and later unless
    this is 0, and TF32 results differ from float32 by ~1e-4 relative -- enough to
    make generated values hardware specific.
    """
    return os.environ.get("MLX_ENABLE_TF32", "1 (default)")


def device_description(mx) -> str:
    try:
        info = mx.metal.device_info()
        return f"{info.get('device_name', '?')} ({info.get('architecture', '?')})"
    except Exception:  # noqa: BLE001 -- provenance only
        return "unknown"


def vendored_mlx_version() -> str:
    header = ROOT / "Source/Cmlx/mlx/mlx/version.h"
    if not header.exists():
        return "unknown"
    text = header.read_text()
    parts = [
        re.search(rf"#define MLX_VERSION_{part}\s+(\d+)", text) for part in ("MAJOR", "MINOR", "PATCH")
    ]
    if not all(parts):
        return "unknown"
    return ".".join(m.group(1) for m in parts)  # type: ignore[union-attr]


def emit_file(
    mx,
    file: str,
    cases: list[Case | ModuleCase | OptimizerCase | ScheduleCase],
    skip_errors: bool = False,
    extra_imports: t.Sequence[str] = (),
    evaluate: bool = True,
) -> tuple[str, list[str]]:
    """returns (swift source, skipped case descriptions)"""
    python_version = getattr(mx, "__version__", "unknown") if evaluate else "not evaluated"
    suite = f"Generated{file}Tests"

    imports = ["Foundation", "MLX", *extra_imports, "Testing"]
    header = (
        "// Copyright © 2026 Apple Inc.\n"
        "//\n"
        "// GENERATED by tools/integration_tests -- DO NOT EDIT.\n"
        "//\n"
        "// Values in this file were produced by python mlx and are compared against\n"
        "// the Swift API.  Regenerate with:\n"
        "//\n"
        "//     python3 tools/integration_tests/generate.py\n"
        "//\n"
        f"// python mlx:              {python_version}\n"
        f"// vendored mlx (Cmlx):     {vendored_mlx_version()}\n"
        f"// generator revision:      {GENERATOR_REVISION}\n"
        f"// MLX_ENABLE_TF32:         {matmul_precision() if evaluate else 'n/a'}\n"
        f"// device:                  {device_description(mx) if evaluate else 'n/a'}\n"
        f"// cases:                   {len(cases)}\n"
        "\n"
        + "".join(
            f"{name}\n" if name.startswith("@testable") else f"import {name}\n"
            for name in imports
        )
        + "\n"
        f'@Suite("generated: {file}")\n'
        f"struct {suite} {{\n"
        "\n"
    )

    seen: set[str] = set()
    bodies = []
    skipped: list[str] = []
    for case in cases:
        if case.function_name in seen:
            raise ValueError(f"duplicate case name: {case.file}/{case.name}")
        seen.add(case.function_name)
        try:
            if isinstance(case, ModuleCase):
                bodies.append(emit_module_case(mx, case, evaluate=evaluate))
            elif isinstance(case, OptimizerCase):
                bodies.append(emit_optimizer_case(mx, case, evaluate=evaluate))
            elif isinstance(case, ScheduleCase):
                bodies.append(emit_schedule_case(mx, case, evaluate=evaluate))
            else:
                bodies.append(emit_case(mx, case, evaluate=evaluate))
        except Exception as e:  # noqa: BLE001
            if not skip_errors:
                raise ValueError(f"{case.file}/{case.name}: {type(e).__name__}: {e}") from e
            skipped.append(f"{case.file}/{case.name}: {type(e).__name__}: {e}")

    header = header.replace(
        f"// cases:                   {len(cases)}",
        f"// cases:                   {len(bodies)}",
    )

    return header + "\n".join(bodies) + "}\n", skipped
