# Copyright © 2026 Apple Inc.

"""
Case tables for the integration test generator.

Each `Case` is one python-vs-Swift value comparison; `file` selects the output
file.  Add cases here -- the generator needs no changes for a new op unless the
op needs an input the specs in `core.py` cannot describe (use `Expression` for
those).

Conventions:

- Swift free functions are written `MLX.foo(...)` so the generated code cannot
  accidentally resolve to a stdlib/Foundation overload (`abs`, `min`, `pow`, ...)
- keep the python and Swift expressions on one line each; they are the
  documentation of the mapping
- prefer inputs whose values exercise the op (e.g. positive inputs for `log`)
"""

from core import Bernoulli, Case, Expression, Normal, RandInt, Uniform

# shapes used throughout
S = (4, 3)  # default 2D
S4 = (2, 3, 4, 3)  # for axes: variants

# an input with inf/nan/zero, for the predicates
SPECIAL = Expression(
    py="mx.array([1.0, -2.5, mx.inf, -mx.inf, mx.nan, 0.0])",
    swift="MLXArray([1.0, -2.5, Float.infinity, -Float.infinity, Float.nan, 0.0])",
)

CASES: list[Case] = []


def case(name: str, file: str, inputs: dict, py: str, swift: str, **kwargs) -> None:
    CASES.append(Case(name=name, file=file, inputs=inputs, py=py, swift=swift, **kwargs))


def inputs_case(name: str, inputs: dict, **kwargs) -> None:
    """a case that only declares and verifies inputs -- see RandomInputs below"""
    CASES.append(Case(name=name, file="RandomInputs", inputs=inputs, **kwargs))


# ------------------------------------------------------------- random inputs
#
# Random number generation is verified *here*, once per (generator, dtype) and
# for the draw order within a state, instead of re-verifying the inputs of every
# op case.  Op cases only assert their result.
#
# `generate.py` warns if an op case uses a generator/dtype combination that has
# no coverage in this file.

inputs_case("normal/2d", {"a": Normal(S)})
inputs_case("normal/2d/seed2", {"a": Normal(S)})
inputs_case("normal/1d", {"a": Normal((17,))})
inputs_case("normal/4d", {"a": Normal(S4)})
inputs_case("normal/scalar", {"a": Normal(())})
inputs_case("normal/locScale", {"a": Normal(S, loc=2.5, scale=0.25)})
inputs_case("normal/float16", {"a": Normal(S, dtype="float16")})
inputs_case("normal/bfloat16", {"a": Normal(S, dtype="bfloat16")})
inputs_case("uniform/unit", {"a": Uniform(S)})
inputs_case("uniform/unit/seed2", {"a": Uniform(S)})
inputs_case("uniform/range", {"a": Uniform(S, low=0.1, high=2.0)})
inputs_case("uniform/negative", {"a": Uniform(S, low=-0.9, high=0.9)})
inputs_case("uniform/float16", {"a": Uniform(S, dtype="float16")})
inputs_case("randInt/small", {"a": RandInt(0, 4)})
inputs_case("randInt/large", {"a": RandInt(-100, 1024, (9,))})
inputs_case("randInt/uint32", {"a": RandInt(0, 6, (4,), dtype="uint32")})
inputs_case("bernoulli/half", {"a": Bernoulli(S)})
inputs_case("bernoulli/quarter", {"a": Bernoulli(S, p=0.25)})

# draw order: successive draws from one state must match python's global state,
# which is what lets the op cases declare several inputs and only check results
inputs_case("sequence/normal", {"a": Normal(S), "b": Normal(S), "c": Normal(S)})
inputs_case(
    "sequence/mixed",
    {"a": Normal(S), "b": Uniform(S, low=0.1, high=2.0), "c": RandInt(0, 4), "d": Bernoulli(S)},
)
inputs_case(
    "sequence/shapes",
    {"a": Normal((10, 8)), "b": Normal((8, 13)), "c": Normal((2, 3, 4))},
)


def unary(name: str, py_name: str, swift_name: str, spec=None, *, file="Elementwise") -> None:
    """`f(a)` for a single array argument"""
    case(
        name,
        file,
        {"a": spec or Normal(S)},
        f"mx.{py_name}(a)",
        f"MLX.{swift_name}(a)",
    )


def binary(name: str, py_name: str, swift_name: str, lhs=None, rhs=None) -> None:
    """`f(a, b)` for two array arguments"""
    case(
        name,
        "Binary",
        {"a": lhs or Normal(S), "b": rhs or Normal(S)},
        f"mx.{py_name}(a, b)",
        f"MLX.{swift_name}(a, b)",
    )


# --------------------------------------------------------------- elementwise

POSITIVE = Uniform(S, low=0.1, high=2.0)
UNIT = Uniform(S, low=-0.9, high=0.9)

unary("exp", "exp", "exp")
unary("expm1", "expm1", "expm1")
unary("log", "log", "log", POSITIVE)
unary("log2", "log2", "log2", POSITIVE)
unary("log10", "log10", "log10", POSITIVE)
unary("log1p", "log1p", "log1p", POSITIVE)
unary("sqrt", "sqrt", "sqrt", POSITIVE)
unary("rsqrt", "rsqrt", "rsqrt", POSITIVE)
unary("square", "square", "square")
unary("abs", "abs", "abs")
unary("negative", "negative", "negative")
unary("sign", "sign", "sign")
unary("floor", "floor", "floor")
unary("ceil", "ceil", "ceil")
unary("trunc", "trunc", "trunc")
unary("reciprocal", "reciprocal", "reciprocal", Uniform(S, low=0.5, high=2.0))
unary("sin", "sin", "sin")
unary("cos", "cos", "cos")
unary("tan", "tan", "tan")
unary("sinh", "sinh", "sinh")
unary("cosh", "cosh", "cosh")
unary("tanh", "tanh", "tanh")
unary("asin", "arcsin", "asin", UNIT)
unary("acos", "arccos", "acos", UNIT)
unary("atan", "arctan", "atan", UNIT)
unary("asinh", "arcsinh", "asinh")
unary("acosh", "arccosh", "acosh", Uniform(S, low=1.0, high=3.0))
unary("atanh", "arctanh", "atanh", UNIT)
unary("erf", "erf", "erf")
unary("erfInverse", "erfinv", "erfInverse", UNIT)
unary("sigmoid", "sigmoid", "sigmoid")
unary("degrees", "degrees", "degrees")
unary("radians", "radians", "radians")
unary("stopGradient", "stop_gradient", "stopGradient")

case(
    "round",
    "Elementwise",
    {"a": Normal(S, scale=10.0)},
    "mx.round(a)",
    "MLX.round(a)",
)
case(
    "round/decimals",
    "Elementwise",
    {"a": Normal(S, scale=10.0)},
    "mx.round(a, 2)",
    "MLX.round(a, decimals: 2)",
)
case(
    "clip",
    "Elementwise",
    {"a": Normal(S)},
    "mx.clip(a, -0.5, 0.5)",
    "MLX.clip(a, min: -0.5, max: 0.5)",
)
case(
    "logicalNot",
    "Elementwise",
    {"a": Bernoulli(S)},
    "mx.logical_not(a)",
    "MLX.logicalNot(a)",
)

# predicates on an input containing inf/nan
unary("isNaN", "isnan", "isNaN", SPECIAL)
unary("isInf", "isinf", "isInf", SPECIAL)
unary("isFinite", "isfinite", "isFinite", SPECIAL)
unary("isPosInf", "isposinf", "isPosInf", SPECIAL)
unary("isNegInf", "isneginf", "isNegInf", SPECIAL)
case(
    "nanToNum",
    "Elementwise",
    {"a": SPECIAL},
    "mx.nan_to_num(a, nan=1.0, posinf=100.0, neginf=-100.0)",
    "MLX.nanToNum(a, nan: 1.0, posInf: 100.0, negInf: -100.0)",
    note="arguments are explicit: python defaults posinf/neginf to the dtype "
    "max/min while Swift defaults them to 0",
)

# MLXArray method spellings
case("abs/method", "Elementwise", {"a": Normal(S)}, "a.abs()", "a.abs()")
case("exp/method", "Elementwise", {"a": Normal(S)}, "a.exp()", "a.exp()")
case("sqrt/method", "Elementwise", {"a": POSITIVE}, "a.sqrt()", "a.sqrt()")
case("round/method", "Elementwise", {"a": Normal(S, scale=10.0)}, "a.round()", "a.round()")

# dtype conversions
case(
    "asType/float16",
    "Elementwise",
    {"a": Normal(S)},
    "a.astype(mx.float16)",
    "a.asType(.float16)",
)
case(
    "asType/int32",
    "Elementwise",
    {"a": Normal(S, scale=10.0)},
    "a.astype(mx.int32)",
    "a.asType(.int32)",
)
case(
    "exp/float16",
    "Elementwise",
    {"a": Normal(S, dtype="float16")},
    "mx.exp(a)",
    "MLX.exp(a)",
)
case(
    "exp/bfloat16",
    "Elementwise",
    {"a": Normal(S, dtype="bfloat16")},
    "mx.exp(a)",
    "MLX.exp(a)",
)

# -------------------------------------------------------------------- binary

binary("add", "add", "add")
binary("subtract", "subtract", "subtract")
binary("multiply", "multiply", "multiply")
binary("divide", "divide", "divide")
binary("remainder", "remainder", "remainder")
binary("maximum", "maximum", "maximum")
binary("minimum", "minimum", "minimum")
binary("logAddExp", "logaddexp", "logAddExp")
binary("atan2", "arctan2", "atan2")
binary("floorDivide", "floor_divide", "floorDivide")
binary("pow", "power", "pow", POSITIVE, POSITIVE)
binary("equal", "equal", "equal", RandInt(0, 4), RandInt(0, 4))
binary("notEqual", "not_equal", "notEqual", RandInt(0, 4), RandInt(0, 4))
binary("less", "less", "less")
binary("lessEqual", "less_equal", "lessEqual")
binary("greater", "greater", "greater")
binary("greaterEqual", "greater_equal", "greaterEqual")

# operators
for op_name, op in [
    ("add", "+"),
    ("subtract", "-"),
    ("multiply", "*"),
    ("divide", "/"),
    ("remainder", "%"),
]:
    case(f"operator/{op_name}", "Binary", {"a": Normal(S), "b": Normal(S)}, f"a {op} b", f"a {op} b")
    case(f"operator/{op_name}/scalarRHS", "Binary", {"a": Normal(S)}, f"a {op} 1.3", f"a {op} 1.3")
    case(f"operator/{op_name}/scalarLHS", "Binary", {"a": Normal(S)}, f"0.5 {op} a", f"0.5 {op} a")

case(
    "operator/pow",
    "Binary",
    {"a": POSITIVE, "b": POSITIVE},
    "a ** b",
    "a ** b",
)
case(
    "operator/negate",
    "Binary",
    {"a": Normal(S)},
    "-a",
    "-a",
)

for op_name, py_op, swift_op in [
    ("equal", "==", ".=="),
    ("notEqual", "!=", ".!="),
    ("less", "<", ".<"),
    ("lessEqual", "<=", ".<="),
    ("greater", ">", ".>"),
    ("greaterEqual", ">=", ".>="),
]:
    case(
        f"operator/{op_name}",
        "Binary",
        {"a": RandInt(0, 4), "b": RandInt(0, 4)},
        f"a {py_op} b",
        f"a {swift_op} b",
    )

# logical (bool inputs)
case(
    "logicalAnd",
    "Binary",
    {"a": Bernoulli(S), "b": Bernoulli(S)},
    "mx.logical_and(a, b)",
    "MLX.logicalAnd(a, b)",
)
case(
    "logicalOr",
    "Binary",
    {"a": Bernoulli(S), "b": Bernoulli(S)},
    "mx.logical_or(a, b)",
    "MLX.logicalOr(a, b)",
)
case(
    "logicalXor",
    "Binary",
    {"a": Bernoulli(S), "b": Bernoulli(S)},
    "mx.logical_xor(a, b)",
    "MLX.logicalXor(a, b)",
)
case(
    "which",
    "Binary",
    {"mask": Bernoulli(S), "a": Normal(S), "b": Normal(S)},
    "mx.where(mask, a, b)",
    "MLX.which(mask, a, b)",
)

# bitwise (integer inputs)
INTS = RandInt(0, 16)
case("bitwiseAnd", "Binary", {"a": INTS, "b": INTS}, "mx.bitwise_and(a, b)", "MLX.bitwiseAnd(a, b)")
case("bitwiseOr", "Binary", {"a": INTS, "b": INTS}, "mx.bitwise_or(a, b)", "MLX.bitwiseOr(a, b)")
case("bitwiseXOr", "Binary", {"a": INTS, "b": INTS}, "mx.bitwise_xor(a, b)", "MLX.bitwiseXOr(a, b)")
case("bitwiseInvert", "Binary", {"a": INTS}, "mx.bitwise_invert(a)", "MLX.bitwiseInvert(a)")
case(
    "leftShift",
    "Binary",
    {"a": INTS, "b": RandInt(0, 4)},
    "mx.left_shift(a, b)",
    "MLX.leftShift(a, b)",
)
case(
    "rightShift",
    "Binary",
    {"a": RandInt(0, 1024), "b": RandInt(0, 4)},
    "mx.right_shift(a, b)",
    "MLX.rightShift(a, b)",
)

# --------------------------------------------------------------- matmul-ish

case(
    "matmul",
    "Binary",
    {"a": Normal((10, 8)), "b": Normal((8, 13))},
    "mx.matmul(a, b)",
    "MLX.matmul(a, b)",
)
case(
    "matmul/batched",
    "Binary",
    {"a": Normal((2, 4, 8)), "b": Normal((2, 8, 3))},
    "mx.matmul(a, b)",
    "MLX.matmul(a, b)",
)
case(
    "inner",
    "Binary",
    {"a": Normal((12,)), "b": Normal((12,))},
    "mx.inner(a, b)",
    "MLX.inner(a, b)",
)
case(
    "outer",
    "Binary",
    {"a": Normal((5,)), "b": Normal((4,))},
    "mx.outer(a, b)",
    "MLX.outer(a, b)",
)
case(
    "vecdot",
    "Binary",
    {"a": Normal((4, 8)), "b": Normal((4, 8))},
    "mx.vecdot(a, b)",
    "MLX.vecdot(a, b)",
)
case(
    "tensordot",
    "Binary",
    {"a": Normal((2, 3, 4)), "b": Normal((4, 3, 2))},
    "mx.tensordot(a, b, axes=([1, 2], [1, 0]))",
    "MLX.tensordot(a, b, axes: ([1, 2], [1, 0]))",
)
case(
    "addMM",
    "Binary",
    {"c": Normal((4, 5)), "a": Normal((4, 6)), "b": Normal((6, 5))},
    "mx.addmm(c, a, b, alpha=0.5, beta=2.0)",
    "MLX.addMM(c, a, b, alpha: 0.5, beta: 2.0)",
)
case(
    "kron",
    "Binary",
    {"a": Normal((2, 3)), "b": Normal((3, 2))},
    "mx.kron(a, b)",
    "MLX.kron(a, b)",
)
# these two exist to pin the primitives Muon's Newton-Schulz iteration is built
# from, so a mismatch there can be attributed to the optimizer rather than the ops
case(
    "norm/frobeniusKeepDims",
    "Binary",
    {"a": Normal((3, 4))},
    "mx.linalg.norm(a, keepdims=True)",
    "MLX.norm(a, keepDims: true)",
)
case(
    "addMM/newtonSchulzForm",
    "Binary",
    {"a": Normal((3, 3))},
    "mx.addmm(-4.7750 * a, a, a, beta=1.0, alpha=2.0315)",
    "MLX.addMM(-4.7750 * a, a, a, alpha: 2.0315, beta: 1.0)",
)
case(
    "einsum",
    "Binary",
    {"a": Normal((4, 5)), "b": Normal((5, 6))},
    'mx.einsum("ij,jk->ik", a, b)',
    'MLX.einsum("ij,jk->ik", a, b)',
)

# ----------------------------------------------------------------- reduction

REDUCTIONS = [
    # (case name, python name, swift name)
    ("sum", "sum", "sum"),
    ("mean", "mean", "mean"),
    ("min", "min", "min"),
    ("max", "max", "max"),
    ("product", "prod", "product"),
    ("logSumExp", "logsumexp", "logSumExp"),
]

for name, py_name, swift_name in REDUCTIONS:
    case(name, "Reduction", {"a": Normal(S)}, f"mx.{py_name}(a)", f"MLX.{swift_name}(a)")
    case(
        f"{name}/axis",
        "Reduction",
        {"a": Normal(S)},
        f"mx.{py_name}(a, axis=-1)",
        f"MLX.{swift_name}(a, axis: -1)",
    )
    case(
        f"{name}/axes",
        "Reduction",
        {"a": Normal(S4)},
        f"mx.{py_name}(a, axis=[0, -1])",
        f"MLX.{swift_name}(a, axes: [0, -1])",
    )
    case(
        f"{name}/keepDims",
        "Reduction",
        {"a": Normal(S4)},
        f"mx.{py_name}(a, axis=[0, -1], keepdims=True)",
        f"MLX.{swift_name}(a, axes: [0, -1], keepDims: true)",
    )
    case(
        f"{name}/method",
        "Reduction",
        {"a": Normal(S)},
        f"a.{py_name}(axis=-1)",
        f"a.{swift_name}(axis: -1)",
    )

for name, py_name, swift_name in [("variance", "var", "variance"), ("std", "std", "std")]:
    case(name, "Reduction", {"a": Normal(S)}, f"mx.{py_name}(a)", f"MLX.{swift_name}(a)")
    case(
        f"{name}/axis",
        "Reduction",
        {"a": Normal(S)},
        f"mx.{py_name}(a, axis=-1)",
        f"MLX.{swift_name}(a, axis: -1)",
    )
    case(
        f"{name}/ddof",
        "Reduction",
        {"a": Normal(S)},
        f"mx.{py_name}(a, axis=-1, ddof=1)",
        f"MLX.{swift_name}(a, axis: -1, ddof: 1)",
    )

case("median", "Reduction", {"a": Normal(S)}, "mx.median(a)", "MLX.median(a)")
case(
    "median/axis",
    "Reduction",
    {"a": Normal(S)},
    "mx.median(a, axis=-1)",
    "MLX.median(a, axis: -1)",
)

for name, py_name, swift_name in [("all", "all", "all"), ("any", "any", "any")]:
    case(name, "Reduction", {"a": Bernoulli(S)}, f"mx.{py_name}(a)", f"MLX.{swift_name}(a)")
    case(
        f"{name}/axis",
        "Reduction",
        {"a": Bernoulli(S)},
        f"mx.{py_name}(a, axis=-1)",
        f"MLX.{swift_name}(a, axis: -1)",
    )

for name, py_name, swift_name in [("argMin", "argmin", "argMin"), ("argMax", "argmax", "argMax")]:
    case(name, "Reduction", {"a": Normal(S)}, f"mx.{py_name}(a)", f"MLX.{swift_name}(a)")
    case(
        f"{name}/axis",
        "Reduction",
        {"a": Normal(S)},
        f"mx.{py_name}(a, axis=-1)",
        f"MLX.{swift_name}(a, axis: -1)",
    )

CUMULATIVE = [
    ("cumsum", "cumsum", "cumsum"),
    ("cumprod", "cumprod", "cumprod"),
    ("cummax", "cummax", "cummax"),
    ("cummin", "cummin", "cummin"),
    ("logCumsumExp", "logcumsumexp", "logCumsumExp"),
]

for name, py_name, swift_name in CUMULATIVE:
    case(
        f"{name}/axis",
        "Reduction",
        {"a": Normal(S)},
        f"mx.{py_name}(a, axis=-1)",
        f"MLX.{swift_name}(a, axis: -1)",
    )
    case(
        f"{name}/reverse",
        "Reduction",
        {"a": Normal(S)},
        f"mx.{py_name}(a, axis=-1, reverse=True)",
        f"MLX.{swift_name}(a, axis: -1, reverse: true)",
    )
    case(
        f"{name}/exclusive",
        "Reduction",
        {"a": Normal(S)},
        f"mx.{py_name}(a, axis=-1, inclusive=False)",
        f"MLX.{swift_name}(a, axis: -1, inclusive: false)",
    )

case(
    "softmax/axis",
    "Reduction",
    {"a": Normal(S)},
    "mx.softmax(a, axis=-1)",
    "MLX.softmax(a, axis: -1)",
)
case(
    "softmax/axes",
    "Reduction",
    {"a": Normal(S4)},
    "mx.softmax(a, axis=[0, -1])",
    "MLX.softmax(a, axes: [0, -1])",
)
case(
    "softmax/precise",
    "Reduction",
    {"a": Normal(S)},
    "mx.softmax(a.astype(mx.float32), axis=-1)",
    "MLX.softmax(a, axis: -1, precise: true)",
    note="precise: computes in float32; the python side is already float32",
)

# --------------------------------------------------------------------- shape

case(
    "concatenated",
    "Shape",
    {"a": Normal(S), "b": Normal(S)},
    "mx.concatenate([a, b])",
    "MLX.concatenated([a, b])",
)
case(
    "concatenated/axis",
    "Shape",
    {"a": Normal(S), "b": Normal(S)},
    "mx.concatenate([a, b], axis=1)",
    "MLX.concatenated([a, b], axis: 1)",
)
case(
    "stacked",
    "Shape",
    {"a": Normal(S), "b": Normal(S)},
    "mx.stack([a, b])",
    "MLX.stacked([a, b])",
)
case(
    "stacked/axis",
    "Shape",
    {"a": Normal(S), "b": Normal(S)},
    "mx.stack([a, b], axis=-1)",
    "MLX.stacked([a, b], axis: -1)",
)
case("flipped", "Shape", {"a": Normal(S)}, "mx.flip(a)", "MLX.flipped(a)")
case("flipped/axis", "Shape", {"a": Normal(S)}, "mx.flip(a, axis=1)", "MLX.flipped(a, axis: 1)")
case(
    "tiled",
    "Shape",
    {"a": Normal(S)},
    "mx.tile(a, [2, 1])",
    "MLX.tiled(a, repetitions: [2, 1])",
)
case(
    "repeated",
    "Shape",
    {"a": Normal(S)},
    "mx.repeat(a, 2, axis=1)",
    "MLX.repeated(a, count: 2, axis: 1)",
)
case(
    "padded",
    "Shape",
    {"a": Normal(S)},
    "mx.pad(a, 2)",
    "MLX.padded(a, width: 2)",
)
case(
    "padded/edge",
    "Shape",
    {"a": Normal(S)},
    'mx.pad(a, 2, mode="edge")',
    "MLX.padded(a, width: 2, mode: .edge)",
)
case(
    "padded/widths",
    "Shape",
    {"a": Normal(S)},
    "mx.pad(a, [(1, 2), (0, 1)])",
    "MLX.padded(a, widths: [IntOrPair([1, 2]), IntOrPair([0, 1])])",
)
case(
    "padded/value",
    "Shape",
    {"a": Normal(S)},
    "mx.pad(a, 1, constant_values=2.5)",
    "MLX.padded(a, width: 1, value: MLXArray(Float(2.5)))",
)
case(
    "roll",
    "Shape",
    {"a": Normal(S)},
    "mx.roll(a, 2, axis=0)",
    "MLX.roll(a, shift: 2, axis: 0)",
)
case(
    "transposed",
    "Shape",
    {"a": Normal(S)},
    "mx.transpose(a)",
    "MLX.transposed(a)",
)
case(
    "transposed/axes",
    "Shape",
    {"a": Normal((2, 3, 4))},
    "mx.transpose(a, [2, 0, 1])",
    "MLX.transposed(a, axes: [2, 0, 1])",
)
case(
    "reshaped",
    "Shape",
    {"a": Normal(S)},
    "mx.reshape(a, (3, 4))",
    "MLX.reshaped(a, [3, 4])",
)
case(
    "squeezed",
    "Shape",
    {"a": Normal((4, 1, 3))},
    "mx.squeeze(a, axis=1)",
    "MLX.squeezed(a, axis: 1)",
)
case(
    "expandedDimensions",
    "Shape",
    {"a": Normal(S)},
    "mx.expand_dims(a, axis=[0, -1])",
    "MLX.expandedDimensions(a, axes: [0, -1])",
)
case(
    "movedAxis",
    "Shape",
    {"a": Normal((2, 3, 4))},
    "mx.moveaxis(a, 0, 2)",
    "MLX.movedAxis(a, source: 0, destination: 2)",
)
case(
    "swappedAxes",
    "Shape",
    {"a": Normal((2, 3, 4))},
    "mx.swapaxes(a, 0, 2)",
    "MLX.swappedAxes(a, 0, 2)",
)
case(
    "flattened",
    "Shape",
    {"a": Normal((2, 3, 4))},
    "mx.flatten(a)",
    "MLX.flattened(a)",
)
case(
    "flattened/range",
    "Shape",
    {"a": Normal((2, 3, 4))},
    "mx.flatten(a, 1, 2)",
    "MLX.flattened(a, start: 1, end: 2)",
)
case(
    "unflatten",
    "Shape",
    {"a": Normal((2, 12))},
    "mx.unflatten(a, 1, (3, 4))",
    "MLX.unflatten(a, axis: 1, shape: [3, 4])",
)
case(
    "broadcast",
    "Shape",
    {"a": Normal(S)},
    "mx.broadcast_to(a, [2, 4, 3])",
    "MLX.broadcast(a, to: [2, 4, 3])",
)
case(
    "atLeast2D",
    "Shape",
    {"a": Normal((5,))},
    "mx.atleast_2d(a)",
    "MLX.atLeast2D(a)",
)
case(
    "take",
    "Shape",
    {"a": Normal(S), "indices": RandInt(0, 4, (5,))},
    "mx.take(a, indices, axis=0)",
    "MLX.take(a, indices, axis: 0)",
)
case(
    "take/flat",
    "Shape",
    {"a": Normal(S), "indices": RandInt(0, 12, (4,))},
    "mx.take(a, indices)",
    "MLX.take(a, indices)",
)
case(
    "takeAlong",
    "Shape",
    {"a": Normal(S), "indices": RandInt(0, 3, (4, 1))},
    "mx.take_along_axis(a, indices, axis=1)",
    "MLX.takeAlong(a, indices, axis: 1)",
)
case(
    "sorted",
    "Shape",
    {"a": Normal(S)},
    "mx.sort(a, axis=-1)",
    "MLX.sorted(a, axis: -1)",
)
case(
    "top",
    "Shape",
    {"a": Normal(S)},
    "mx.topk(a, 2, axis=-1)",
    "MLX.top(a, k: 2, axis: -1)",
)
case("tril", "Shape", {"a": Normal((4, 4))}, "mx.tril(a)", "MLX.tril(a)")
case("triu/k", "Shape", {"a": Normal((4, 4))}, "mx.triu(a, k=1)", "MLX.triu(a, k: 1)")
case("tri", "Shape", {}, "mx.tri(4, 3, 0, dtype=mx.float32)", "MLX.tri(4, m: 3, k: 0, dtype: .float32)")
case("diag", "Shape", {"a": Normal((4, 4))}, "mx.diag(a)", "MLX.diag(a)")
case(
    "diagonal/offset",
    "Shape",
    {"a": Normal((4, 4))},
    "mx.diagonal(a, offset=1)",
    "MLX.diagonal(a, offset: 1)",
)

# ------------------------------------------------------------------ indexing

case(
    "index/range",
    "Indexing",
    {"a": Normal((6, 5))},
    "a[1:4, 0:3]",
    "a[1 ..< 4, 0 ..< 3]",
)
case(
    "index/lastColumn",
    "Indexing",
    {"a": Normal((6, 5))},
    "a[:, -1]",
    "a[0..., -1]",
)
case(
    "index/newAxis",
    "Indexing",
    {"a": Normal((6, 5))},
    "a[..., None]",
    "a[.ellipsis, .newAxis]",
)
case(
    "index/stride",
    "Indexing",
    {"a": Normal((6, 5))},
    "a[::2]",
    "a[.stride(by: 2)]",
)
case(
    "index/reverse",
    "Indexing",
    {"a": Normal((6, 5))},
    "a[..., ::-1]",
    "a[.ellipsis, .stride(by: -1)]",
)
case(
    "index/array",
    "Indexing",
    {"a": Normal((6, 5)), "indices": RandInt(0, 6, (4,))},
    "a[indices]",
    "a[indices]",
)

# --------------------------------------------------- elementwise (continued)

case("positive", "Elementwise", {"a": Normal(S)}, "mx.positive(a)", "MLX.positive(a)")
case(
    "view/int16",
    "Elementwise",
    {"a": RandInt(0, 1024)},
    "mx.view(a, mx.int16)",
    "MLX.view(a, dtype: .int16)",
)
case(
    "contiguous",
    "Elementwise",
    {"a": Normal((4, 3))},
    "mx.contiguous(a.T)",
    "MLX.contiguous(a.T)",
)
case(
    "isClose",
    "Elementwise",
    {"a": Normal(S), "b": Normal(S)},
    "mx.isclose(a, b, rtol=0.5, atol=0.1)",
    "MLX.isClose(a, b, rtol: 0.5, atol: 0.1)",
)
case(
    "allClose",
    "Elementwise",
    {"a": Normal(S)},
    "mx.allclose(a, a + 0.001, rtol=0.01)",
    "MLX.allClose(a, a + 0.001, rtol: 0.01)",
)
case(
    "arrayEqual",
    "Elementwise",
    {"a": RandInt(0, 4)},
    "mx.array_equal(a, a)",
    "MLX.arrayEqual(a, a)",
)
case(
    "arrayEqual/false",
    "Elementwise",
    {"a": RandInt(0, 4), "b": RandInt(0, 4)},
    "mx.array_equal(a, b)",
    "MLX.arrayEqual(a, b)",
)

# complex: built from two real arrays, verified as real/imaginary parts
COMPLEX = Expression(py="r + 1j * i", swift="r + i.asImaginary()")

case(
    "realPart",
    "Elementwise",
    {"r": Normal(S), "i": Normal(S), "c": COMPLEX},
    "mx.real(c)",
    "c.realPart()",
    note="Swift exposes realPart() as a method, not a free function",
)
case(
    "imaginaryPart",
    "Elementwise",
    {"r": Normal(S), "i": Normal(S), "c": COMPLEX},
    "mx.imag(c)",
    "c.imaginaryPart()",
)
case(
    "conjugate",
    "Elementwise",
    {"r": Normal(S), "i": Normal(S), "c": COMPLEX},
    "mx.conjugate(c)",
    "MLX.conjugate(c)",
)
case(
    "asImaginary",
    "Elementwise",
    {"r": Normal(S), "i": Normal(S), "c": COMPLEX},
    "c",
    "c",
    note="verifies the complex value the other complex cases are built from",
)

# -------------------------------------------------------- shape (continued)

case(
    "putAlong",
    "Shape",
    {"a": Normal(S), "indices": RandInt(0, 3, (4, 1)), "values": Normal((4, 1))},
    "mx.put_along_axis(a, indices, values, axis=1)",
    "MLX.putAlong(a, indices, values: values, axis: 1)",
)
case(
    "argSort",
    "Shape",
    {"a": Normal(S)},
    "mx.argsort(a, axis=-1)",
    "MLX.argSort(a, axis: -1)",
)
case(
    "argSort/flat",
    "Shape",
    {"a": Normal(S)},
    "mx.argsort(a, axis=None)",
    "MLX.argSort(a)",
)
case(
    "partitioned",
    "Shape",
    {"a": Normal(S)},
    "mx.partition(a, 1, axis=-1)",
    "MLX.partitioned(a, kth: 1, axis: -1)",
)
case(
    "argPartition",
    "Shape",
    {"a": Normal(S)},
    "mx.argpartition(a, 1, axis=-1)",
    "MLX.argPartition(a, kth: 1, axis: -1)",
)
case("diff", "Shape", {"a": Normal(S)}, "mx.diff(a)", "MLX.diff(a)")
case(
    "diff/n2",
    "Shape",
    {"a": Normal((4, 5))},
    "mx.diff(a, n=2, axis=1)",
    "MLX.diff(a, n: 2, axis: 1)",
)
case(
    "asStrided",
    "Shape",
    {"a": Normal(S)},
    "mx.as_strided(a, [3, 3], [1, 2], 0)",
    "MLX.asStrided(a, [3, 3], strides: [1, 2], offset: 0)",
)
case("atLeast1D", "Shape", {"a": Normal(())}, "mx.atleast_1d(a)", "MLX.atLeast1D(a)")
case("atLeast3D", "Shape", {"a": Normal((5,))}, "mx.atleast_3d(a)", "MLX.atLeast3D(a)")
case("trace", "Shape", {"a": Normal((4, 4))}, "mx.trace(a)", "MLX.trace(a)")
case(
    "trace/offset",
    "Shape",
    {"a": Normal((4, 4))},
    "mx.trace(a, offset=1)",
    "MLX.trace(a, offset: 1)",
)
case(
    "hadamardTransform",
    "Shape",
    {"a": Normal((4, 16))},
    "mx.hadamard_transform(a)",
    "MLX.hadamardTransform(a)",
)
case(
    "hadamardTransform/scale",
    "Shape",
    {"a": Normal((4, 16))},
    "mx.hadamard_transform(a, scale=0.25)",
    "MLX.hadamardTransform(a, scale: 0.25)",
)

# ------------------------------------------------------------------- factory
#
# no inputs: these are deterministic, so they also serve as a check that the
# generator's expectations are not accidentally seed dependent

case("arange/stop", "Factory", {}, "mx.arange(10)", "MLX.arange(10)")
case("arange/range", "Factory", {}, "mx.arange(2, 12)", "MLX.arange(2, 12)")
case(
    "arange/step",
    "Factory",
    {},
    "mx.arange(0.0, 2.0, 0.25)",
    "MLX.arange(0.0, 2.0, step: 0.25)",
)
case(
    "arange/dtype",
    "Factory",
    {},
    "mx.arange(0, 10, 2, dtype=mx.int32)",
    "MLX.arange(0, 10, step: 2, dtype: .int32)",
)
case(
    "linspace",
    "Factory",
    {},
    "mx.linspace(0.0, 1.0, 9)",
    "MLX.linspace(0.0, 1.0, count: 9)",
)
case(
    "linspace/int",
    "Factory",
    {},
    "mx.linspace(0, 10, 6)",
    "MLX.linspace(0, 10, count: 6)",
    note="integer bounds still produce float32, matching python",
)
case(
    "linspace/dtype",
    "Factory",
    {},
    "mx.linspace(0, 10, 6, dtype=mx.int32)",
    "MLX.linspace(0, 10, count: 6, dtype: .int32)",
)
case(
    "linspace/endpoint",
    "Factory",
    {},
    "mx.linspace(0.0, 1.0, 6)[:-1]",
    "MLX.linspace(0.0, 1.0, count: 5, endpoint: false)",
    note="python has no endpoint parameter: the half-open interval is the "
    "inclusive one with the last sample dropped",
)
case("zeros", "Factory", {}, "mx.zeros([3, 4])", "MLX.zeros([3, 4])")
case(
    "zeros/dtype",
    "Factory",
    {},
    "mx.zeros([3, 4], dtype=mx.int32)",
    "MLX.zeros([3, 4], dtype: .int32)",
)
case("ones", "Factory", {}, "mx.ones([3, 4])", "MLX.ones([3, 4])")
case(
    "ones/dtype",
    "Factory",
    {},
    "mx.ones([3, 4], dtype=mx.int16)",
    "MLX.ones([3, 4], dtype: .int16)",
)
case(
    "zeros/like",
    "Factory",
    {"a": Normal(S)},
    "mx.zeros_like(a)",
    "MLX.zeros(like: a)",
)
case("ones/like", "Factory", {"a": Normal(S)}, "mx.ones_like(a)", "MLX.ones(like: a)")
case(
    "full",
    "Factory",
    {},
    "mx.full([2, 3], 2.5)",
    "MLX.full([2, 3], values: Float(2.5))",
)
case(
    "full/dtype",
    "Factory",
    {},
    "mx.full([2, 3], 7, dtype=mx.int32)",
    "MLX.full([2, 3], values: MLXArray(7), dtype: .int32)",
)
case("eye", "Factory", {}, "mx.eye(4)", "MLX.eye(4)")
case(
    "eye/rectangular",
    "Factory",
    {},
    "mx.eye(4, 6, 1)",
    "MLX.eye(4, m: 6, k: 1)",
)
case("identity", "Factory", {}, "mx.identity(5)", "MLX.identity(5)")
case("bartlett", "Factory", {}, "mx.bartlett(16)", "MLX.bartlett(16)")
case("hanning", "Factory", {}, "mx.hanning(16)", "MLX.hanning(16)")
case("hamming", "Factory", {}, "mx.hamming(16)", "MLX.hamming(16)")
case("blackman", "Factory", {}, "mx.blackman(16)", "MLX.blackman(16)")

# --------------------------------------------------------------- convolution

CONV1D_INPUT = Normal((2, 10, 4))
CONV1D_WEIGHT = Normal((3, 3, 4))

case(
    "convolve/full",
    "Convolution",
    {"a": Normal((20,)), "v": Normal((4,))},
    "mx.convolve(a, v)",
    "MLX.convolve(a, v)",
)
case(
    "convolve/same",
    "Convolution",
    {"a": Normal((20,)), "v": Normal((4,))},
    'mx.convolve(a, v, mode="same")',
    "MLX.convolve(a, v, mode: .same)",
)
case(
    "convolve/valid",
    "Convolution",
    {"a": Normal((20,)), "v": Normal((4,))},
    'mx.convolve(a, v, mode="valid")',
    "MLX.convolve(a, v, mode: .valid)",
)
case(
    "conv1d",
    "Convolution",
    {"input": CONV1D_INPUT, "weight": CONV1D_WEIGHT},
    "mx.conv1d(input, weight)",
    "MLX.conv1d(input, weight)",
)
case(
    "conv1d/stridePadding",
    "Convolution",
    {"input": CONV1D_INPUT, "weight": CONV1D_WEIGHT},
    "mx.conv1d(input, weight, stride=2, padding=1)",
    "MLX.conv1d(input, weight, stride: 2, padding: 1)",
)
case(
    "conv1d/dilation",
    "Convolution",
    {"input": CONV1D_INPUT, "weight": CONV1D_WEIGHT},
    "mx.conv1d(input, weight, dilation=2)",
    "MLX.conv1d(input, weight, dilation: 2)",
)
case(
    "conv1d/groups",
    "Convolution",
    {"input": Normal((2, 10, 4)), "weight": Normal((4, 3, 2))},
    "mx.conv1d(input, weight, groups=2)",
    "MLX.conv1d(input, weight, groups: 2)",
)
case(
    "conv2d",
    "Convolution",
    {"input": Normal((2, 8, 8, 3)), "weight": Normal((4, 3, 3, 3))},
    "mx.conv2d(input, weight)",
    "MLX.conv2d(input, weight)",
)
case(
    "conv2d/stridePadding",
    "Convolution",
    {"input": Normal((2, 8, 8, 3)), "weight": Normal((4, 3, 3, 3))},
    "mx.conv2d(input, weight, stride=(2, 1), padding=(1, 0))",
    "MLX.conv2d(input, weight, stride: [2, 1], padding: [1, 0])",
)
case(
    "conv3d",
    "Convolution",
    {"input": Normal((1, 4, 6, 6, 2)), "weight": Normal((3, 2, 3, 3, 2))},
    "mx.conv3d(input, weight)",
    "MLX.conv3d(input, weight)",
)
case(
    "convTransposed1d",
    "Convolution",
    {"input": Normal((2, 8, 4)), "weight": Normal((3, 3, 4))},
    "mx.conv_transpose1d(input, weight)",
    "MLX.convTransposed1d(input, weight)",
)
case(
    "convTransposed1d/stride",
    "Convolution",
    {"input": Normal((2, 8, 4)), "weight": Normal((3, 3, 4))},
    "mx.conv_transpose1d(input, weight, stride=2, padding=1, output_padding=1)",
    "MLX.convTransposed1d(input, weight, stride: 2, padding: 1, outputPadding: 1)",
)
case(
    "convTransposed2d",
    "Convolution",
    {"input": Normal((2, 6, 6, 3)), "weight": Normal((4, 3, 3, 3))},
    "mx.conv_transpose2d(input, weight)",
    "MLX.convTransposed2d(input, weight)",
)
case(
    "convTransposed3d",
    "Convolution",
    {"input": Normal((1, 4, 4, 4, 2)), "weight": Normal((3, 2, 2, 2, 2))},
    "mx.conv_transpose3d(input, weight)",
    "MLX.convTransposed3d(input, weight)",
)
case(
    "convGeneral",
    "Convolution",
    {"input": Normal((2, 8, 8, 3)), "weight": Normal((4, 3, 3, 3))},
    "mx.conv_general(input, weight, stride=2, padding=1)",
    "MLX.convGeneral(input, weight, strides: 2, padding: 1)",
)
case(
    "convGeneral/flip",
    "Convolution",
    {"input": Normal((2, 8, 8, 3)), "weight": Normal((4, 3, 3, 3))},
    "mx.conv_general(input, weight, kernel_dilation=2, flip=True)",
    "MLX.convGeneral(input, weight, kernelDilation: 2, flip: true)",
)

# -------------------------------------------------------------- quantization

W = Uniform((64, 128), low=-1.0, high=1.0)
X = Normal((8, 128))

for bits in (2, 4, 8):
    case(
        f"quantized/bits{bits}/wq",
        "Quantization",
        {"w": W},
        f"mx.quantize(w, group_size=64, bits={bits})[0]",
        f"MLX.quantized(w, groupSize: 64, bits: {bits}).wq",
    )
    case(
        f"quantized/bits{bits}/scales",
        "Quantization",
        {"w": W},
        f"mx.quantize(w, group_size=64, bits={bits})[1]",
        f"MLX.quantized(w, groupSize: 64, bits: {bits}).scales",
    )
    case(
        f"quantized/bits{bits}/biases",
        "Quantization",
        {"w": W},
        f"mx.quantize(w, group_size=64, bits={bits})[2]",
        f"MLX.quantized(w, groupSize: 64, bits: {bits}).biases!",
    )
    case(
        f"dequantized/bits{bits}",
        "Quantization",
        {
            "w": W,
            "q": Expression(
                py=f"mx.quantize(w, group_size=64, bits={bits})",
                swift=f"MLX.quantized(w, groupSize: 64, bits: {bits})",
                array=False,
            ),
        },
        f"mx.dequantize(q[0], q[1], q[2], group_size=64, bits={bits})",
        f"MLX.dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: {bits})",
    )
    case(
        f"quantizedMM/bits{bits}",
        "Quantization",
        {
            "w": W,
            "x": X,
            "q": Expression(
                py=f"mx.quantize(w, group_size=64, bits={bits})",
                swift=f"MLX.quantized(w, groupSize: 64, bits: {bits})",
                array=False,
            ),
        },
        f"mx.quantized_matmul(x, q[0], q[1], q[2], group_size=64, bits={bits})",
        f"MLX.quantizedMM(x, q.wq, scales: q.scales, biases: q.biases, groupSize: 64, "
        f"bits: {bits})",
    )

case(
    "quantized/groupSize32",
    "Quantization",
    {"w": W},
    "mx.quantize(w, group_size=32, bits=4)[0]",
    "MLX.quantized(w, groupSize: 32, bits: 4).wq",
)
case(
    "quantizedMM/noTranspose",
    "Quantization",
    {
        "w": Uniform((128, 64), low=-1.0, high=1.0),
        "x": X,
        "q": Expression(
            py="mx.quantize(w, group_size=64, bits=4)",
            swift="MLX.quantized(w, groupSize: 64, bits: 4)",
            array=False,
        ),
    },
    "mx.quantized_matmul(x, q[0], q[1], q[2], transpose=False, group_size=64, bits=4)",
    "MLX.quantizedMM(x, q.wq, scales: q.scales, biases: q.biases, transpose: false, "
    "groupSize: 64, bits: 4)",
)
case(
    "gatherQuantizedMM",
    "Quantization",
    {
        "w": W,
        "x": X,
        "q": Expression(
            py="mx.quantize(w, group_size=64, bits=4)",
            swift="MLX.quantized(w, groupSize: 64, bits: 4)",
            array=False,
        ),
    },
    "mx.gather_qmm(x, q[0], q[1], q[2], group_size=64, bits=4)",
    "MLX.gatherQuantizedMM(x, q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: 4)",
    note="without indices this is quantizedMM; the indexed form needs the "
    "multi-input generator",
)
case(
    "blockMaskedMM",
    "Quantization",
    {"a": Normal((32, 64)), "b": Normal((64, 32))},
    "mx.block_masked_mm(a, b, 32)",
    "MLX.blockMaskedMM(a, b, blockSize: 32)",
)
case(
    "gatherMM",
    "Quantization",
    {
        "a": Normal((4, 3, 5)),
        "b": Normal((6, 5, 2)),
        "lhsIndices": RandInt(0, 4, (2,), dtype="uint32"),
        "rhsIndices": RandInt(0, 6, (2,), dtype="uint32"),
    },
    "mx.gather_mm(a, b, lhsIndices, rhsIndices)",
    "MLX.gatherMM(a, b, lhsIndices: lhsIndices, rhsIndices: rhsIndices)",
)
case(
    "toFP8",
    "Quantization",
    {"a": Uniform(S, low=-4.0, high=4.0)},
    "mx.to_fp8(a)",
    "MLX.toFP8(a)",
)
case(
    "fromFP8",
    "Quantization",
    {
        "a": Uniform(S, low=-4.0, high=4.0),
        "encoded": Expression(py="mx.to_fp8(a)", swift="MLX.toFP8(a)"),
    },
    "mx.from_fp8(encoded, dtype=mx.float32)",
    "MLX.fromFP8(encoded, dtype: .float32)",
)

# ----------------------------------------------------------------------- fft

FFT_1D = Normal((100,))
FFT_3D = Normal((8, 8, 8))
COMPLEX_1D = Expression(py="r + 1j * i", swift="r + i.asImaginary()")


def fft_case(name: str, py: str, swift: str, *, complex_input=True, shape=(100,)) -> None:
    if complex_input:
        inputs = {"r": Normal(shape), "i": Normal(shape), "c": COMPLEX_1D}
    else:
        inputs = {"c": Normal(shape)}
    case(name, "FFT", inputs, py, swift)


# 1D, complex input
for fn, swift_fn in [("fft", "fft"), ("ifft", "ifft")]:
    fft_case(fn, f"mx.fft.{fn}(c)", f"MLX.{swift_fn}(c)")
    fft_case(f"{fn}/nShort", f"mx.fft.{fn}(c, n=80)", f"MLX.{swift_fn}(c, n: 80)")
    fft_case(f"{fn}/nLong", f"mx.fft.{fn}(c, n=120)", f"MLX.{swift_fn}(c, n: 120)")
    fft_case(
        f"{fn}/axis",
        f"mx.fft.{fn}(c, axis=0)",
        f"MLX.{swift_fn}(c, axis: 0)",
        shape=(10, 10),
    )
    fft_case(
        f"{fn}/ortho",
        f'mx.fft.{fn}(c, norm="ortho")',
        f"MLX.{swift_fn}(c, norm: .ortho)",
    )

# rfft takes a real input, irfft takes a complex input and produces a real one
fft_case("rfft", "mx.fft.rfft(c)", "MLX.rfft(c)", complex_input=False)
fft_case("rfft/n", "mx.fft.rfft(c, n=80)", "MLX.rfft(c, n: 80)", complex_input=False)
fft_case(
    "rfft/forward",
    'mx.fft.rfft(c, norm="forward")',
    "MLX.rfft(c, norm: .forward)",
    complex_input=False,
)
fft_case("irfft", "mx.fft.irfft(c)", "MLX.irfft(c)")
fft_case("irfft/n", "mx.fft.irfft(c, n=120)", "MLX.irfft(c, n: 120)")

# 2D / ND
#
# note: `fft2`/`ifft2` default `axes` to [-2, -1], but the n-dimensional forms
# require `axes` whenever `s` is given
for fn in ("fft2", "ifft2", "fftn", "ifftn"):
    fft_case(fn, f"mx.fft.{fn}(c)", f"MLX.{fn}(c)", shape=(8, 8, 8))
    if fn.endswith("2"):
        fft_case(
            f"{fn}/s",
            f"mx.fft.{fn}(c, s=[3, 4])",
            f"MLX.{fn}(c, s: [3, 4])",
            shape=(8, 8, 8),
        )
    else:
        fft_case(
            f"{fn}/s",
            f"mx.fft.{fn}(c, s=[3, 4], axes=[0, 1])",
            f"MLX.{fn}(c, s: [3, 4], axes: [0, 1])",
            shape=(8, 8, 8),
        )
    fft_case(
        f"{fn}/axes",
        f"mx.fft.{fn}(c, axes=[0, 2])",
        f"MLX.{fn}(c, axes: [0, 2])",
        shape=(8, 8, 8),
    )

for fn in ("rfft2", "rfftn"):
    fft_case(fn, f"mx.fft.{fn}(c)", f"MLX.{fn}(c)", complex_input=False, shape=(8, 8, 8))
    fft_case(
        f"{fn}/axes",
        f"mx.fft.{fn}(c, axes=[0, 2])",
        f"MLX.{fn}(c, axes: [0, 2])",
        complex_input=False,
        shape=(8, 8, 8),
    )

for fn in ("irfft2", "irfftn"):
    fft_case(fn, f"mx.fft.{fn}(c)", f"MLX.{fn}(c)", shape=(8, 8, 8))
    fft_case(
        f"{fn}/axes",
        f"mx.fft.{fn}(c, axes=[0, 2])",
        f"MLX.{fn}(c, axes: [0, 2])",
        shape=(8, 8, 8),
    )

case("fftfreq", "FFT", {}, "mx.fft.fftfreq(16)", "MLX.fftfreq(16)")
case("fftfreq/d", "FFT", {}, "mx.fft.fftfreq(16, 0.25)", "MLX.fftfreq(16, d: 0.25)")
case("rfftfreq", "FFT", {}, "mx.fft.rfftfreq(16)", "MLX.rfftfreq(16)")
case(
    "fftshift",
    "FFT",
    {"a": Normal((8, 8))},
    "mx.fft.fftshift(a)",
    "MLX.fftshift(a)",
)
case(
    "fftshift/axes",
    "FFT",
    {"a": Normal((8, 8))},
    "mx.fft.fftshift(a, axes=[0])",
    "MLX.fftshift(a, axes: [0])",
)
case(
    "ifftshift",
    "FFT",
    {"a": Normal((8, 8))},
    "mx.fft.ifftshift(a)",
    "MLX.ifftshift(a)",
)

# -------------------------------------------------------------------- linalg
#
# linalg runs on the CPU in mlx; both sides say so explicitly.  Inputs are
# built to be well conditioned (SPD / triangular) so the results are stable.

SQUARE = Normal((4, 4))
SPD = Expression(
    py="mx.matmul(a, a.T) + 4.0 * mx.eye(4)",
    swift="MLX.matmul(a, a.T) + 4.0 * MLX.eye(4)",
)
LOWER = Expression(py="mx.tril(spd)", swift="MLX.tril(spd)")

case("norm", "Linalg", {"a": Normal(S)}, "mx.linalg.norm(a)", "MLX.norm(a, stream: .cpu)")
case(
    "norm/fro",
    "Linalg",
    {"a": Normal(S)},
    'mx.linalg.norm(a, ord="fro")',
    "MLX.norm(a, ord: .fro, stream: .cpu)",
)
case(
    "norm/ord1/axis",
    "Linalg",
    {"a": Normal(S)},
    "mx.linalg.norm(a, ord=1, axis=0)",
    "MLX.norm(a, ord: 1.0, axis: 0, stream: .cpu)",
)
case(
    "norm/axes",
    "Linalg",
    {"a": Normal((2, 4, 3))},
    "mx.linalg.norm(a, axis=[1, 2])",
    "MLX.norm(a, axes: [1, 2], stream: .cpu)",
)
case(
    "norm/keepDims",
    "Linalg",
    {"a": Normal(S)},
    "mx.linalg.norm(a, axis=-1, keepdims=True)",
    "MLX.norm(a, axis: -1, keepDims: true, stream: .cpu)",
)
case(
    "inv",
    "Linalg",
    {"a": SQUARE, "spd": SPD},
    "mx.linalg.inv(spd, stream=mx.cpu)",
    "MLX.inv(spd, stream: .cpu)",
)
case(
    "triInv",
    "Linalg",
    {"a": SQUARE, "spd": SPD, "lower": LOWER},
    "mx.linalg.tri_inv(lower, stream=mx.cpu)",
    "MLX.triInv(lower, stream: .cpu)",
)
case(
    "cholesky",
    "Linalg",
    {"a": SQUARE, "spd": SPD},
    "mx.linalg.cholesky(spd, stream=mx.cpu)",
    "MLX.cholesky(spd, stream: .cpu)",
)
case(
    "cholesky/upper",
    "Linalg",
    {"a": SQUARE, "spd": SPD},
    "mx.linalg.cholesky(spd, upper=True, stream=mx.cpu)",
    "MLX.cholesky(spd, upper: true, stream: .cpu)",
)
case(
    "choleskyInv",
    "Linalg",
    {
        "a": SQUARE,
        "spd": SPD,
        "l": Expression(
            py="mx.linalg.cholesky(spd, stream=mx.cpu)",
            swift="MLX.cholesky(spd, stream: .cpu)",
        ),
    },
    "mx.linalg.cholesky_inv(l, stream=mx.cpu)",
    "MLX.choleskyInv(l, stream: .cpu)",
)
case(
    "pinv",
    "Linalg",
    {"a": Normal((5, 3))},
    "mx.linalg.pinv(a, stream=mx.cpu)",
    "MLX.pinv(a, stream: .cpu)",
)
case(
    "cross",
    "Linalg",
    {"a": Normal((4, 3)), "b": Normal((4, 3))},
    "mx.linalg.cross(a, b)",
    "MLX.cross(a, b)",
)
case(
    "solve",
    "Linalg",
    {"a": SQUARE, "spd": SPD, "b": Normal((4, 2))},
    "mx.linalg.solve(spd, b, stream=mx.cpu)",
    "MLX.solve(spd, b, stream: .cpu)",
)
case(
    "solveTriangular",
    "Linalg",
    {"a": SQUARE, "spd": SPD, "lower": LOWER, "b": Normal((4, 2))},
    "mx.linalg.solve_triangular(lower, b, stream=mx.cpu)",
    "MLX.solveTriangular(lower, b, stream: .cpu)",
)
case(
    "det",
    "Linalg",
    {"a": SQUARE, "spd": SPD},
    "mx.linalg.det(spd, stream=mx.cpu)",
    "MLX.det(spd, stream: .cpu)",
)
case(
    "eigvalsh",
    "Linalg",
    {"a": SQUARE, "spd": SPD},
    "mx.linalg.eigvalsh(spd, stream=mx.cpu)",
    "MLX.eigvalsh(spd, stream: .cpu)",
    note="symmetric input: eigenvalues are returned in ascending order",
)

# ---------------------------------------------------------------------- fast

case(
    "rmsNorm",
    "Fast",
    {"x": Normal((2, 8, 16)), "weight": Normal((16,))},
    "mx.fast.rms_norm(x, weight, 1e-5)",
    "MLX.rmsNorm(x, weight: weight, eps: 1e-5)",
)
case(
    "layerNorm",
    "Fast",
    {"x": Normal((2, 8, 16)), "weight": Normal((16,)), "bias": Normal((16,))},
    "mx.fast.layer_norm(x, weight, bias, 1e-5)",
    "MLX.layerNorm(x, weight: weight, bias: bias, eps: 1e-5)",
)
case(
    "layerNorm/noAffine",
    "Fast",
    {"x": Normal((2, 8, 16))},
    "mx.fast.layer_norm(x, None, None, 1e-5)",
    "MLX.layerNorm(x, eps: 1e-5)",
)
case(
    "RoPE",
    "Fast",
    {"x": Normal((2, 4, 8, 16))},
    "mx.fast.rope(x, 16, traditional=False, base=10000.0, scale=1.0, offset=0)",
    "MLX.RoPE(x, dimensions: 16, traditional: false, base: 10000.0, scale: 1.0, offset: 0)",
)
case(
    "RoPE/traditional",
    "Fast",
    {"x": Normal((2, 4, 8, 16))},
    "mx.fast.rope(x, 8, traditional=True, base=10000.0, scale=0.5, offset=2)",
    "MLX.RoPE(x, dimensions: 8, traditional: true, base: 10000.0, scale: 0.5, offset: 2)",
)
case(
    "scaledDotProductAttention",
    "Fast",
    {
        "queries": Normal((1, 2, 4, 8)),
        "keys": Normal((1, 2, 4, 8)),
        "values": Normal((1, 2, 4, 8)),
    },
    "mx.fast.scaled_dot_product_attention(queries, keys, values, scale=0.35)",
    "MLX.scaledDotProductAttention(queries: queries, keys: keys, values: values, "
    "scale: 0.35, mask: nil)",
)
case(
    "scaledDotProductAttention/mask",
    "Fast",
    {
        "queries": Normal((1, 2, 4, 8)),
        "keys": Normal((1, 2, 4, 8)),
        "values": Normal((1, 2, 4, 8)),
        "mask": Expression(
            py="mx.tril(mx.ones([4, 4])) == 1",
            swift="MLX.tril(MLX.ones([4, 4])) .== 1",
        ),
    },
    "mx.fast.scaled_dot_product_attention(queries, keys, values, scale=0.35, mask=mask)",
    "MLX.scaledDotProductAttention(queries: queries, keys: keys, values: values, "
    "scale: 0.35, mask: mask)",
)

# -------------------------------------------------------------------- random

case("gumbel", "Random", {}, "mx.random.gumbel([4, 3])", "MLXRandom.gumbel([4, 3])")
case(
    "gumbel/dtype",
    "Random",
    {},
    "mx.random.gumbel([4, 3], dtype=mx.float16)",
    "MLXRandom.gumbel([4, 3], dtype: .float16)",
)
case(
    "laplace",
    "Random",
    {},
    "mx.random.laplace([4, 3])",
    "MLXRandom.laplace([4, 3], dtype: .float32)",
    note="the Swift dtype: parameter has no default",
)
case(
    "laplace/locScale",
    "Random",
    {},
    "mx.random.laplace([4, 3], loc=1.0, scale=2.0)",
    "MLXRandom.laplace([4, 3], dtype: .float32, loc: 1.0, scale: 2.0)",
)
case(
    "truncatedNormal",
    "Random",
    {},
    "mx.random.truncated_normal(-1.0, 1.0, [4, 3])",
    "MLXRandom.truncatedNormal(low: -1.0, high: 1.0, [4, 3])",
)
case(
    "categorical",
    "Random",
    {"logits": Normal((4, 5))},
    "mx.random.categorical(logits)",
    "MLXRandom.categorical(logits)",
)
case(
    "categorical/count",
    "Random",
    {"logits": Normal((4, 5))},
    "mx.random.categorical(logits, num_samples=3)",
    "MLXRandom.categorical(logits, count: 3)",
)
case(
    "permutation/count",
    "Random",
    {},
    "mx.random.permutation(10)",
    "MLXRandom.permutation(10)",
)
case(
    "permutation/array",
    "Random",
    {"a": Normal((6, 2))},
    "mx.random.permutation(a)",
    "MLXRandom.permutation(a)",
)
case(
    "multivariateNormal",
    "Random",
    {
        "mean": Expression(py="mx.zeros([3])", swift="MLX.zeros([3])"),
        "covariance": Expression(py="mx.eye(3) * 2.0", swift="MLX.eye(3) * 2.0"),
    },
    "mx.random.multivariate_normal(mean, covariance, [4], stream=mx.cpu)",
    "MLXRandom.multivariateNormal(mean: mean, covariance: covariance, shape: [4], "
    "dtype: .float32, stream: .cpu)",
)
case("key", "Random", {}, "mx.random.key(42)", "MLXRandom.key(42)")

# ----------------------------------------------------------------- activations

ACT = Normal((4, 3))


def activation(name: str, py: str, swift: str, spec=None) -> None:
    case(name, "Activations", {"x": spec or ACT}, py, swift)


activation("relu", "nn.relu(x)", "MLXNN.relu(x)")
activation("reluSquared", "nn.relu2(x)", "MLXNN.reluSquared(x)")
activation("relu6", "nn.relu6(x)", "MLXNN.relu6(x)")
activation("leakyRelu", "nn.leaky_relu(x)", "MLXNN.leakyRelu(x)")
activation(
    "leakyRelu/slope",
    "nn.leaky_relu(x, negative_slope=0.2)",
    "MLXNN.leakyRelu(x, negativeSlope: 0.2)",
)
activation("elu", "nn.elu(x)", "MLXNN.elu(x)")
activation("elu/alpha", "nn.elu(x, alpha=0.5)", "MLXNN.elu(x, alpha: 0.5)")
activation("celu", "nn.celu(x)", "MLXNN.celu(x)")
activation("celu/alpha", "nn.celu(x, alpha=0.5)", "MLXNN.celu(x, alpha: 0.5)")
activation("silu", "nn.silu(x)", "MLXNN.silu(x)")
activation("mish", "nn.mish(x)", "MLXNN.mish(x)")
activation("selu", "nn.selu(x)", "MLXNN.selu(x)")
activation("softplus", "nn.softplus(x)", "MLXNN.softplus(x)")
activation("softsign", "nn.softsign(x)", "MLXNN.softsign(x)")
activation("softshrink", "nn.softshrink(x)", "MLXNN.softshrink(x)")
activation(
    "softshrink/lambda",
    "nn.softshrink(x, lambd=0.2)",
    "MLXNN.softshrink(x, lambda: 0.2)",
)
activation("softmin", "nn.softmin(x)", "MLXNN.softmin(x)")
activation("softmin/axis", "nn.softmin(x, axis=0)", "MLXNN.softmin(x, axis: 0)")
activation("logSoftmax", "nn.log_softmax(x)", "MLXNN.logSoftmax(x)")
activation(
    "logSoftmax/axis", "nn.log_softmax(x, axis=0)", "MLXNN.logSoftmax(x, axis: 0)"
)
activation("logSigmoid", "nn.log_sigmoid(x)", "MLXNN.logSigmoid(x)")
activation("gelu", "nn.gelu(x)", "MLXNN.gelu(x)")
activation("geluApproximate", "nn.gelu_approx(x)", "MLXNN.geluApproximate(x)")
activation(
    "geluFastApproximate", "nn.gelu_fast_approx(x)", "MLXNN.geluFastApproximate(x)"
)
activation("glu", "nn.glu(x)", "MLXNN.glu(x)", spec=Normal((4, 8)))
activation(
    "glu/axis", "nn.glu(x, axis=0)", "MLXNN.glu(x, axis: 0)", spec=Normal((4, 8))
)
activation("step", "nn.step(x)", "MLXNN.step(x)")
activation(
    "step/threshold", "nn.step(x, threshold=0.5)", "MLXNN.step(x, threshold: 0.5)"
)
activation("hardSwish", "nn.hardswish(x)", "MLXNN.hardSwish(x)")
activation("hardTanH", "nn.hard_tanh(x)", "MLXNN.hardTanH(x)")
activation(
    "hardTanH/range",
    "nn.hard_tanh(x, min_val=-0.5, max_val=0.5)",
    "MLXNN.hardTanH(x, min: -0.5, max: 0.5)",
)
activation("hardShrink", "nn.hard_shrink(x)", "MLXNN.hardShrink(x)")
activation(
    "hardShrink/lambda",
    "nn.hard_shrink(x, lambd=0.2)",
    "MLXNN.hardShrink(x, lambda: 0.2)",
)
case(
    "prelu",
    "Activations",
    {"x": Normal((4, 3)), "alpha": Uniform((3,), low=0.1, high=0.5)},
    "nn.prelu(x, alpha)",
    "MLXNN.prelu(x, alpha: alpha)",
)

# --------------------------------------------------------------------- losses

PREDICTIONS = Normal((4, 3))
TARGETS = Normal((4, 3))
LOGITS = Normal((4, 5))
CLASSES = RandInt(0, 5, (4,))
PROBABILITIES = Expression(py="mx.softmax(t, axis=-1)", swift="MLX.softmax(t, axis: -1)")

for reduction in ("none", "mean", "sum"):
    swift_reduction = f".{reduction}"
    case(
        f"crossEntropy/{reduction}",
        "Losses",
        {"logits": LOGITS, "targets": CLASSES},
        f'nn.losses.cross_entropy(logits, targets, reduction="{reduction}")',
        f"MLXNN.crossEntropy(logits: logits, targets: targets, "
        f"reduction: {swift_reduction})",
    )
    case(
        f"mseLoss/{reduction}",
        "Losses",
        {"predictions": PREDICTIONS, "targets": TARGETS},
        f'nn.losses.mse_loss(predictions, targets, reduction="{reduction}")',
        f"MLXNN.mseLoss(predictions: predictions, targets: targets, "
        f"reduction: {swift_reduction})",
    )
    case(
        f"l1Loss/{reduction}",
        "Losses",
        {"predictions": PREDICTIONS, "targets": TARGETS},
        f'nn.losses.l1_loss(predictions, targets, reduction="{reduction}")',
        f"MLXNN.l1Loss(predictions: predictions, targets: targets, "
        f"reduction: {swift_reduction})",
    )

case(
    "crossEntropy/weights",
    "Losses",
    {"logits": LOGITS, "targets": CLASSES, "weights": Uniform((4,), low=0.5, high=1.5)},
    'nn.losses.cross_entropy(logits, targets, weights=weights, reduction="mean")',
    "MLXNN.crossEntropy(logits: logits, targets: targets, weights: weights, "
    "reduction: .mean)",
)
case(
    "crossEntropy/labelSmoothing",
    "Losses",
    {"logits": LOGITS, "targets": CLASSES},
    'nn.losses.cross_entropy(logits, targets, label_smoothing=0.1, reduction="mean")',
    "MLXNN.crossEntropy(logits: logits, targets: targets, labelSmoothing: 0.1, "
    "reduction: .mean)",
)
case(
    "crossEntropy/probabilities",
    "Losses",
    {"logits": LOGITS, "t": Normal((4, 5)), "targets": PROBABILITIES},
    'nn.losses.cross_entropy(logits, targets, reduction="mean")',
    "MLXNN.crossEntropy(logits: logits, targets: targets, reduction: .mean)",
)
case(
    "binaryCrossEntropy/logits",
    "Losses",
    {"logits": Normal((4, 3)), "targets": Bernoulli((4, 3))},
    'nn.losses.binary_cross_entropy(logits, targets.astype(mx.float32), reduction="mean")',
    "MLXNN.binaryCrossEntropy(logits: logits, targets: targets.asType(.float32), "
    "reduction: .mean)",
)
case(
    "binaryCrossEntropy/probabilities",
    "Losses",
    {
        "p": Uniform((4, 3), low=0.1, high=0.9),
        "targets": Bernoulli((4, 3)),
    },
    'nn.losses.binary_cross_entropy(p, targets.astype(mx.float32), with_logits=False, '
    'reduction="mean")',
    "MLXNN.binaryCrossEntropy(logits: p, targets: targets.asType(.float32), "
    "withLogits: false, reduction: .mean)",
)
case(
    "nllLoss",
    "Losses",
    {
        "logits": Normal((4, 5)),
        "inputs": Expression(
            py="mx.log(mx.softmax(logits, axis=-1))",
            swift="MLX.log(MLX.softmax(logits, axis: -1))",
        ),
        "targets": CLASSES,
    },
    'nn.losses.nll_loss(inputs, targets, reduction="mean")',
    "MLXNN.nllLoss(inputs: inputs, targets: targets, reduction: .mean)",
)
case(
    "klDivLoss",
    "Losses",
    {
        "a": Normal((4, 5)),
        "b": Normal((4, 5)),
        "inputs": Expression(
            py="mx.log(mx.softmax(a, axis=-1))",
            swift="MLX.log(MLX.softmax(a, axis: -1))",
        ),
        "targets": Expression(
            py="mx.log(mx.softmax(b, axis=-1))",
            swift="MLX.log(MLX.softmax(b, axis: -1))",
        ),
    },
    'nn.losses.kl_div_loss(inputs, targets, reduction="mean")',
    "MLXNN.klDivLoss(inputs: inputs, targets: targets, reduction: .mean)",
)
case(
    "smoothL1Loss",
    "Losses",
    {"predictions": PREDICTIONS, "targets": TARGETS},
    'nn.losses.smooth_l1_loss(predictions, targets, reduction="mean")',
    "MLXNN.smoothL1Loss(predictions: predictions, targets: targets, reduction: .mean)",
)
case(
    "smoothL1Loss/beta",
    "Losses",
    {"predictions": PREDICTIONS, "targets": TARGETS},
    'nn.losses.smooth_l1_loss(predictions, targets, beta=0.5, reduction="none")',
    "MLXNN.smoothL1Loss(predictions: predictions, targets: targets, beta: 0.5, "
    "reduction: .none)",
)
case(
    "tripletLoss",
    "Losses",
    {
        "anchors": Normal((4, 8)),
        "positives": Normal((4, 8)),
        "negatives": Normal((4, 8)),
    },
    'nn.losses.triplet_loss(anchors, positives, negatives, reduction="mean")',
    "MLXNN.tripletLoss(anchors: anchors, positives: positives, negatives: negatives, "
    "reduction: .mean)",
)
case(
    "hingeLoss",
    "Losses",
    {
        "inputs": Normal((4, 3)),
        "signs": Bernoulli((4, 3)),
        "targets": Expression(
            py="mx.where(signs, 1.0, -1.0)",
            swift="MLX.which(signs, 1.0, -1.0)",
        ),
    },
    'nn.losses.hinge_loss(inputs, targets, reduction="mean")',
    "MLXNN.hingeLoss(inputs: inputs, targets: targets, reduction: .mean)",
)
case(
    "huberLoss",
    "Losses",
    {"inputs": PREDICTIONS, "targets": TARGETS},
    'nn.losses.huber_loss(inputs, targets, delta=0.5, reduction="mean")',
    "MLXNN.huberLoss(inputs: inputs, targets: targets, delta: 0.5, reduction: .mean)",
)
case(
    "logCoshLoss",
    "Losses",
    {"inputs": PREDICTIONS, "targets": TARGETS},
    'nn.losses.log_cosh_loss(inputs, targets, reduction="mean")',
    "MLXNN.logCoshLoss(inputs: inputs, targets: targets, reduction: .mean)",
)
case(
    "cosineSimilarityLoss",
    "Losses",
    {"x1": Normal((4, 8)), "x2": Normal((4, 8))},
    'nn.losses.cosine_similarity_loss(x1, x2, reduction="mean")',
    "MLXNN.cosineSimilarityLoss(x1: x1, x2: x2, reduction: .mean)",
)

# files that need more than `import MLX`
EXTRA_IMPORTS = {
    "Activations": ["MLXNN"],
    "Losses": ["MLXNN"],
}

# ------------------------------------------------------------------- defaults
#
# Calls with *no* optional arguments on both sides.  A default that differs
# between python and Swift is a bug (that is how `nanToNum`'s posInf/negInf and
# `tensordot`'s axes were found), so these cases exist to pin the defaults down
# rather than to cover new functions.
#
# One convention to know: where python takes `axis=None` to mean "the flattened
# array", mlx-swift uses a separate overload with no `axis:` at all.  For those
# the honest pairing is python `axis=None` against the bare Swift call, and the
# case is named `.../flat`.

DEFAULT_A = Normal(S)


def defaults(name: str, inputs: dict, py: str, swift: str, **kwargs) -> None:
    case(name, "Defaults", inputs, py, swift, **kwargs)


defaults(
    "nanToNum",
    {"a": SPECIAL},
    "mx.nan_to_num(a)",
    "MLX.nanToNum(a)",
    note="python replaces +/-inf with the dtype max/min unless told otherwise",
)
defaults(
    "tensordot",
    {"a": Normal((2, 3, 4)), "b": Normal((3, 4, 5))},
    "mx.tensordot(a, b)",
    "MLX.tensordot(a, b)",
    note="axes defaults to 2 (as in numpy), not 1",
)
defaults(
    "isClose/within",
    {"a": DEFAULT_A, "b": Expression(py="a + 1e-6", swift="a + 1e-6")},
    "mx.isclose(a, b)",
    "MLX.isClose(a, b)",
    note="1e-6 apart: inside the default rtol 1e-5 / atol 1e-8",
)
defaults(
    "isClose/outside",
    {"a": DEFAULT_A, "b": Expression(py="a + 1e-3", swift="a + 1e-3")},
    "mx.isclose(a, b)",
    "MLX.isClose(a, b)",
)
defaults(
    "allClose",
    {"a": DEFAULT_A, "b": Expression(py="a + 1e-6", swift="a + 1e-6")},
    "mx.allclose(a, b)",
    "MLX.allClose(a, b)",
)
defaults("arrayEqual", {"a": DEFAULT_A}, "mx.array_equal(a, a)", "MLX.arrayEqual(a, a)")
defaults("diff", {"a": DEFAULT_A}, "mx.diff(a)", "MLX.diff(a)")
defaults("trace", {"a": Normal((4, 4))}, "mx.trace(a)", "MLX.trace(a)")
defaults("diag", {"a": Normal((4, 4))}, "mx.diag(a)", "MLX.diag(a)")
defaults("diagonal", {"a": Normal((4, 4))}, "mx.diagonal(a)", "MLX.diagonal(a)")
defaults("flipped", {"a": DEFAULT_A}, "mx.flip(a)", "MLX.flipped(a)")
defaults("round", {"a": Normal(S, scale=10.0)}, "mx.round(a)", "MLX.round(a)")
defaults("softmax", {"a": DEFAULT_A}, "mx.softmax(a)", "MLX.softmax(a)")
defaults("cumsum", {"a": DEFAULT_A}, "mx.cumsum(a)", "MLX.cumsum(a)")
defaults("cumprod", {"a": DEFAULT_A}, "mx.cumprod(a)", "MLX.cumprod(a)")
defaults("cummax", {"a": DEFAULT_A}, "mx.cummax(a)", "MLX.cummax(a)")
defaults("cummin", {"a": DEFAULT_A}, "mx.cummin(a)", "MLX.cummin(a)")
defaults(
    "logCumsumExp", {"a": DEFAULT_A}, "mx.logcumsumexp(a)", "MLX.logCumsumExp(a)"
)
defaults("median", {"a": DEFAULT_A}, "mx.median(a)", "MLX.median(a)")
defaults("variance", {"a": DEFAULT_A}, "mx.var(a)", "MLX.variance(a)")
defaults("std", {"a": DEFAULT_A}, "mx.std(a)", "MLX.std(a)")
defaults("hadamardTransform", {"a": Normal((4, 16))}, "mx.hadamard_transform(a)", "MLX.hadamardTransform(a)")
defaults("linspace", {}, "mx.linspace(0.0, 1.0)", "MLX.linspace(0.0, 1.0)")
defaults("eye", {}, "mx.eye(4)", "MLX.eye(4)")
defaults(
    "tri",
    {},
    "mx.tri(4, 4, 0)",
    "MLX.tri(4)",
    note="python requires m and k; Swift defaults them to n and 0",
)
defaults("convolve", {"a": Normal((20,)), "v": Normal((4,))}, "mx.convolve(a, v)", "MLX.convolve(a, v)")
defaults(
    "conv1d",
    {"input": Normal((2, 10, 4)), "weight": Normal((3, 3, 4))},
    "mx.conv1d(input, weight)",
    "MLX.conv1d(input, weight)",
)
defaults(
    "quantized",
    {"w": Uniform((64, 128), low=-1.0, high=1.0)},
    "mx.quantize(w)[0]",
    "MLX.quantized(w).wq",
    note="group size and bit count default in the C++ core (64 / 4)",
)
defaults(
    "dequantized",
    {
        "w": Uniform((64, 128), low=-1.0, high=1.0),
        "q": Expression(py="mx.quantize(w)", swift="MLX.quantized(w)", array=False),
    },
    "mx.dequantize(q[0], q[1], q[2])",
    "MLX.dequantized(q.wq, scales: q.scales, biases: q.biases)",
)
defaults(
    "quantizedMM",
    {
        "w": Uniform((64, 128), low=-1.0, high=1.0),
        "x": Normal((8, 128)),
        "q": Expression(py="mx.quantize(w)", swift="MLX.quantized(w)", array=False),
    },
    "mx.quantized_matmul(x, q[0], q[1], q[2])",
    "MLX.quantizedMM(x, q.wq, scales: q.scales, biases: q.biases)",
)
defaults("norm", {"a": DEFAULT_A}, "mx.linalg.norm(a)", "MLX.norm(a, stream: .cpu)")
defaults(
    "cholesky",
    {"a": Normal((4, 4)), "spd": SPD},
    "mx.linalg.cholesky(spd, stream=mx.cpu)",
    "MLX.cholesky(spd, stream: .cpu)",
    note="upper defaults to false on both sides",
)
defaults(
    "triInv",
    {"a": Normal((4, 4)), "spd": SPD, "lower": LOWER},
    "mx.linalg.tri_inv(lower, stream=mx.cpu)",
    "MLX.triInv(lower, stream: .cpu)",
)
defaults(
    "solveTriangular",
    {"a": Normal((4, 4)), "spd": SPD, "lower": LOWER, "b": Normal((4, 2))},
    "mx.linalg.solve_triangular(lower, b, stream=mx.cpu)",
    "MLX.solveTriangular(lower, b, stream: .cpu)",
)
defaults("fft", {"r": Normal((100,)), "i": Normal((100,)), "c": COMPLEX_1D}, "mx.fft.fft(c)", "MLX.fft(c)")
defaults("fftfreq", {}, "mx.fft.fftfreq(16)", "MLX.fftfreq(16)")
defaults("gumbel", {}, "mx.random.gumbel()", "MLXRandom.gumbel()")
defaults("uniform", {}, "mx.random.uniform()", "MLXRandom.uniform()")
defaults("normal", {}, "mx.random.normal()", "MLXRandom.normal()")
defaults("bernoulli", {}, "mx.random.bernoulli()", "MLXRandom.bernoulli()")

# python `axis=None` vs the bare Swift overload
defaults("sorted/flat", {"a": DEFAULT_A}, "mx.sort(a, axis=None)", "MLX.sorted(a)")
defaults("argSort/flat", {"a": DEFAULT_A}, "mx.argsort(a, axis=None)", "MLX.argSort(a)")
defaults(
    "partitioned/flat",
    {"a": DEFAULT_A},
    "mx.partition(a, 3, axis=None)",
    "MLX.partitioned(a, kth: 3)",
)
defaults(
    "argPartition/flat",
    {"a": DEFAULT_A},
    "mx.argpartition(a, 3, axis=None)",
    "MLX.argPartition(a, kth: 3)",
)
defaults("top/flat", {"a": DEFAULT_A}, "mx.topk(a, 3, axis=None)", "MLX.top(a, k: 3)")
defaults("roll/flat", {"a": DEFAULT_A}, "mx.roll(a, 2)", "MLX.roll(a, shift: 2)")
defaults(
    "takeAlong/flat",
    {"a": DEFAULT_A, "indices": RandInt(0, 12, (4,))},
    "mx.take_along_axis(a, indices)",
    "MLX.takeAlong(a, indices)",
)
defaults("repeated/flat", {"a": DEFAULT_A}, "mx.repeat(a, 2)", "MLX.repeated(a, count: 2)")

# loss reduction defaults differ per function (`none` vs `mean`)
defaults(
    "crossEntropy",
    {"logits": Normal((4, 5)), "targets": RandInt(0, 5, (4,))},
    "nn.losses.cross_entropy(logits, targets)",
    "MLXNN.crossEntropy(logits: logits, targets: targets)",
)
defaults(
    "binaryCrossEntropy",
    {"logits": Normal((4, 3)), "targets": Bernoulli((4, 3))},
    "nn.losses.binary_cross_entropy(logits, targets.astype(mx.float32))",
    "MLXNN.binaryCrossEntropy(logits: logits, targets: targets.asType(.float32))",
)
defaults(
    "l1Loss",
    {"predictions": Normal((4, 3)), "targets": Normal((4, 3))},
    "nn.losses.l1_loss(predictions, targets)",
    "MLXNN.l1Loss(predictions: predictions, targets: targets)",
)
defaults(
    "mseLoss",
    {"predictions": Normal((4, 3)), "targets": Normal((4, 3))},
    "nn.losses.mse_loss(predictions, targets)",
    "MLXNN.mseLoss(predictions: predictions, targets: targets)",
)
defaults(
    "smoothL1Loss",
    {"predictions": Normal((4, 3)), "targets": Normal((4, 3))},
    "nn.losses.smooth_l1_loss(predictions, targets)",
    "MLXNN.smoothL1Loss(predictions: predictions, targets: targets)",
)
defaults(
    "huberLoss",
    {"inputs": Normal((4, 3)), "targets": Normal((4, 3))},
    "nn.losses.huber_loss(inputs, targets)",
    "MLXNN.huberLoss(inputs: inputs, targets: targets)",
)
defaults(
    "logCoshLoss",
    {"inputs": Normal((4, 3)), "targets": Normal((4, 3))},
    "nn.losses.log_cosh_loss(inputs, targets)",
    "MLXNN.logCoshLoss(inputs: inputs, targets: targets)",
)
defaults(
    "hingeLoss",
    {
        "inputs": Normal((4, 3)),
        "signs": Bernoulli((4, 3)),
        "targets": Expression(py="mx.where(signs, 1.0, -1.0)", swift="MLX.which(signs, 1.0, -1.0)"),
    },
    "nn.losses.hinge_loss(inputs, targets)",
    "MLXNN.hingeLoss(inputs: inputs, targets: targets)",
)
defaults(
    "nllLoss",
    {
        "logits": Normal((4, 5)),
        "inputs": Expression(
            py="mx.log(mx.softmax(logits, axis=-1))",
            swift="MLX.log(MLX.softmax(logits, axis: -1))",
        ),
        "targets": RandInt(0, 5, (4,)),
    },
    "nn.losses.nll_loss(inputs, targets)",
    "MLXNN.nllLoss(inputs: inputs, targets: targets)",
)
defaults(
    "tripletLoss",
    {"anchors": Normal((4, 8)), "positives": Normal((4, 8)), "negatives": Normal((4, 8))},
    "nn.losses.triplet_loss(anchors, positives, negatives)",
    "MLXNN.tripletLoss(anchors: anchors, positives: positives, negatives: negatives)",
)
defaults(
    "cosineSimilarityLoss",
    {"x1": Normal((4, 8)), "x2": Normal((4, 8))},
    "nn.losses.cosine_similarity_loss(x1, x2)",
    "MLXNN.cosineSimilarityLoss(x1: x1, x2: x2)",
)

EXTRA_IMPORTS["Defaults"] = ["MLXNN"]
