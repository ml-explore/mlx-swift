# Copyright © 2024 Apple Inc.

import typing as t
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

defaultShape = [4, 3]
emittedTests = {}

random.seed(0)


def new_seed() -> int:
    return random.randint(0, 1000)


def assert_equal(indent, lhs, rhs, accuracy=None) -> str:
    if accuracy is None:
        return f'{" " * indent}XCTAssertEqual({lhs}, {rhs})\n'
    else:
        return f'{" " * indent}XCTAssertEqual({lhs}, {rhs},\n{" " * (indent + 4)}accuracy: {accuracy})\n'


def tuple_to_swift_array(t) -> str:
    tuple_str = ", ".join([str(i) for i in t])
    return f"[{tuple_str}]"


def test_name(name) -> str:
    name = name.replace(".", "")
    name = "test" + name[:1].capitalize() + name[1:]
    if name in emittedTests:
        count = emittedTests[name]
        count += 1
        emittedTests[name] = count
        name += str(count)
    else:
        emittedTests[name] = 0
    return name


def verify_array(indent, name: str, array: mx.array) -> str:
    result = ""

    shape = array.shape
    result += assert_equal(indent, f"{name}.shape", tuple_to_swift_array(shape))

    dtype = array.dtype
    result += assert_equal(indent, f"{name}.dtype", "." + str(dtype).split(".")[-1])

    if dtype == mx.bool_:
        all = mx.all(array).item()
        result += assert_equal(
            indent, f"{name}.all().item()", "true" if all else "false"
        )

        any = mx.any(array).item()
        result += assert_equal(
            indent, f"{name}.any().item()", "true" if any else "false"
        )

    else:
        mean = mx.mean(array).item()
        result += assert_equal(
            indent, f"{name}.mean().item(Float.self)", mean, mean * 0.02
        )

        sum = mx.sum(array).item()
        result += assert_equal(
            indent, f"{name}.sum().item(Float.self)", sum, sum * 0.02
        )

    return result


def create_argument(indent, name, value) -> t.Tuple[str, mx.array]:
    if value is None:
        return (f"let {name} = MLXRandom.normal([4, 3])", mx.random.normal([4, 3]))
        
    if value == "scalar":
        return (f"let {name} = MLXRandom.normal()", mx.random.normal())

    if isinstance(value, t.Tuple):
        return (
            f"let {name} = MLXRandom.uniform(0.0 ..< 1.0, {tuple_to_swift_array(value)})",
            mx.random.uniform(0, 1, value),
        )

    if isinstance(value, int) or isinstance(value, float):
        return (f"let {name} = {value}", value)

    if isinstance(value, dict) and "low" in value:
        return (
            f"let {name} = MLXRandom.uniform(low: {value['low']}, high: {value['high']}, [4, 3])",
            mx.random.uniform(value["low"], value["high"], [4, 3]),
        )

    if isinstance(value, dict) and "int" in value:
        return (
            f"let {name} = MLXRandom.randInt(low: 0, high: 10, {tuple_to_swift_array(value['shape'])})",
            mx.random.randint(0, 10, value["shape"]),
        )


def test_operator(
    name: str, op: str, *, swift_name: str = None, lhs=None, rhs=None
) -> str:
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)
    (rhs_decl, rhs) = create_argument(indent, "b", rhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    result += (" " * indent) + rhs_decl + "\n"
    if isinstance(rhs, mx.array):
        result += verify_array(indent, "b", rhs)

    result += (" " * indent) + f"let result = a {swift_name or op} b\n"

    c = eval(f"lhs {op} rhs")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def test_array_function1(
    name: str,
    function_name: str,
    extra="",
    *,
    swift_name: str = None,
    swift_extra="",
    lhs=None,
) -> str:
    # function with 1 array args (self)
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    result += (
        " " * indent
    ) + f"let result = a.{swift_name or function_name}({swift_extra})\n"

    c = eval(f"lhs.{function_name}({extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def test_free_function1(
    name: str,
    function_name: str,
    extra="",
    *,
    swift_name: str = None,
    swift_extra="",
    lhs=None,
) -> str:
    # free function with 1 array arg
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    sep = ", " if len(swift_extra) != 0 else ""
    result += (
        " " * indent
    ) + f"let result = {swift_name or function_name}(a{sep}{swift_extra})\n"

    c = eval(f"mx.{function_name}(lhs{sep}{extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def test_array_function2(
    name: str,
    function_name: str,
    extra="",
    *,
    swift_name: str = None,
    swift_extra="",
    lhs=None,
    rhs=None,
) -> str:
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)
    (rhs_decl, rhs) = create_argument(indent, "b", rhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    result += (" " * indent) + rhs_decl + "\n"
    if isinstance(rhs, mx.array):
        result += verify_array(indent, "b", rhs)

    sep = ", " if len(swift_extra) != 0 else ""
    result += (
        " " * indent
    ) + f"let result = a.{swift_name or function_name}(b{sep}{swift_extra})\n"

    c = eval(f"lhs.{function_name}(rhs{sep}{extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def test_free_function2(
    name: str,
    function_name: str,
    extra="",
    *,
    swift_name: str = None,
    swift_extra="",
    lhs=None,
    rhs=None,
) -> str:
    # free function with 2 array args
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (lhs_decl, lhs) = create_argument(indent, "a", lhs)
    (rhs_decl, rhs) = create_argument(indent, "b", rhs)

    result += (" " * indent) + lhs_decl + "\n"
    if isinstance(lhs, mx.array):
        result += verify_array(indent, "a", lhs)

    result += (" " * indent) + rhs_decl + "\n"
    if isinstance(rhs, mx.array):
        result += verify_array(indent, "b", rhs)

    sep = ", " if len(swift_extra) != 0 else ""
    result += (
        " " * indent
    ) + f"let result = {swift_name or function_name}(a, b{sep}{swift_extra})\n"

    c = eval(f"mx.{function_name}(lhs, rhs{sep}{extra})")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def adhoc_preamble(name: str):
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})"
    mx.random.seed(seed)

    print(result)


def adhoc_indent(line: str):
    indent = 8
    print((" " * indent) + line)


def adhoc_postamble():
    result = ""
    indent = 4
    result += (" " * indent) + "}\n"
    print(result)


def test_fft(
    function_name: str, n=None, s=None, axis=None, axes=None, *, value=None
) -> str:
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(function_name + "_") + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (r_decl, r) = create_argument(indent, "r", value)
    (i_decl, i) = create_argument(indent, "i", value)

    result += (" " * indent) + r_decl + "\n"
    if isinstance(r, mx.array):
        result += verify_array(indent, "r", r)

    result += (" " * indent) + i_decl + "\n"
    if isinstance(i, mx.array):
        result += verify_array(indent, "i", i)

    # combine into a complex array
    result += (" " * indent) + f"let c = r + i.asImaginary()\n"
    c = r + 1j * i

    e = f"mx.fft.{function_name}(c"
    result += (" " * indent) + f"let result = {function_name}(c"
    if n is not None:
        result += f", n: {n}"
        e += f", n=n"
    if s is not None:
        result += f", s: {s}"
        e += f", s=s"
    if axis is not None:
        result += f", axis: {axis}"
        e += f", axis=axis"
    if axes is not None:
        result += f", axes: {tuple_to_swift_array(axes)}"
        e += f", axes=axes"
    result += ", stream: .cpu)\n"
    e += ", stream=mx.cpu)"

    c = eval(e)

    if c.dtype == mx.complex64:
        # split back out real and imaginary
        result += (" " * indent) + f"let resultReal = result.realPart()\n"
        result += (" " * indent) + f"let resultImaginary = result.imaginaryPart()\n"

        r = c.astype(mx.float32)
        i = (c / (1j)).astype(mx.float32)

        result += verify_array(indent, "resultReal", r)
        result += verify_array(indent, "resultImaginary", i)
    else:
        result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def test_optimizer(
    name: str, *, params=None, swift_params=None, swift_name: str = None, value=None
) -> str:
    # free function with 1 array arg
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (a_decl, a) = create_argument(indent, "a", value)
    (a_grad_decl, a_grad) = create_argument(indent, "aGrad", value)

    result += (" " * indent) + a_decl + "\n"
    if isinstance(a, mx.array):
        result += verify_array(indent, "a", a)

    result += (" " * indent) + a_grad_decl + "\n"
    if isinstance(a_grad, mx.array):
        result += verify_array(indent, "aGrad", a_grad)

    result += (
        " " * indent
    ) + f'let aModel = ModuleParameters(values: ["a": .value(a)])\n'
    result += (
        " " * indent
    ) + f'let aGradParams = ModuleParameters(values: ["a": .value(aGrad)])\n'
    result += (
        (" " * indent)
        + f"let result = {swift_name or name}(learningRate: 0.1{swift_params or ''}).apply(gradients: aGradParams, modelParameters: aModel)\n"
    )

    model = {"a": a}
    gradient = {"a": a_grad}

    c = eval(
        f"optim.{name}(learning_rate=0.1{params or ''}).apply_gradients(gradient, model)"
    )

    result += verify_array(indent, 'result[unwrapping: "a"]!', c["a"])

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


def test_unary_layer(
    name: str,
    *,
    params=None,
    swift_params=None,
    swift_name: str = None,
    test_subset=None,
    value=None,
) -> str:
    # free function with 1 array arg
    result = ""
    indent = 4
    result += (" " * indent) + "func " + test_name(name) + "() {\n"

    seed = new_seed()
    indent += 4
    result += (" " * indent) + f"MLXRandom.seed({seed})\n"
    mx.random.seed(seed)

    (a_decl, a) = create_argument(indent, "a", value)

    result += (" " * indent) + a_decl + "\n"
    if isinstance(a, mx.array):
        result += verify_array(indent, "a", a)

    # some of the layers, e.g. InstanceNorm produce a result that sums to near zero
    # and this is difficult to compare.  if test_subset is True then only look at
    # part of the result

    swift_subset = ""
    python_subset = ""
    if test_subset == "column":
        swift_subset = "[.ellipsis, 0]"
        python_subset = "[..., 0]"
    elif test_subset == "0, 0":
        swift_subset = "[0, 0]"
        python_subset = "[0, 0]"

    result += (
        (" " * indent)
        + f"let result = {swift_name or name}({swift_params or ''})(a){swift_subset}\n"
    )

    c = eval(f"nn.{name}({params or ''})(a){python_subset}")

    result += verify_array(indent, "result", c)

    indent -= 4
    result += (" " * indent) + "}\n"

    return result


print(
    """
// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
@testable import MLXOptimizers
import XCTest

/// Integration tests comparing results vs known results from python
/// integration.  Generated by `tools/generate_integration_tests.py`.
///
/// Note: this is not meant to be complete coverage, merely a sanity
/// check that the wrapping of the c++ core matches python (e.g. calls
/// the same functions).
class MLXIntegrationTests: XCTestCase {
"""
)

# test for random seed

adhoc_preamble("randomSeed")
(r_decl, r) = create_argument(8, "r", "scalar")
print((" " * 8) + r_decl)
print(assert_equal(8, "r.item(Float.self)", r.item(), 0.001))
adhoc_postamble()


# generate tests for operators

arithmeticOps = [
    ("addOp", "+"),
    ("subOp", "-"),
    ("mulOp", "*"),
    ("divOp", "/"),
    ("modOp", "%"),
]

for name, op in arithmeticOps:
    print(test_operator(name, op))
    print(test_operator(name, op, lhs=0.5))
    print(test_operator(name, op, rhs=1.3))

# ** doesn't support a float lhs and requires a particular range
print(
    test_operator(
        "powOp", "**", lhs=dict(low=0.1, high=2.0), rhs=dict(low=0.1, high=2.0)
    )
)
print(test_operator("powOp", "**", lhs=dict(low=0.1, high=2.0), rhs=1.3))

logicalOps = [
    ("equalOp", "==", ".=="),
    ("notEqualOp", "!=", ".!="),
    ("lessThanOp", "<", ".<"),
    ("lessThanEqualOp", "<=", ".<="),
    ("greaterThanOp", ">", ".>"),
    ("greaterThanEqualOp", ">=", ".>="),
]

for name, op, swiftop in logicalOps:
    print(test_operator(name, op, swift_name=swiftop))
    print(test_operator(name, op, swift_name=swiftop, rhs=1.3))


# generate tests for single array functions

array_only_functions = [
    # these have MLXArray functions (in some cases python does not)
    dict(name="abs"),
    dict(name="all", axis=True, axes=True),
    dict(name="any", axis=True, axes=True),
    dict(name="argmax", swift_name="argMax", axis=True),
    dict(name="argmin", swift_name="argMin", axis=True),
    dict(name="cummax", axis=True),
    dict(name="cummin", axis=True),
    dict(name="cumprod", axis=True),
    dict(name="cumsum", axis=True),
    dict(
        name="expand_dims",
        swift_name="expandedDimensions",
        axis=True,
        axes=True,
        no_bare=True,
        free_only=True,
    ),
    dict(name="floor", free_only=True),
    dict(name="log", lhs=dict(low=0.1, high=2.0)),
    dict(name="log2", lhs=dict(low=0.1, high=2.0)),
    dict(name="log10", lhs=dict(low=0.1, high=2.0)),
    dict(name="log1p", lhs=dict(low=0.1, high=2.0)),
    dict(name="logsumexp", swift_name="logSumExp", axis=True, axes=True),
    dict(name="max", axis=True, axes=True),
    dict(name="mean", axis=True, axes=True),
    dict(name="min", axis=True, axes=True),
    dict(name="prod", swift_name="product", axis=True, axes=True),
    dict(name="reciprocal"),
    dict(name="round"),
    dict(name="sin"),
    dict(name="cos"),
    dict(name="sqrt", lhs=dict(low=0.1, high=2.0)),
    dict(name="sum", axis=True, axes=True),
    dict(name="var", swift_name="variance", axis=True, axes=True),
    # free functions only
    dict(name="arccos", swift_name="acos", free_only=True, lhs=dict(low=0.1, high=1.0)),
    dict(name="arccosh", swift_name="acosh", free_only=True, lhs=dict(low=1, high=3)),
    dict(name="arcsin", swift_name="asin", free_only=True, lhs=dict(low=0.1, high=1.0)),
    dict(name="arcsinh", swift_name="asinh", free_only=True, lhs=dict(low=1, high=3)),
    dict(name="arctan", swift_name="atan", free_only=True, lhs=dict(low=0.1, high=1.0)),
    dict(
        name="arctanh", swift_name="atanh", free_only=True, lhs=dict(low=0.1, high=0.9)
    ),
    dict(name="ceil", free_only=True),
    dict(name="cosh", free_only=True),
    dict(name="erf", free_only=True),
    dict(
        name="erfinv",
        swift_name="erfInverse",
        free_only=True,
        lhs=dict(low=0.1, high=0.9),
    ),
    dict(name="logical_not", swift_name="logicalNot", free_only=True),
    dict(name="negative", free_only=True),
    dict(name="sigmoid", free_only=True),
    dict(name="sign", free_only=True),
    dict(name="sinh", free_only=True),
    dict(name="softmax", swift_name="softMax", axis=True, axes=True, free_only=True),
    dict(name="tan", free_only=True),
    dict(name="tanh", free_only=True),
]

for config in array_only_functions:
    function_name = config["name"]
    swift_name = config.get("swift_name", function_name)
    lhs = config.get("lhs", None)

    if "no_bare" not in config:
        if "free_only" not in config:
            print(
                test_array_function1(
                    swift_name, function_name, swift_name=swift_name, lhs=lhs
                )
            )
        print(
            test_free_function1(
                swift_name, function_name, swift_name=swift_name, lhs=lhs
            )
        )

    if "axis" in config:
        if "free_only" not in config:
            print(
                test_array_function1(
                    swift_name,
                    function_name,
                    "axis=-1",
                    swift_name=swift_name,
                    swift_extra="axis: -1",
                    lhs=lhs,
                )
            )
        print(
            test_free_function1(
                swift_name,
                function_name,
                "axis=-1",
                swift_name=swift_name,
                swift_extra="axis: -1",
                lhs=lhs,
            )
        )

    if "axes" in config:
        if "free_only" not in config:
            print(
                test_array_function1(
                    swift_name,
                    function_name,
                    "axis=[0, -1]",
                    swift_name=swift_name,
                    swift_extra="axes: [0, -1]",
                    lhs=(2, 3, 4, 3),
                )
            )
        print(
            test_free_function1(
                swift_name,
                function_name,
                "axis=[0, -1]",
                swift_name=swift_name,
                swift_extra="axes: [0, -1]",
                lhs=(2, 3, 4, 3),
            )
        )

# generate tests for two array functions

two_array_functions = [
    # free functions only
    dict(name="add", swift_name="MLX.add", free_only=True),
    dict(name="conv1d", free_only=True, lhs=(4, 10, 4), rhs=(2, 10, 4)),
    dict(name="conv2d", free_only=True, lhs=(4, 10, 12, 4), rhs=(2, 10, 12, 4)),
    dict(name="convolve", free_only=True, lhs=(20,), rhs=(4,)),
    dict(name="divide", free_only=True),
    dict(name="equal", free_only=True),
    dict(name="greater", free_only=True),
    dict(name="greater_equal", swift_name="greaterEqual", free_only=True),
    dict(name="less", free_only=True),
    dict(name="less_equal", swift_name="lessEqual", free_only=True),
    dict(name="logaddexp", swift_name="logAddExp", free_only=True),
    dict(name="matmul", free_only=True, lhs=(10, 8), rhs=(8, 13)),
    dict(name="maximum", free_only=True),
    dict(name="minimum", free_only=True),
    dict(name="multiply", free_only=True),
    dict(name="not_equal", swift_name="notEqual", free_only=True),
    dict(name="remainder", free_only=True),
    dict(name="subtract", free_only=True),
]

for config in two_array_functions:
    function_name = config["name"]
    swift_name = config.get("swift_name", function_name)
    lhs = config.get("lhs", None)
    rhs = config.get("rhs", None)

    if "no_bare" not in config:
        if "free_only" not in config:
            print(
                test_array_function2(
                    swift_name, function_name, swift_name=swift_name, lhs=lhs, rhs=rhs
                )
            )
        print(
            test_free_function2(
                swift_name, function_name, swift_name=swift_name, lhs=lhs, rhs=rhs
            )
        )

    if "axis" in config:
        if "free_only" not in config:
            print(
                test_array_function2(
                    swift_name,
                    function_name,
                    "axis=-1",
                    swift_name=swift_name,
                    swift_extra="axis: -1",
                    lhs=lhs,
                    rhs=rhs,
                )
            )
        print(
            test_free_function2(
                swift_name,
                function_name,
                "axis=-1",
                swift_name=swift_name,
                swift_extra="axis: -1",
                lhs=lhs,
                rhs=rhs,
            )
        )

    if "axes" in config:
        if "free_only" not in config:
            print(
                test_array_function2(
                    swift_name,
                    function_name,
                    "axis=[0, -1]",
                    swift_name=swift_name,
                    swift_extra="axes: [0, -1]",
                    lhs=(2, 3, 4, 3),
                    rhs=(2, 3, 4, 3),
                )
            )
        print(
            test_free_function2(
                swift_name,
                function_name,
                "axis=[0, -1]",
                swift_name=swift_name,
                swift_extra="axes: [0, -1]",
                lhs=(2, 3, 4, 3),
                rhs=(2, 3, 4, 3),
            )
        )

# misc functions not covered above


adhoc_preamble("quantize")
(w_decl, w) = create_argument(
    8,
    "w",
    (
        32,
        256,
    ),
)
(w_q, scales, biases) = mx.quantize(w, bits=8)
adhoc_indent(w_decl)
adhoc_indent("let (wq, scales, biases) = quantized(w, bits: 8)")
print(verify_array(8, "wq", w_q))
print(verify_array(8, "scales", scales))
print(verify_array(8, "biases", biases))
adhoc_postamble()

# FFTs

fft_functions = [
    ("fft", (100, 100), [dict(n=80), dict(n=120), dict(axis=0)]),
    ("ifft", (100,), [dict(n=80), dict(n=120), dict(axis=0)]),
    ("rfft", (100,), [dict(n=80), dict(n=120), dict(axis=0)]),
    ("irfft", (100,), [dict(n=80), dict(n=120), dict(axis=0)]),
    (
        "fft2",
        (8, 8, 8),
        [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
    ),
    (
        "ifft2",
        (8, 8, 8),
        [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
    ),
    (
        "fftn",
        (8, 8, 8),
        [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
    ),
    (
        "ifftn",
        (8, 8, 8),
        [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
    ),
    (
        "rfft2",
        (8, 8, 8),
        [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
    ),
    (
        "irfft2",
        (8, 8, 8),
        [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
    ),
    (
        "rfftn",
        (8, 8, 8),
        [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
    ),
    (
        "irfftn",
        (8, 8, 8),
        [dict(s=[3, 4]), dict(axes=[0, 2]), dict(s=[10, 5], axes=[2, 1])],
    ),
]

for fft, shape, args_array in fft_functions:
    print(test_fft(fft, value=shape))

    for args in args_array:
        print(test_fft(fft, value=shape, **args))

# optimizers

optimizer = [
    dict(
        name="SGD",
        params=[
            (", momentum=0.1", ", momentum: 0.1"),
            (", momentum=0.1, dampening=0.1", ", momentum: 0.1, dampening: 0.1"),
        ],
    ),
    dict(name="RMSprop"),
    dict(name="Adagrad", swift_name="AdaGrad"),
    dict(name="AdaDelta"),
    dict(name="Adam"),
    dict(name="AdamW"),
    dict(name="Adamax"),
    dict(
        name="Lion",
        params=[
            (", weight_decay=0.1", ", weightDecay: 0.1"),
        ],
    ),
    dict(
        name="Adafactor",
        params=[
            (", beta_1=0.1", ", beta1: 0.1"),
        ],
    ),
]

for o in optimizer:
    print(test_optimizer(o["name"], swift_name=o.get("swift_name", None)))
    if "params" in o:
        for p, s in o["params"]:
            print(
                test_optimizer(
                    o["name"],
                    params=p,
                    swift_params=s,
                    swift_name=o.get("swift_name", None),
                )
            )

# AdaFactor has different behavior for different shapes
print(test_optimizer("Adafactor", value=(10,)))

# layers

unary_layers = [
    # attention layers
    dict(name="GLU"),
    dict(name="Sigmoid"),
    dict(name="Mish"),
    dict(name="ReLU"),
    dict(name="LeakyReLU"),
    dict(name="ReLU6"),
    dict(name="Softmax", swift_name="SoftMax"),
    dict(name="Softplus", swift_name="SoftPlus"),
    dict(name="Softsign", swift_name="SoftSign"),
    dict(name="CELU"),
    dict(name="SiLU"),
    dict(name="LogSoftmax", swift_name="LogSoftMax"),
    dict(name="LogSigmoid"),
    dict(name="PReLU"),
    dict(name="GELU"),
    dict(name="Tanh"),
    dict(name="Hardswish", swift_name="HardSwish"),
    dict(name="Step"),
    dict(name="SELU"),
    dict(
        name="Linear",
        params_required=True,
        params=[
            ("16, 5", "16, 5"),
        ],
    ),
    dict(
        name="Conv1d",
        params_required=True,
        params=[
            (
                "in_channels=16, out_channels=2, kernel_size=8",
                "inputChannels: 16, outputChannels: 2, kernelSize: 8",
            ),
        ],
    ),
    dict(
        name="Conv2d",
        params_required=True,
        value=(2, 8, 8, 4),
        params=[
            (
                "in_channels=4, out_channels=2, kernel_size=8",
                "inputChannels: 4, outputChannels: 2, kernelSize: 8",
            ),
        ],
    ),
    dict(name="Dropout"),
    dict(name="Dropout2d"),
    dict(name="Dropout3d", value=(2, 8, 8, 4)),
    dict(
        name="Embedding",
        params_required=True,
        value=dict(int=True, shape=(2, 8, 8, 4)),
        params=[
            # note: the range of the inputs is integers from [0..10) -- num_embeddings
            # must be at least that size
            ("num_embeddings=10, dims=8", "embeddingCount: 10, dimensions: 8"),
        ],
    ),
    # normalization
    dict(
        name="InstanceNorm",
        params_required=True,
        test_subset="0, 0",
        params=[
            ("8", "dimensions: 8"),
        ],
    ),
    dict(
        name="LayerNorm",
        params_required=True,
        test_subset="column",
        params=[
            ("16", "dimensions: 16"),
        ],
    ),
    dict(
        name="RMSNorm",
        params_required=True,
        params=[
            ("16", "dimensions: 16"),
        ],
    ),
    dict(
        name="GroupNorm",
        params_required=True,
        test_subset="0, 0",
        params=[
            ("4, 16", "groupCount: 4, dimensions: 16"),
        ],
    ),
    dict(
        name="BatchNorm",
        params_required=True,
        test_subset="0, 0",
        params=[
            ("16", "featureCount: 16"),
        ],
    ),
    # positional encoding
    dict(
        name="RoPE",
        params_required=True,
        params=[
            ("8", "dimensions: 8"),
        ],
    ),
    dict(
        name="SinusoidalPositionalEncoding",
        params_required=True,
        params=[
            ("8", "dimensions: 8"),
        ],
    ),
]

for l in unary_layers:
    value = l.get("value", (2, 8, 16))
    test_subset = l.get("test_subset", None)
    if "params_required" not in l:
        print(
            test_unary_layer(
                l["name"],
                swift_name=l.get("swift_name", None),
                test_subset=test_subset,
                value=value,
            )
        )

    if "params" in l:
        for p, s in l["params"]:
            print(
                test_unary_layer(
                    l["name"],
                    params=p,
                    swift_params=s,
                    swift_name=l.get("swift_name", None),
                    test_subset=test_subset,
                    value=value,
                )
            )

print("}\n")
