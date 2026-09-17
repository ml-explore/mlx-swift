# Copyright © 2026 Apple Inc.

"""
Case tables for `MLXNN.Module` integration tests.

Each `module_case` builds a layer in both languages, replaces **every** parameter
with a deterministic function of its shape, asserts the parameter *names* match
python's, then compares the output of one forward pass.

Why the parameter rewrite: python `nn.Linear(16, 5)` and Swift `Linear(16, 5)`
both initialize from the random state, and the old generator relied on both
drawing the same values in the same order.  That holds by luck for a few layers
and not at all for others.  Replacing the parameters removes the dependency
entirely -- and checking the parameter keys catches the thing that actually
breaks users (mlx-swift loads python checkpoints by these names).

Only layers with a **single** output are here; `LSTM` (returns hidden and cell)
belongs to the multiple-return generator.
"""

from core import Bernoulli, Expression, ModuleCase, Normal, RandInt, Uniform

MODULE_CASES: list[ModuleCase] = []


def module_case(name: str, file: str, py: str, swift: str, **kwargs) -> None:
    MODULE_CASES.append(ModuleCase(name=name, file=file, py=py, swift=swift, **kwargs))


# a 3D input works for the elementwise layers, Linear and the normalizations
X3 = Normal((2, 8, 16))

# ------------------------------------------------------------ activation layers

# Swift's `HardTanh(min:max:)`, `Softmax(axis:)`, `LogSoftmax(axis:)` and
# `Softmin(axis:)` take arguments that the python *modules* do not (python only
# exposes them on the free functions), so only the default construction can be
# compared here; the arguments are covered by the `Activations` function cases.
ACTIVATION_LAYERS = [
    # (name, python, swift)
    ("Identity", "nn.Identity()", "Identity()"),
    ("Sigmoid", "nn.Sigmoid()", "Sigmoid()"),
    ("ReLU", "nn.ReLU()", "ReLU()"),
    ("ReLU6", "nn.ReLU6()", "ReLU6()"),
    ("ReLUSquared", "nn.ReLU2()", "ReLUSquared()"),
    ("LeakyReLU", "nn.LeakyReLU()", "LeakyReLU()"),
    ("LeakyReLU/slope", "nn.LeakyReLU(negative_slope=0.2)", "LeakyReLU(negativeSlope: 0.2)"),
    ("ELU", "nn.ELU()", "ELU()"),
    ("ELU/alpha", "nn.ELU(alpha=0.5)", "ELU(alpha: 0.5)"),
    ("CELU", "nn.CELU()", "CELU()"),
    ("SiLU", "nn.SiLU()", "SiLU()"),
    ("SELU", "nn.SELU()", "SELU()"),
    ("Mish", "nn.Mish()", "Mish()"),
    ("Tanh", "nn.Tanh()", "Tanh()"),
    ("GELU", "nn.GELU()", "GELU()"),
    ("GELU/precise", 'nn.GELU(approx="precise")', "GELU(approximation: .precise)"),
    ("GELU/fast", 'nn.GELU(approx="fast")', "GELU(approximation: .fast)"),
    ("HardSwish", "nn.Hardswish()", "HardSwish()"),
    ("HardTanh", "nn.HardTanh()", "HardTanh()"),
    ("HardShrink", "nn.HardShrink()", "HardShrink()"),
    ("HardShrink/lambda", "nn.HardShrink(lambd=0.2)", "HardShrink(lambda: 0.2)"),
    ("Softplus", "nn.Softplus()", "Softplus()"),
    ("Softsign", "nn.Softsign()", "Softsign()"),
    ("Softshrink", "nn.Softshrink()", "Softshrink()"),
    ("Softshrink/lambda", "nn.Softshrink(lambd=0.2)", "Softshrink(lambda: 0.2)"),
    ("Softmax", "nn.Softmax()", "Softmax()"),
    ("Softmin", "nn.Softmin()", "Softmin()"),
    ("LogSoftmax", "nn.LogSoftmax()", "LogSoftmax()"),
    ("LogSigmoid", "nn.LogSigmoid()", "LogSigmoid()"),
    ("Step", "nn.Step()", "Step()"),
    ("Step/threshold", "nn.Step(threshold=0.5)", "Step(threshold: 0.5)"),
    ("GLU", "nn.GLU()", "GLU()"),
]

for name, py, swift in ACTIVATION_LAYERS:
    module_case(name, "ModuleActivations", py, swift, inputs={"x": X3})

# PReLU has a parameter, so it also exercises the parameter rewrite
module_case(
    "PReLU", "ModuleActivations", "nn.PReLU()", "PReLU()", inputs={"x": X3}
)
module_case(
    "PReLU/count",
    "ModuleActivations",
    "nn.PReLU(num_parameters=16, init=0.3)",
    "PReLU(count: 16, value: 0.3)",
    inputs={"x": X3},
)

# ------------------------------------------------------------------ linear

module_case(
    "Linear", "ModuleLinear", "nn.Linear(16, 5)", "Linear(16, 5)", inputs={"x": X3}
)
module_case(
    "Linear/noBias",
    "ModuleLinear",
    "nn.Linear(16, 5, bias=False)",
    "Linear(16, 5, bias: false)",
    inputs={"x": X3},
)
module_case(
    "Bilinear",
    "ModuleLinear",
    "nn.Bilinear(16, 8, 5)",
    "Bilinear(16, 8, 5)",
    inputs={"x": Normal((2, 16)), "y": Normal((2, 8))},
    call_py="module(x, y)",
    call_swift="module(x, y)",
)
module_case(
    "Embedding",
    "ModuleLinear",
    "nn.Embedding(10, 8)",
    "Embedding(embeddingCount: 10, dimensions: 8)",
    inputs={"x": RandInt(0, 10, (2, 6))},
)
module_case(
    "Embedding/asLinear",
    "ModuleLinear",
    "nn.Embedding(10, 8)",
    "Embedding(embeddingCount: 10, dimensions: 8)",
    inputs={"x": Normal((2, 6, 8))},
    call_py="module.as_linear(x)",
    call_swift="module.asLinear(x)",
)

# ------------------------------------------------------------- convolution

module_case(
    "Conv1d",
    "ModuleConvolution",
    "nn.Conv1d(4, 3, 3)",
    "Conv1d(inputChannels: 4, outputChannels: 3, kernelSize: 3)",
    inputs={"x": Normal((2, 10, 4))},
)
module_case(
    "Conv1d/stridePadding",
    "ModuleConvolution",
    "nn.Conv1d(4, 3, 3, stride=2, padding=1)",
    "Conv1d(inputChannels: 4, outputChannels: 3, kernelSize: 3, stride: 2, padding: 1)",
    inputs={"x": Normal((2, 10, 4))},
)
module_case(
    "Conv1d/noBias",
    "ModuleConvolution",
    "nn.Conv1d(4, 3, 3, bias=False)",
    "Conv1d(inputChannels: 4, outputChannels: 3, kernelSize: 3, bias: false)",
    inputs={"x": Normal((2, 10, 4))},
)
module_case(
    "Conv2d",
    "ModuleConvolution",
    "nn.Conv2d(3, 4, 3)",
    "Conv2d(inputChannels: 3, outputChannels: 4, kernelSize: 3)",
    inputs={"x": Normal((2, 8, 8, 3))},
)
module_case(
    "Conv2d/stridePadding",
    "ModuleConvolution",
    "nn.Conv2d(3, 4, 3, stride=2, padding=1)",
    "Conv2d(inputChannels: 3, outputChannels: 4, kernelSize: 3, stride: 2, padding: 1)",
    inputs={"x": Normal((2, 8, 8, 3))},
)
module_case(
    "Conv3d",
    "ModuleConvolution",
    "nn.Conv3d(2, 3, 2)",
    "Conv3d(inputChannels: 2, outputChannels: 3, kernelSize: 2)",
    inputs={"x": Normal((1, 4, 6, 6, 2))},
)
module_case(
    "ConvTransposed1d",
    "ModuleConvolution",
    "nn.ConvTranspose1d(4, 3, 3)",
    "ConvTransposed1d(inputChannels: 4, outputChannels: 3, kernelSize: 3)",
    inputs={"x": Normal((2, 8, 4))},
)
module_case(
    "ConvTransposed1d/stride",
    "ModuleConvolution",
    "nn.ConvTranspose1d(4, 3, 3, stride=2, padding=1, output_padding=1)",
    "ConvTransposed1d(inputChannels: 4, outputChannels: 3, kernelSize: 3, stride: 2, "
    "padding: 1, outputPadding: 1)",
    inputs={"x": Normal((2, 8, 4))},
)
module_case(
    "ConvTransposed2d",
    "ModuleConvolution",
    "nn.ConvTranspose2d(3, 4, 3)",
    "ConvTransposed2d(inputChannels: 3, outputChannels: 4, kernelSize: 3)",
    inputs={"x": Normal((2, 6, 6, 3))},
)
module_case(
    "ConvTransposed3d",
    "ModuleConvolution",
    "nn.ConvTranspose3d(2, 3, 2)",
    "ConvTransposed3d(inputChannels: 2, outputChannels: 3, kernelSize: 2)",
    inputs={"x": Normal((1, 4, 4, 4, 2))},
)

# ---------------------------------------------------------- normalization

module_case(
    "LayerNorm",
    "ModuleNormalization",
    "nn.LayerNorm(16)",
    "LayerNorm(dimensions: 16)",
    inputs={"x": X3},
)
module_case(
    "LayerNorm/noAffine",
    "ModuleNormalization",
    "nn.LayerNorm(16, affine=False)",
    "LayerNorm(dimensions: 16, affine: false)",
    inputs={"x": X3},
)
module_case(
    "RMSNorm",
    "ModuleNormalization",
    "nn.RMSNorm(16)",
    "RMSNorm(dimensions: 16)",
    inputs={"x": X3},
)
module_case(
    "GroupNorm",
    "ModuleNormalization",
    "nn.GroupNorm(4, 16)",
    "GroupNorm(groupCount: 4, dimensions: 16)",
    inputs={"x": X3},
)
module_case(
    "GroupNorm/pytorchCompatible",
    "ModuleNormalization",
    "nn.GroupNorm(4, 16, pytorch_compatible=True)",
    "GroupNorm(groupCount: 4, dimensions: 16, pytorchCompatible: true)",
    inputs={"x": X3},
)
module_case(
    "InstanceNorm",
    "ModuleNormalization",
    "nn.InstanceNorm(16)",
    "InstanceNorm(dimensions: 16)",
    inputs={"x": X3},
)
module_case(
    "InstanceNorm/affine",
    "ModuleNormalization",
    "nn.InstanceNorm(16, affine=True)",
    "InstanceNorm(dimensions: 16, affine: true)",
    inputs={"x": X3},
)
module_case(
    "BatchNorm/eval",
    "ModuleNormalization",
    "nn.BatchNorm(16)",
    "BatchNorm(featureCount: 16)",
    inputs={"x": X3},
    note="eval mode uses the (rewritten) running statistics",
)
module_case(
    "BatchNorm/training",
    "ModuleNormalization",
    "nn.BatchNorm(16)",
    "BatchNorm(featureCount: 16)",
    inputs={"x": X3},
    training=True,
    note="training mode normalizes with the batch statistics; deterministic, "
    "unlike Dropout",
)
module_case(
    "BatchNorm/noTrackRunningStats",
    "ModuleNormalization",
    "nn.BatchNorm(16, track_running_stats=False)",
    "BatchNorm(featureCount: 16, trackRunningStats: false)",
    inputs={"x": X3},
)

# --------------------------------------------------------------- dropout
#
# only eval mode: in training mode these draw random numbers during the forward
# pass, and python and Swift arrive there with different random states because
# their module initializers consume a different number of keys

for name, py, swift, shape in [
    ("Dropout", "nn.Dropout()", "Dropout()", (2, 8, 16)),
    ("Dropout2d", "nn.Dropout2d()", "Dropout2d()", (2, 8, 8, 4)),
    ("Dropout3d", "nn.Dropout3d()", "Dropout3d()", (2, 4, 8, 8, 4)),
]:
    module_case(
        f"{name}/eval",
        "ModuleDropout",
        py,
        swift,
        inputs={"x": Normal(shape)},
        note="eval mode is the identity; training mode is stochastic and cannot be "
        "compared value-by-value",
    )

# --------------------------------------------------------------- pooling

POOL_SHAPES = {
    "1d": (2, 16, 4),
    "2d": (2, 8, 8, 4),
    "3d": (2, 4, 8, 8, 4),
}

for kind in ("Max", "Avg"):
    for dimension, shape in POOL_SHAPES.items():
        module_case(
            f"{kind}Pool{dimension}",
            "ModulePooling",
            f"nn.{kind}Pool{dimension}(kernel_size=2, stride=2)",
            f"{kind}Pool{dimension}(kernelSize: 2, stride: 2)",
            inputs={"x": Normal(shape)},
        )
    module_case(
        f"{kind}Pool2d/padding",
        "ModulePooling",
        f"nn.{kind}Pool2d(kernel_size=3, stride=2, padding=1)",
        f"{kind}Pool2d(kernelSize: 3, stride: 2, padding: 1)",
        inputs={"x": Normal((2, 8, 8, 4))},
    )

# --------------------------------------------------- positional encoding

module_case(
    "RoPE",
    "ModulePositional",
    "nn.RoPE(8)",
    "RoPE(dimensions: 8)",
    inputs={"x": Normal((2, 4, 8))},
)
module_case(
    "RoPE/traditional",
    "ModulePositional",
    "nn.RoPE(8, traditional=True, base=500.0, scale=0.5)",
    "RoPE(dimensions: 8, traditional: true, base: 500.0, scale: 0.5)",
    inputs={"x": Normal((2, 4, 8))},
)
module_case(
    "RoPE/offset",
    "ModulePositional",
    "nn.RoPE(8)",
    "RoPE(dimensions: 8)",
    inputs={"x": Normal((2, 4, 8))},
    call_py="module(x, offset=2)",
    call_swift="module(x, offset: 2)",
)
module_case(
    "SinusoidalPositionalEncoding",
    "ModulePositional",
    "nn.SinusoidalPositionalEncoding(8)",
    "SinusoidalPositionalEncoding(dimensions: 8)",
    inputs={"x": Normal((2, 4, 8))},
)
module_case(
    "SinusoidalPositionalEncoding/cosineFirst",
    "ModulePositional",
    "nn.SinusoidalPositionalEncoding(8, cos_first=True, full_turns=True)",
    "SinusoidalPositionalEncoding(dimensions: 8, cosineFirst: true, fullTurns: true)",
    inputs={"x": Normal((2, 4, 8))},
)
module_case(
    "ALiBi",
    "ModulePositional",
    "nn.ALiBi()",
    "ALiBi()",
    inputs={"scores": Normal((1, 4, 6, 6))},
    call_py="module(scores)",
    call_swift="module(attentionScores: scores)",
)

# ------------------------------------------------------------- upsample

module_case(
    "Upsample/nearest",
    "ModuleUpsample",
    "nn.Upsample(scale_factor=2)",
    "Upsample(scaleFactor: 2.0)",
    inputs={"x": Normal((2, 4, 4, 3))},
)
module_case(
    "Upsample/linear",
    "ModuleUpsample",
    'nn.Upsample(scale_factor=2, mode="linear")',
    "Upsample(scaleFactor: 2.0, mode: .linear())",
    inputs={"x": Normal((2, 4, 4, 3))},
)
module_case(
    "Upsample/linear/alignCorners",
    "ModuleUpsample",
    'nn.Upsample(scale_factor=2, mode="linear", align_corners=True)',
    "Upsample(scaleFactor: 2.0, mode: .linear(alignCorners: true))",
    inputs={"x": Normal((2, 4, 4, 3))},
)
module_case(
    "Upsample/cubic",
    "ModuleUpsample",
    'nn.Upsample(scale_factor=2, mode="cubic")',
    "Upsample(scaleFactor: 2.0, mode: .cubic())",
    inputs={"x": Normal((2, 4, 4, 3))},
)
module_case(
    "Upsample/scaleFactors",
    "ModuleUpsample",
    "nn.Upsample(scale_factor=(2, 3))",
    "Upsample(scaleFactor: [2.0, 3.0])",
    inputs={"x": Normal((2, 4, 4, 3))},
)

# ------------------------------------------------------------ attention

module_case(
    "MultiHeadAttention",
    "ModuleAttention",
    "nn.MultiHeadAttention(16, 4)",
    "MultiHeadAttention(dimensions: 16, numHeads: 4)",
    inputs={"x": Normal((2, 6, 16))},
    call_py="module(x, x, x)",
    call_swift="module(x, keys: x, values: x)",
)
module_case(
    "MultiHeadAttention/bias",
    "ModuleAttention",
    "nn.MultiHeadAttention(16, 4, bias=True)",
    "MultiHeadAttention(dimensions: 16, numHeads: 4, bias: true)",
    inputs={"x": Normal((2, 6, 16))},
    call_py="module(x, x, x)",
    call_swift="module(x, keys: x, values: x)",
)
module_case(
    "MultiHeadAttention/mask",
    "ModuleAttention",
    "nn.MultiHeadAttention(16, 4)",
    "MultiHeadAttention(dimensions: 16, numHeads: 4)",
    inputs={
        "x": Normal((2, 6, 16)),
        "mask": Expression(
            py="nn.MultiHeadAttention.create_additive_causal_mask(6)",
            swift="MultiHeadAttention.createAdditiveCausalMask(6)",
        ),
    },
    call_py="module(x, x, x, mask)",
    call_swift="module(x, keys: x, values: x, mask: mask)",
)

# ------------------------------------------------------------ recurrent
#
# LSTM returns (hidden, cell) and belongs to the multiple-return generator

module_case(
    "RNN",
    "ModuleRecurrent",
    "nn.RNN(4, 3)",
    "RNN(inputSize: 4, hiddenSize: 3)",
    inputs={"x": Normal((2, 5, 4))},
)
module_case(
    "RNN/noBias",
    "ModuleRecurrent",
    "nn.RNN(4, 3, bias=False)",
    "RNN(inputSize: 4, hiddenSize: 3, bias: false)",
    inputs={"x": Normal((2, 5, 4))},
)
module_case(
    "RNN/hidden",
    "ModuleRecurrent",
    "nn.RNN(4, 3)",
    "RNN(inputSize: 4, hiddenSize: 3)",
    inputs={"x": Normal((2, 5, 4)), "hidden": Normal((2, 3))},
    call_py="module(x, hidden)",
    call_swift="module(x, hidden: hidden)",
)
module_case(
    "GRU",
    "ModuleRecurrent",
    "nn.GRU(4, 3)",
    "GRU(inputSize: 4, hiddenSize: 3)",
    inputs={"x": Normal((2, 5, 4))},
)
module_case(
    "GRU/noBias",
    "ModuleRecurrent",
    "nn.GRU(4, 3, bias=False)",
    "GRU(inputSize: 4, hiddenSize: 3, bias: false)",
    inputs={"x": Normal((2, 5, 4))},
)

# files that need more than `import MLX`
MODULE_IMPORTS = {
    "ModuleActivations": ["MLXNN"],
    "ModuleLinear": ["MLXNN"],
    "ModuleConvolution": ["MLXNN"],
    "ModuleNormalization": ["MLXNN"],
    "ModuleDropout": ["MLXNN"],
    "ModulePooling": ["MLXNN"],
    "ModulePositional": ["MLXNN"],
    "ModuleUpsample": ["MLXNN"],
    "ModuleAttention": ["MLXNN"],
    "ModuleRecurrent": ["MLXNN"],
}
