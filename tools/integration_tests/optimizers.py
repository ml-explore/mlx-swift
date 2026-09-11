# Copyright © 2026 Apple Inc.

"""
Case tables for optimizers and learning rate schedules.

Two kinds of case:

- `optimizer_case` runs an optimizer for several steps on a small set of
  parameters and compares the resulting parameters.  The gradient is
  `2 * (parameter - target)` recomputed from the current parameters each step, so
  the trajectory depends on the optimizer's state -- Adam's bias correction,
  Adafactor's step factor and every momentum term only appear from step 2 onwards,
  which a single-step comparison cannot see.
- `schedule_case` samples a schedule over a range of steps and compares the whole
  curve.

Every optimizer case uses one 2-D and one 1-D parameter, because some optimizers
(Adafactor, Muon) treat ranks differently.

`$SCHEDULE` in an optimizer expression is replaced with the schedule: python takes
it as `learning_rate=`, while Swift assigns `optimizer.learningRate` each step.

A note on Muon: its update is the orthogonalized momentum, computed with a
5 iteration Newton-Schulz map.  That map compounds any difference in its input, so
Muon is the one optimizer here where a float32-level difference can grow past the
default 1e-4 tolerance.  The `newtonSchulz/*` cases in `OptimizerFunctions` compare
one application of the iteration on its own; if those agree and the multi-step Muon
cases do not, the residual is compounding rather than a formula error, and the case
should say so with `tolerance="loose"`.
"""

from core import Normal, OptimizerCase, ScheduleCase, Uniform

OPTIMIZER_CASES: list[OptimizerCase] = []
SCHEDULE_CASES: list[ScheduleCase] = []

# one matrix and one vector, so rank dependent behavior is exercised
PARAMETERS = {"weight": Normal((4, 3)), "bias": Normal((5,))}


def optimizer_case(name: str, py: str, swift: str, **kwargs) -> None:
    OPTIMIZER_CASES.append(
        OptimizerCase(
            name=name,
            file=kwargs.pop("file", "Optimizers"),
            py=py,
            swift=swift,
            parameters=kwargs.pop("parameters", PARAMETERS),
            **kwargs,
        )
    )


def schedule_case(name: str, py: str, swift: str, **kwargs) -> None:
    SCHEDULE_CASES.append(
        ScheduleCase(name=name, file=kwargs.pop("file", "Schedules"), py=py, swift=swift, **kwargs)
    )


# ---------------------------------------------------------------- optimizers

optimizer_case("SGD", "optim.SGD(learning_rate=0.1)", "SGD(learningRate: 0.1)")
optimizer_case(
    "SGD/momentum",
    "optim.SGD(learning_rate=0.1, momentum=0.9)",
    "SGD(learningRate: 0.1, momentum: 0.9)",
)
optimizer_case(
    "SGD/dampening",
    "optim.SGD(learning_rate=0.1, momentum=0.9, dampening=0.1)",
    "SGD(learningRate: 0.1, momentum: 0.9, dampening: 0.1)",
)
optimizer_case(
    "SGD/weightDecay",
    "optim.SGD(learning_rate=0.1, momentum=0.9, weight_decay=0.1)",
    "SGD(learningRate: 0.1, momentum: 0.9, weightDecay: 0.1)",
)
optimizer_case(
    "SGD/nesterov",
    "optim.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)",
    "SGD(learningRate: 0.1, momentum: 0.9, nesterov: true)",
)
optimizer_case("RMSprop", "optim.RMSprop(learning_rate=0.1)", "RMSprop(learningRate: 0.1)")
optimizer_case(
    "RMSprop/alpha",
    "optim.RMSprop(learning_rate=0.1, alpha=0.5, eps=1e-6)",
    "RMSprop(learningRate: 0.1, alpha: 0.5, eps: 1e-6)",
)
optimizer_case("AdaGrad", "optim.Adagrad(learning_rate=0.1)", "AdaGrad(learningRate: 0.1)")
optimizer_case("AdaDelta", "optim.AdaDelta(learning_rate=0.1)", "AdaDelta(learningRate: 0.1)")
optimizer_case(
    "AdaDelta/rho",
    "optim.AdaDelta(learning_rate=0.1, rho=0.5, eps=1e-5)",
    "AdaDelta(learningRate: 0.1, rho: 0.5, eps: 1e-5)",
)
optimizer_case("Adam", "optim.Adam(learning_rate=0.1)", "Adam(learningRate: 0.1)")
optimizer_case(
    "Adam/betas",
    "optim.Adam(learning_rate=0.1, betas=[0.8, 0.9])",
    "Adam(learningRate: 0.1, betas: (0.8, 0.9))",
)
optimizer_case(
    "Adam/biasCorrection",
    "optim.Adam(learning_rate=0.1, bias_correction=True)",
    "Adam(learningRate: 0.1, biasCorrection: true)",
    note="bias correction is step dependent: identical at step 1, not later",
)
optimizer_case("AdamW", "optim.AdamW(learning_rate=0.1)", "AdamW(learningRate: 0.1)")
optimizer_case(
    "AdamW/weightDecay",
    "optim.AdamW(learning_rate=0.1, weight_decay=0.1, bias_correction=True)",
    "AdamW(learningRate: 0.1, weightDecay: 0.1, biasCorrection: true)",
)
optimizer_case("Adamax", "optim.Adamax(learning_rate=0.1)", "Adamax(learningRate: 0.1)")
optimizer_case("Lion", "optim.Lion(learning_rate=0.1)", "Lion(learningRate: 0.1)")
optimizer_case(
    "Lion/weightDecay",
    "optim.Lion(learning_rate=0.1, betas=[0.8, 0.9], weight_decay=0.1)",
    "Lion(learningRate: 0.1, betas: (0.8, 0.9), weightDecay: 0.1)",
)
optimizer_case(
    "Adafactor",
    "optim.Adafactor(learning_rate=0.1, relative_step=False)",
    "Adafactor(learningRate: 0.1, relativeStep: false)",
    note="the 2-D parameter is factored and the 1-D one is not",
)
optimizer_case(
    "Adafactor/relativeStep",
    "optim.Adafactor()",
    "Adafactor()",
    note="without a learning rate the step size comes from the step count",
)
optimizer_case(
    "Adafactor/beta1",
    "optim.Adafactor(learning_rate=0.1, beta_1=0.9, relative_step=False)",
    "Adafactor(learningRate: 0.1, beta1: 0.9, relativeStep: false)",
)
optimizer_case(
    "Adafactor/noScaleParameter",
    "optim.Adafactor(learning_rate=0.1, scale_parameter=False, relative_step=False, "
    "weight_decay=0.1)",
    "Adafactor(learningRate: 0.1, weightDecay: 0.1, scaleParameter: false, "
    "relativeStep: false)",
)
optimizer_case(
    "Adafactor/warmupInit",
    "optim.Adafactor(warmup_init=True)",
    "Adafactor(warmupInit: true)",
)
optimizer_case(
    "Muon",
    "optim.Muon(learning_rate=0.1)",
    "Muon(learningRate: 0.1)",
    tolerance="loose",
    note="loose: Muon compounds the Float-vs-double hyperparameter difference "
    "(todos/muon-04) to ~1e-4; momentumZero/oneStep/newtonSchulz are the controls",
)
# Muon narrowing cases: the orthogonalization only runs for rank >= 2, so splitting
# the parameter sets says whether a mismatch is in the momentum path or in the
# Newton-Schulz iteration (which amplifies any difference it is given)
optimizer_case(
    "Muon/vectorOnly",
    "optim.Muon(learning_rate=0.1)",
    "Muon(learningRate: 0.1)",
    parameters={"bias": Normal((5,))},
    note="rank 1: plain momentum update, no orthogonalization",
)
optimizer_case(
    "Muon/matrixOnly",
    "optim.Muon(learning_rate=0.1)",
    "Muon(learningRate: 0.1)",
    parameters={"weight": Normal((4, 3))},
    tolerance="loose",
    note="tall matrix: the Newton-Schulz iteration transposes first",
)
optimizer_case(
    "Muon/wide",
    "optim.Muon(learning_rate=0.1)",
    "Muon(learningRate: 0.1)",
    parameters={"weight": Normal((3, 6))},
    tolerance="loose",
    note="wide matrix: no transpose in the iteration",
)
optimizer_case(
    "Muon/oneStep",
    "optim.Muon(learning_rate=0.1)",
    "Muon(learningRate: 0.1)",
    parameters={"weight": Normal((4, 3))},
    steps=1,
    note="first update only, so the momentum state does not carry",
)
optimizer_case(
    "Muon/oneNewtonSchulzStep",
    "optim.Muon(learning_rate=0.1, ns_steps=1)",
    "Muon(learningRate: 0.1, nsSteps: 1)",
    parameters={"weight": Normal((4, 3))},
    note="a single iteration: if this agrees and the 5 step default does not, the "
    "difference is amplification rather than a formula error",
)
optimizer_case(
    "Muon/twoSteps",
    "optim.Muon(learning_rate=0.1)",
    "Muon(learningRate: 0.1)",
    parameters={"weight": Normal((4, 3))},
    steps=2,
    tolerance="loose",
    note="with oneStep and threeSteps this shows how fast a difference grows",
)
optimizer_case(
    "Muon/momentumZero",
    "optim.Muon(learning_rate=0.1, momentum=0.0, nesterov=False)",
    "Muon(learningRate: 0.1, momentum: 0.0, nesterov: false)",
    parameters={"weight": Normal((4, 3))},
    note="momentum 0 removes the (1 - momentum) coefficient, which python computes "
    "in double and Swift in Float; if this agrees and the default does not, that "
    "coefficient is the seed of the difference",
)
optimizer_case(
    "Muon/noWeightDecay",
    "optim.Muon(learning_rate=0.1, weight_decay=0.0)",
    "Muon(learningRate: 0.1, weightDecay: 0.0)",
    tolerance="loose",
    note="isolates the weight decay term (python defaults it to 0.01)",
)
optimizer_case(
    "Lion/oneStep",
    "optim.Lion(learning_rate=0.1)",
    "Lion(learningRate: 0.1)",
    steps=1,
    note="Lion's update is lr * sign(c), so a mismatched beta shows up as a "
    "difference of exactly 2 * lr per element",
)
optimizer_case(
    "Muon/noNesterov",
    "optim.Muon(learning_rate=0.1, momentum=0.8, weight_decay=0.0, nesterov=False, ns_steps=3)",
    "Muon(learningRate: 0.1, momentum: 0.8, weightDecay: 0.0, nesterov: false, nsSteps: 3)",
    tolerance="loose",
)

# a longer run: state accumulation over more steps
optimizer_case(
    "Adam/tenSteps",
    "optim.Adam(learning_rate=0.05, bias_correction=True)",
    "Adam(learningRate: 0.05, biasCorrection: true)",
    steps=10,
)
optimizer_case(
    "SGD/singleStep",
    "optim.SGD(learning_rate=0.1, momentum=0.9)",
    "SGD(learningRate: 0.1, momentum: 0.9)",
    steps=1,
    note="the first step alone, for comparison with the multi-step cases",
)

# different parameter shapes
optimizer_case(
    "Adam/shapes",
    "optim.Adam(learning_rate=0.1)",
    "Adam(learningRate: 0.1)",
    parameters={
        "scalar": Normal(()),
        "vector": Normal((7,)),
        "matrix": Normal((3, 5)),
        "tensor": Normal((2, 3, 4)),
    },
)
optimizer_case(
    "Adafactor/shapes",
    "optim.Adafactor(learning_rate=0.1, relative_step=False)",
    "Adafactor(learningRate: 0.1, relativeStep: false)",
    parameters={"vector": Normal((7,)), "matrix": Normal((3, 5)), "tensor": Normal((2, 3, 4))},
)

# positive parameters, so weight decay and orthogonalization act on a different
# sign pattern
optimizer_case(
    "SGD/positive",
    "optim.SGD(learning_rate=0.05, momentum=0.9, weight_decay=0.2)",
    "SGD(learningRate: 0.05, momentum: 0.9, weightDecay: 0.2)",
    parameters={"weight": Uniform((4, 3), low=0.5, high=1.5), "bias": Uniform((5,), low=0.5, high=1.5)},
)

# ------------------------------------------------------- optimizer + schedule
#
# python takes the schedule as `learning_rate`; Swift assigns `learningRate` each
# step.  These cases check that the two conventions produce the same trajectory
# (i.e. that step 0 uses `schedule(0)` on both sides).

optimizer_case(
    "SGD/cosineDecay",
    "optim.SGD(learning_rate=$SCHEDULE, momentum=0.9)",
    "SGD(learningRate: $SCHEDULE, momentum: 0.9)",
    schedule=("optim.cosine_decay(0.1, 10)", "cosineDecay(0.1, decaySteps: 10)"),
    steps=5,
)
optimizer_case(
    "Adam/exponentialDecay",
    "optim.Adam(learning_rate=$SCHEDULE)",
    "Adam(learningRate: $SCHEDULE)",
    schedule=("optim.exponential_decay(0.1, 0.9)", "exponentialDecay(0.1, decayRate: 0.9)"),
    steps=5,
)
optimizer_case(
    "SGD/stepDecay",
    "optim.SGD(learning_rate=$SCHEDULE)",
    "SGD(learningRate: $SCHEDULE)",
    schedule=(
        "optim.step_decay(0.1, 0.5, 2)",
        "stepDecay(0.1, decayRate: 0.5, stepSize: 2)",
    ),
    steps=6,
)

# ----------------------------------------------------------------- schedules

schedule_case(
    "exponentialDecay",
    "optim.exponential_decay(0.1, 0.9)",
    "exponentialDecay(0.1, decayRate: 0.9)",
)
schedule_case(
    "stepDecay",
    "optim.step_decay(0.1, 0.5, 3)",
    "stepDecay(0.1, decayRate: 0.5, stepSize: 3)",
)
schedule_case(
    "cosineDecay",
    "optim.cosine_decay(0.1, 8)",
    "cosineDecay(0.1, decaySteps: 8)",
    note="constant at the end value beyond decaySteps, which the 12 samples cover",
)
schedule_case(
    "cosineDecay/end",
    "optim.cosine_decay(0.1, 8, 0.01)",
    "cosineDecay(0.1, decaySteps: 8, end: 0.01)",
)
schedule_case(
    "linearSchedule",
    "optim.linear_schedule(0.0, 0.1, 8)",
    "linearSchedule(0.0, end: 0.1, steps: 8)",
)
schedule_case(
    "linearSchedule/down",
    "optim.linear_schedule(0.1, 0.0, 5)",
    "linearSchedule(0.1, end: 0.0, steps: 5)",
)
schedule_case(
    "joinSchedules",
    "optim.join_schedules("
    "[optim.linear_schedule(0.0, 0.1, 4), optim.cosine_decay(0.1, 8)], [4])",
    "joinSchedules("
    "[linearSchedule(0.0, end: 0.1, steps: 4), cosineDecay(0.1, decaySteps: 8)], "
    "boundaries: [4])",
    note="warmup then decay -- the classic use, and the boundary handling is the "
    "part that is easy to get wrong",
)
schedule_case(
    "joinSchedules/three",
    "optim.join_schedules("
    "[optim.linear_schedule(0.0, 0.1, 3), optim.step_decay(0.1, 0.5, 2), "
    "optim.linear_schedule(0.05, 0.0, 4)], [3, 7])",
    "joinSchedules("
    "[linearSchedule(0.0, end: 0.1, steps: 3), stepDecay(0.1, decayRate: 0.5, stepSize: 2), "
    "linearSchedule(0.05, end: 0.0, steps: 4)], boundaries: [3, 7])",
)

# these files need the optimizer module; `apply(gradients:modelParameters:)` is
# internal, so the generated tests use @testable
OPTIMIZER_IMPORTS = {
    "Optimizers": ["MLXNN", "@testable import MLXOptimizers"],
    "Schedules": ["@testable import MLXOptimizers"],
}


# ------------------------------------------------------- optimizer functions
#
# plain single-value cases (see cases.py) that need the optimizer module

from core import Case, Expression  # noqa: E402

OPTIMIZER_FUNCTION_CASES: list[Case] = [
    Case(
        name="clipGradNorm/norm",
        file="OptimizerFunctions",
        inputs={"a": Normal((4, 3)), "b": Normal((5,))},
        py='optim.clip_grad_norm({"a": a, "b": b}, 1.0)[1]',
        swift="clipGradNorm(gradients: ModuleParameters.unflattened([(\"a\", a), (\"b\", b)]), "
        "maxNorm: 1.0).1",
        note="the returned norm is the norm *before* clipping",
    ),
    Case(
        name="clipGradNorm/clipped",
        file="OptimizerFunctions",
        inputs={"a": Normal((4, 3)), "b": Normal((5,))},
        py='optim.clip_grad_norm({"a": a, "b": b}, 1.0)[0]["a"]',
        swift="clipGradNorm(gradients: ModuleParameters.unflattened([(\"a\", a), (\"b\", b)]), "
        "maxNorm: 1.0).0[unwrapping: \"a\"]!",
        note="the norm of a normal [4, 3] + [5] is well above 1, so this clips",
    ),
    Case(
        name="clipGradNorm/underTheLimit",
        file="OptimizerFunctions",
        inputs={"a": Uniform((4, 3), low=-0.02, high=0.02)},
        py='optim.clip_grad_norm({"a": a}, 1.0)[0]["a"]',
        swift="clipGradNorm(gradients: ModuleParameters.unflattened([(\"a\", a)]), "
        "maxNorm: 1.0).0[unwrapping: \"a\"]!",
        note="already inside the limit, so the gradients pass through unchanged",
    ),
    Case(
        name="clipGradNorm/array",
        file="OptimizerFunctions",
        inputs={"a": Normal((4, 3)), "b": Normal((5,))},
        py='optim.clip_grad_norm({"a": a, "b": b}, 0.5)[1]',
        swift="clipGradNorm(gradients: [a, b], maxNorm: 0.5).1",
        note="the Collection<MLXArray> overload should agree with the "
        "ModuleParameters one",
    ),
]

# Newton-Schulz on its own: the Muon cases compound whatever this produces, so
# comparing one application says whether a Muon mismatch is in the iteration or in
# the arithmetic around it
for label, shape in [("tall", (4, 3)), ("wide", (3, 6)), ("square", (4, 4))]:
    for steps in (1, 5):
        OPTIMIZER_FUNCTION_CASES.append(
            Case(
                name=f"newtonSchulz/{label}/steps{steps}",
                file="OptimizerFunctions",
                inputs={"a": Normal(shape)},
                py=f"optim.Muon(learning_rate=0.1)._zeropower_via_newtonschulz5(a, {steps})",
                swift=f"Muon(learningRate: 0.1).zeropowerViaNewtonSchulz5(a, steps: {steps})",
            )
        )

OPTIMIZER_FUNCTION_CASES.append(
    Case(
        name="newtonSchulz/nearlyOrthogonal",
        file="OptimizerFunctions",
        inputs={
            "a": Normal((4, 4)),
            "q": Expression(
                py="mx.linalg.qr(a, stream=mx.cpu)[0]",
                swift="MLX.qr(a, stream: .cpu).0",
            ),
        },
        py="optim.Muon(learning_rate=0.1)._zeropower_via_newtonschulz5(q, 5)",
        swift="Muon(learningRate: 0.1).zeropowerViaNewtonSchulz5(q, steps: 5)",
        note="an already orthogonal input is the fixed point of the iteration",
    )
)

OPTIMIZER_IMPORTS["OptimizerFunctions"] = ["MLXNN", "@testable import MLXOptimizers"]
