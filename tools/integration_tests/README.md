# Integration test generator (single-value ops)

Generates swift-testing tests that compare Swift `MLX` results against values
produced by python `mlx`.

```sh
# generate (needs python mlx + a GPU; python mlx should match Source/Cmlx/mlx)
# every run first checks the cases against the installed python API (--no-check
# skips it)
python3 tools/integration_tests/generate.py

# see what would be generated, without mlx
python3 tools/integration_tests/generate.py --list

# check that every case's Swift expression parses, without mlx (temp dir)
python3 tools/integration_tests/generate.py --syntax-only

# exercise the generator itself with a numpy backend (temp dir, throwaway values)
python3 tools/integration_tests/generate.py --self-test
```

Output: `Tests/MLXTests/Integration/Generated/Generated<File>Tests.swift`
(the `Generated` prefix matches the suite type name and avoids colliding with the
hand written `FFTTests`/`LinalgTests`/`QuantizationTests`), formatted with
`swift-format` (so the repo's `pre-commit` style check stays green).
`Tests/MLXTests/Integration/IntegrationSupport.swift` is **hand written** and
holds all of the comparison policy.  It replaces the retired
`tools/generate_integration_tests.py` / `Tests/MLXTests/IntegrationTests.swift`
pair (`MAINTENANCE.md` step 10).

The tables are 605 cases in 26 files (514 function + 91 module):

| file | what |
| --- | --- |
| `RandomInputs` | random generation itself: per generator/dtype, per shape, draw order |
| `Elementwise` | unary math, predicates, dtype conversions, complex parts |
| `Binary` | binary ops, operators, comparisons, logical/bitwise, matmul family |
| `Reduction` | reductions, cumulative ops, argMin/argMax, softmax |
| `Shape` | shape/structure ops, sort/partition, take/put, trace, hadamard |
| `Indexing` | subscript forms (ranges, `.ellipsis`, `.newAxis`, strides, arrays) |
| `Factory` | `arange`, `linspace`, `zeros`/`ones`/`full`, `eye`, windows |
| `Convolution` | `convolve`, `conv1d/2d/3d`, `convTransposed*`, `convGeneral` |
| `Quantization` | `quantized`/`dequantized`/`quantizedMM` per bits, gather/masked MM, fp8 |
| `FFT` | the whole `mx.fft` surface, including complex results |
| `Linalg` | single-value linalg (`inv`, `cholesky`, `solve`, `det`, ...) |
| `Fast` | `rmsNorm`, `layerNorm`, `RoPE`, `scaledDotProductAttention` |
| `Random` | `gumbel`, `laplace`, `truncatedNormal`, `categorical`, `permutation`, ... |
| `Activations` | the `MLXNN` activation free functions |
| `Losses` | the `MLXNN` loss functions, per `reduction` |
| `Defaults` | bare calls (no optional arguments) on both sides, to pin defaults down |
| `ModuleActivations` | the activation layers (`ReLU`, `GELU`, `Softmax`, `PReLU`, ...) |
| `ModuleLinear` | `Linear`, `Bilinear`, `Embedding` (+ `asLinear`) |
| `ModuleConvolution` | `Conv1d/2d/3d`, `ConvTransposed1d/2d/3d` |
| `ModuleNormalization` | `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm`, `BatchNorm` (eval + training) |
| `ModuleDropout` | `Dropout`, `Dropout2d`, `Dropout3d` in eval mode |
| `ModulePooling` | `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d` |
| `ModulePositional` | `RoPE`, `SinusoidalPositionalEncoding`, `ALiBi` |
| `ModuleUpsample` | `Upsample` in every mode |
| `ModuleAttention` | `MultiHeadAttention` (+ bias, causal mask) |
| `ModuleRecurrent` | `RNN`, `GRU` |

## Scope

Two kinds of case, both producing **one array**:

- `cases.py` -- free functions, operators, methods (`Case`)
- `modules.py` -- `MLXNN.Module` layers (`ModuleCase`)
- `optimizers.py` -- optimizers over several steps (`OptimizerCase`) and learning
  rate schedules (`ScheduleCase`)

Multiple return values (`split`, `qr`, `svd`, `lu`, `divmod`, `LSTM`, ...) still
need their own generator -- see `tools/integration-tests-plan.md`.

## Optimizer and schedule cases

```python
optimizer_case(
    "Adam/biasCorrection",
    "optim.Adam(learning_rate=0.1, bias_correction=True)",
    "Adam(learningRate: 0.1, biasCorrection: true)",
    steps=3,
)

schedule_case(
    "cosineDecay",
    "optim.cosine_decay(0.1, 8)",
    "cosineDecay(0.1, decaySteps: 8)",
)
```

An optimizer case runs `steps` updates on one 2-D and one 1-D parameter (some
optimizers treat ranks differently), with the gradient of
`sum((parameter - target) ** 2)` recomputed from the *current* parameters each
step, and compares the resulting parameters.  Multiple steps are the point:
Adam's bias correction, Adafactor's step factor and every momentum term are
identical at step 1 and diverge afterwards, which is what the retired
single-step generator could not see.

A schedule case samples the schedule over `0 ..< steps` and compares the whole
curve as one array.

`$SCHEDULE` in an optimizer expression is substituted with the schedule; python
passes it as `learning_rate=` and updates it internally, while Swift assigns
`optimizer.learningRate = schedule(step)` each step -- so those cases also check
that the two conventions agree on which step uses which rate.

`apply(gradients:modelParameters:)` is internal, so the generated optimizer tests
use `@testable import MLXOptimizers`.

## Module cases

```python
module_case(
    "Linear", "ModuleLinear",           # -> GeneratedModuleLinearTests.swift
    "nn.Linear(16, 5)",                 # python module
    "Linear(16, 5)",                    # swift module
    inputs={"x": Normal((2, 8, 16))},   # declared before the module is built
)
```

emits, for each case:

```swift
let x = MLXRandom.normal([2, 8, 16], dtype: .float32, loc: 0.0, scale: 1.0)
let module = Linear(16, 5)
module.update(parameters: module.mapParameters { deterministicParameter($0) })
module.train(false)
expectParameters(module, [("bias", [5]), ("weight", [5, 16])])
let result = module(x)
expectSummary(result, ArraySummary(...), tolerance: .float32)
```

Three things to know:

1. **Parameters are rewritten, not shared.** Every parameter is replaced with a
   deterministic function of its own shape (`deterministicParameter` in
   `IntegrationSupport.swift`, `PARAMETER_PY` in `core.py` -- they must stay in
   sync).  The old generator compared `nn.Linear(16, 5)` against `Linear(16, 5)`
   and relied on both drawing the *same* random initialization in the same order:
   true by luck for a few layers, false in general.
2. **Parameter names and shapes are asserted** against python's
   (`expectParameters`).  These names are how mlx-swift loads python checkpoints,
   so a rename is a compatibility break -- and a transposed weight shows up here
   with a clear message instead of as a confusing value mismatch.
3. **Stochastic forward passes cannot be compared.** `Dropout` in training mode
   draws random numbers during the call, and the two implementations arrive there
   with different random states because their initializers consume a different
   number of keys.  Dropout cases run in eval mode; `training=True` is only for
   layers that are deterministic in training mode (`BatchNorm`).

Layers with a non-array output (`LSTM` -> `(hidden, cell)`) are out of scope here.

Use `call_py`/`call_swift` when the call is not simply `module(x)`:

```python
module_case(
    "MultiHeadAttention", "ModuleAttention",
    "nn.MultiHeadAttention(16, 4)", "MultiHeadAttention(dimensions: 16, numHeads: 4)",
    inputs={"x": Normal((2, 6, 16))},
    call_py="module(x, x, x)",
    call_swift="module(x, keys: x, values: x)",
)
```

## Adding cases

Everything lives in `cases.py`; the generator needs no changes:

```python
case(
    "takeAlong",                                    # display name + seed source
    "Shape",                                        # -> ShapeTests.swift
    {"a": Normal((4, 3)), "indices": RandInt(0, 3, (4, 1))},
    "mx.take_along_axis(a, indices, axis=1)",       # python
    "MLX.takeAlong(a, indices, axis: 1)",           # swift
)
```

Input specs (`core.py`): `Normal`, `Uniform`, `RandInt`, `Bernoulli`, `Scalar`
and `Expression` -- the last is an escape hatch that takes an explicit
python/Swift pair, for anything the others cannot express:

```python
Expression(
    py="mx.array([1.0, mx.inf, mx.nan])",
    swift="MLXArray([1.0, Float.infinity, Float.nan])",
)
```

Inputs are declared in order and later specs can use earlier names, which is how
derived inputs work -- a complex value (`r + 1j * i` / `r + i.asImaginary()`), a
well conditioned matrix (`a @ a.T + 4 * I`), the `L` from a `cholesky`, or a
non-array value such as the tuple from `quantize` (`array=False`):

```python
{
    "w": Uniform((64, 128), low=-1.0, high=1.0),
    "q": Expression(
        py="mx.quantize(w, group_size=64, bits=4)",
        swift="MLX.quantized(w, groupSize: 64, bits: 4)",
        array=False,
    ),
}
```

A case whose result is `complex64` is verified as two real arrays
(`result.realPart()` / `result.imaginaryPart()`), which is how the `FFT` cases
work.

`tools/audit_integration_coverage.py` reads these tables to report coverage (so a
new case counts immediately, before regenerating) and flags any case written
against a deprecated Swift spelling.

All case types evaluate their python side with the same namespace -- `mx`, `nn`
and `optim` -- so any case can reach any part of the python API.

Conventions:

- Swift free functions are written `MLX.foo(...)` so generated code cannot
  resolve to a stdlib/Foundation overload (`abs`, `min`, `max`, `pow`, `round`).
- Use `/` in case names for variants (`sum/axes`, `padded/edge`); the Swift test
  function is `test_sum_axes` and the display name is the case name.
- Add the *reason* for an unusual input as `note=` -- it is emitted as a comment.

## How determinism works

Each case gets a seed derived from `crc32("<file>/<name>")`, so adding or
removing a case never changes any other case's values.

- python: `mx.random.seed(seed)` (global state)
- Swift: `try withIntegrationState(seed:) { ... }`, which scopes three things to
  the case: `withRandomState(MLXRandom.RandomState(seed:))` (task-local, so the
  tests are parallel safe and do not disturb the global state),
  `Device.withDefaultDevice(.gpu)` (scoped rather than a global `setDefault`),
  and `withError` (an mlx error becomes a thrown Swift error that fails only
  that test, instead of ending the process)

Inputs are declared in the same order in both languages, so both sides consume
the same sequence of keys from the state.

## Random inputs are verified once, not per case

Op cases assert **only their result**.  Random number generation is covered by
the `RandomInputs` file, whose cases only declare and verify inputs:

```python
inputs_case("normal/float16", {"a": Normal(S, dtype="float16")})
inputs_case("sequence/mixed", {"a": Normal(S), "b": Uniform(S), "c": RandInt(0, 4)})
```

It covers each generator/dtype combination (two seeds for the common ones,
several shapes, non-default `loc`/`scale`/`low`/`high`) plus the *draw order*
within one state -- the `sequence/*` cases are what justify letting op cases
declare several inputs and check nothing but the result.

`generate.py` prints a warning if an op case uses a generator/dtype that has no
case in `RandomInputs`, so the shortcut cannot silently lose coverage.

If a particular case really does need its inputs checked (e.g. while debugging a
failure), pass `verify_inputs=True`.

## What is verified

`ArraySummary` (see `IntegrationSupport.swift`) records `shape`, `dtype`,
`mean`, `minimum`, `maximum`, `absoluteSum`, an order-sensitive
`positionChecksum` (`sum(|x| * arange(1, n+1)) / n`) and 6 evenly spaced
`samples` with their indices.  The first five are permutation invariant; the
checksum and samples are what catch a wrong axis, a transpose, or a reversed
stride -- the failure mode the old generator (mean + sum only) could not see.

Tolerances are `absolute + relative * |expected|`:

| tolerance | relative | absolute | used for |
| --- | --- | --- | --- |
| `.exact` | 0 | 0 | bool and integer results |
| `.float32` | 1e-4 | 1e-6 | default |
| `.float16` | 5e-3 | 1e-3 | `float16` / `bfloat16` results |
| `.loose` | 1e-2 | 1e-4 | opt-in, for genuinely order-dependent reductions |

The tolerance is chosen from the result dtype, or set per case with
`tolerance="loose"`.

## Provenance

Every generated file records the python `mlx` version, the vendored mlx version
from `Source/Cmlx/mlx/mlx/version.h`, the generator revision and the case count.
When a value drifts, that header says whether to suspect mlx or mlx-swift.  Bump
`GENERATOR_REVISION` in `core.py` when the emitted code changes in a way that
changes values.

## The python API check

`check_api.py` runs before every generation.  It parses the installed mlx stubs
and `nn` sources (it does **not** import mlx, so it needs no GPU) and verifies
that each case's python call actually exists with the arguments it uses:

```
python API: ModuleActivations/HardTanh/range: HardTanh.__init__ has no parameter 'min_val'
python API: ModulePositional/.../cosineFirst: ... has no parameter 'cosine_first' (has [..., 'cos_first', ...])
```

Without it these only surface part way through a generation run, one file at a
time, after the expensive python evaluation.  It reports every problem at once
and exits before generating.

## Notes / gotchas found while writing cases

- The optimizers store hyperparameters as `Float` while python stores them as
  double, so derived coefficients differ in the last bits (`1 - Float(0.95)` is
  `0.050000012`, python's `1 - 0.95` rounds to `0.05`).  Harmless for the smooth
  optimizers; Muon's Newton-Schulz iteration compounds it to ~1e-4..4e-4, so the
  momentum-using Muon cases use `tolerance="loose"` and keep `momentumZero`,
  `oneStep`, `vectorOnly`, `oneNewtonSchulzStep` and `newtonSchulz/*` as strict
  controls.  Fixing it properly means storing hyperparameters as `Double`, which is
  source-breaking -- see `todos/muon-04`.
- `Muon`'s Newton-Schulz iteration used separate scale/matmul/add ops and
  `sqrt((X * X).sum())` where python uses fused `mx.addmm` and `mx.linalg.norm`.
  A fused multiply-add rounds differently, and the iteration compounds it: the
  `Muon` case diverged from python by ~2e-4 relative while the narrowing cases
  agreed.  Now op-for-op identical (also resolves `todos/muon-01`, `muon-02`).
- `Lion`'s default `betas` was `(0.9, 0.999)`; python uses `(0.9, 0.99)`.  Lion's
  update is `lr * sign(c)`, so a wrong beta shows up as a difference of exactly
  `2 * lr` per element -- the reason it was caught here and not by a smooth
  optimizer.  `tools/audit_defaults.py` now compares class initializer defaults as
  well as function defaults, so this class of bug is covered automatically.
- `Adafactor` combined its row/column factors with a `matmul`, which only works
  for rank 2, so any parameter with three or more dimensions produced a shape
  error and then crashed on a nil state on the next step.  It now broadcasts, like
  python.  (The reason this was never caught: `OptimizerTests.checkShape` ignored
  the optimizer it was passed and built its own `SGD`, and its model had only 1-D
  parameters.  Both fixed.)
- `GRU` dropped python's `r * bhn` term when no hidden state was passed in, so the
  hidden bias had no effect on a first call (`RNN` and `LSTM` match python).
- `ALiBi` built its distance matrix from two column vectors instead of a column
  and a row, so the mask collapsed to zeros whenever the query and key lengths
  were equal, and its slope formula extended the geometric series instead of
  python's power-of-two interpolation (wrong for any head count that is not a
  power of two).  Both fixed, with tests in `PositionalEncodingTests.swift`.
- `Pool` (all six pooling layers) built its pad widths per pair rather than per
  dimension, so `padding > 0` padded the channel axis and left the first spatial
  axis unpadded.  Fixed, with tests in `PoolingTests.swift`.
- `MLX.convolve(mode: .same)` mis-ported the asymmetric padding python uses for
  even sized weights (`padLeft / 2 - 1` instead of `padLeft - 1`), so the result
  was one element short.  Fixed, with property tests in `OpsTests.swift`.
- The `Defaults` file exists to pin down default arguments: every case there
  calls with no optional arguments on both sides.  Writing it found two real
  bugs -- `MLX.nanToNum` defaulted `posInf`/`negInf` to `0` where python uses the
  dtype max/min (contradicting its own doc comment), and `MLX.tensordot`
  defaulted `axes` to `1` where python/numpy use `2`.  Both are fixed.
- `MLXRandom.laplace` requires `dtype:` where python defaults it, so the case
  spells it out.
- `linspace` used to take its dtype from the literal type (`Int` -> `int64`,
  `Double` -> `float64`); writing the case surfaced that as a bug and it now
  defaults to `float32` like python (see `MLXArray+InitTests.swift`).
- Where python's `axis=None` means "the flattened array", mlx-swift uses a
  separate overload with no `axis:`; those pairings are named `.../flat`.
- `realPart()` / `imaginaryPart()` are methods in Swift, not free functions.
- Linalg cases pin `stream=mx.cpu` / `stream: .cpu` on both sides and use SPD /
  triangular inputs so the results are well conditioned; `eigvalsh` is included
  (ascending order) but general `eig`/`eigvals` are not -- the ordering and
  eigenvector signs are not unique.
- Not covered here because they do not fit "one array in, one array out":
  `divmod`, `split`, `unstack`, `meshgrid`, `qr`, `svd`, `lu`, `lu_factor`,
  `eig`, `eigh`, `slogdet` (multi-value), `segmented_mm` (segment semantics need
  a closer look), `qqmm` (needs nvfp4 quantized inputs), transforms, modules and
  optimizers.
- The `--self-test` numpy backend does not implement every op (`softmax`,
  `logsumexp`, `cummax`, `topk`, `quantize`, `nn.*`, ...); those cases are
  reported as `skipped`.  Use `--syntax-only` to check *all* cases emit valid
  Swift -- it skips python entirely and emits placeholder values.
- Tests run with `Device.setDefault(device: .gpu)`; python defaults to the GPU
  on macOS too.  If a case needs the CPU, put `stream=mx.cpu` / `stream: .cpu`
  in both expressions.
