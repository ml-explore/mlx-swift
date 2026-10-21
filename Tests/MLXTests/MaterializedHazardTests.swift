// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
@_spi(MaterializedModule) import MLXNN
import Testing

@testable import MLX

// Tests for ways a MaterializedArray / MaterializedModule can end up
// mutable or not actually materialized.
//
// These assert the *desired* invariants, so some of them are expected to
// FAIL until the corresponding fix lands -- they demonstrate the hazard.
// Each test is annotated with what it is expected to observe.
//
// 1. `materialize()` inside `grad`/`vjp` would return a MaterializedArray
//    that still carries the traced graph (primitive + inputs), and that graph
//    is later mutated (detached) by an ordinary eval.  It traps instead.
// 2a. `compile(inputs:outputs:)` swapped the parameters of a sealed module
//     for tracers when they are plain `let` (not `@ParameterInfo`) and so were
//     not replaced with MaterializedArray.  They are now sealed in place and
//     compile treats sealed arrays as constants.
// 2b. `Linear.weight` / `Linear.bias` have public setters (intentionally --
//     outside a MaterializedModule a Linear is mutable), which bypass the
//     sealed-module checks.
// 3. `compile(inputs:outputs:)` bakes sealed state into the trace as
//    constants, but the compile cache does not key on them -- if the state is
//    rebound to different arrays (a different module, a replaced child, a new
//    MaterializedArray) a later call could reuse the stale trace.
//    `CompiledFunction` now checks the identity of the sealed arrays and
//    retraces when they change.
// 4. The `@Sendable` closure APIs (`compile(shapeless:_:)`, `vmapSendable`,
//    `CustomFunction`): these can capture MaterializedArray /
//    MaterializedModule (they are Sendable), and `materialize` must still be
//    rejected inside them while they are being traced.

// MARK: - Support

/// True if `array` still references a computation graph, i.e. it has a
/// primitive (and so inputs/siblings) rather than being a detached leaf.
///
/// `graphDescription` prints `Inputs:` and `Outputs:` lines followed by one
/// line per primitive in the graph -- a detached, evaluated array has none.
private func hasGraph(_ array: MLXArray) -> Bool {
    graphDescription([array])
        .split(separator: "\n")
        .dropFirst(2)
        .contains { !$0.isEmpty }
}

/// True if `a` and `b` refer to the same underlying `mlx::core::array` node
/// (the node ids match), regardless of which `MLXArray` wrapper holds them.
private func sameNode(_ a: MLXArray, _ b: MLXArray) -> Bool {
    let namer = NodeNamer()
    namer.setName("target", for: a)
    return namer.name(for: b) == "target"
}

// MARK: - Bodies for exit tests (must not capture state)

/// (1) `materialize` of a traced value inside `grad`.
private func materializeInsideGrad() {
    let g = grad { (x: MLXArray) -> MLXArray in
        materialize(x * x).sum()
    }
    eval(g(MLXArray(Float(3))))
}

/// (1) batch `materialize` of traced values inside `vjp`.
private func materializeArraysInsideVJP() {
    let (_, ct) = vjp(
        { (inputs: [MLXArray]) -> [MLXArray] in
            let m = materialize([inputs[0] * 2, inputs[0] + 1])
            return [m[0] * m[1]]
        }, primals: [MLXArray(Float(3))], cotangents: [MLXArray(Float(1))])
    eval(ct)
}

/// (1) `materialized()` of a value that does not depend on the traced input,
/// inside `grad`.
private func materializeConstantInsideGrad() {
    let c = MLXArray(Float(2))
    let g = grad { (x: MLXArray) -> MLXArray in
        (x * (c + 1).materialized()).sum()
    }
    eval(g(MLXArray(Float(3))))
}

/// (1) constructing a MaterializedModule inside `grad`.
private func materializedModuleInsideGrad() {
    let g = grad { (x: MLXArray) -> MLXArray in
        let mm = MaterializedModule(Linear(1, 1))
        return mm(x.reshaped(1, 1)).sum()
    }
    eval(g(MLXArray(Float(3))))
}

/// (1) `materialized()` of an existing MaterializedArray inside `grad`.
private func rematerializeInsideGrad() {
    let c = MLXArray(Float(2)).materialized()
    let g = grad { (x: MLXArray) -> MLXArray in
        (x * c.materialized()).sum()
    }
    #expect(g(MLXArray(Float(3))).item(Float.self) == 2)
}

/// (2a) A sealed module shared between a thread that is running
/// `compile(inputs:outputs:)` on it and threads that are just running the
/// forward pass.  If the compile trace swaps the (plain `let`) parameters for
/// tracers, the forward threads can build graphs on tracer placeholders --
/// which fail to evaluate (fatal error) or produce wrong results.  This is a
/// stress test, so a pass is not proof of absence; it is most useful under
/// the thread sanitizer.
private func compileSealedModuleWhileShared() {
    let mm = MaterializedModule(
        Sequential(layers: RMSNorm(dimensions: 16), LayerNorm(dimensions: 16)))
    let x = MLXRandom.normal([4, 16]).materialized()
    let expected = mm(x).materialized()

    DispatchQueue.concurrentPerform(iterations: 8) { worker in
        for _ in 0 ..< 100 {
            if worker == 0 {
                // a fresh CompiledFunction each time so that it traces each time
                let base = mm._base
                let compiled = compile(inputs: [base], outputs: [base]) { (input: MLXArray) in
                    mm(input)
                }
                eval(compiled(x))
            } else {
                let r = mm(x)
                #expect(r.allClose(expected).item(Bool.self))
            }
        }
    }
}

/// (2b) Assign `Linear.weight` on a sealed module.  The module is only
/// reachable through `_base` (SPI) or a reference retained in violation of
/// the `MaterializedModule` contract.
private func writeWeightOfSealedLinear() {
    let mm = MaterializedModule(Linear(4, 4))
    mm._base.weight = MLXArray.zeros([4, 4])
}

/// (2b) Assign `Linear.bias` on a sealed module built with `bias: false`.
private func writeNilBiasOfSealedLinear() {
    let mm = MaterializedModule(Linear(4, 4, bias: false))
    mm._base.bias = MLXArray.ones([4])
}

// MARK: - Tests

@Suite("MaterializedArray / MaterializedModule hazards")
struct MaterializedHazardTests {

    // MARK: (1) materialize() inside a transform

    /// Control: outside of any transform `materialize` produces a detached
    /// leaf.  This also validates `hasGraph`.
    @Test func materializeOutsideTransformIsDetached() {
        let x = MLXArray(Float(3))
        let y = x * x
        #expect(hasGraph(y))

        let m = materialize(y)
        #expect(!hasGraph(m))
    }

    /// Traps.  Inside `grad` an eval of a traced value does not detach it, so
    /// the result would still reference the trace (and be mutated by a later
    /// eval) -- it cannot be a MaterializedArray.
    @Test func materializeInsideGradTraps() async {
        await #expect(processExitsWith: .failure) {
            materializeInsideGrad()
        }
    }

    /// Traps.  Same via the batch overload and `vjp`.
    @Test func materializeArraysInsideVJPTraps() async {
        await #expect(processExitsWith: .failure) {
            materializeArraysInsideVJP()
        }
    }

    /// Traps.  The check is "inside a transformation", not "is a tracer", so
    /// this is rejected even though the array does not depend on the traced
    /// input.
    @Test func materializeConstantInsideGradTraps() async {
        await #expect(processExitsWith: .failure) {
            materializeConstantInsideGrad()
        }
    }

    /// Traps.  Constructing a MaterializedModule materializes its parameters.
    @Test func materializedModuleInsideGradTraps() async {
        await #expect(processExitsWith: .failure) {
            materializedModuleInsideGrad()
        }
    }

    /// Allowed: `materialized()` on an array that is already a
    /// MaterializedArray returns it unchanged and does not trap.
    @Test func rematerializeInsideGradIsAllowed() async {
        await #expect(processExitsWith: .success) {
            rematerializeInsideGrad()
        }
    }

    // MARK: (2a) compile(inputs:outputs:) on a sealed module

    /// Passes.  `RMSNorm.weight` and `LayerNorm.weight/bias` are plain `let`
    /// and so stay `MLXArray` after sealing, but they are sealed in place.
    /// `compile` must not swap them for tracers during the trace (which
    /// would be visible to any other thread using the shared module) or write
    /// the compiled outputs back into them afterwards.
    @Test func compileInputsOutputsDoesNotSwapSealedParameters() {
        let mm = MaterializedModule(
            Sequential(layers: RMSNorm(dimensions: 16), LayerNorm(dimensions: 16)))
        let base = mm._base

        let state = base.innerState()

        // precondition for the test: these are the parameters that could not
        // be replaced with MaterializedArray
        #expect(state.count == 3)
        #expect(state.allSatisfy { !($0 is MaterializedArray) })

        // independent handles on the original nodes
        let originals = state.map { $0.copyContext() }

        var traced = false
        var swappedDuringTrace = [Bool]()

        let compiled = compile(inputs: [base], outputs: [base]) { (x: MLXArray) in
            traced = true
            swappedDuringTrace = zip(base.innerState(), originals).map { !sameNode($0, $1) }
            return mm(x)
        }

        let x = MLXRandom.normal([4, 16])
        let expected = mm(x)
        let result = compiled(x)
        #expect(result.allClose(expected).item(Bool.self))

        // make sure the trace actually ran so the checks below mean something
        #expect(traced)
        #expect(
            swappedDuringTrace.allSatisfy { !$0 },
            "sealed parameters were swapped for tracers during compile: \(swappedDuringTrace)")

        let replacedAfterCall = zip(base.innerState(), originals).map { !sameNode($0, $1) }
        #expect(
            replacedAfterCall.allSatisfy { !$0 },
            "sealed parameters were replaced by compile outputs: \(replacedAfterCall)")
    }

    /// Passes.  This was a race (crash or mismatch) when compile swapped the
    /// sealed parameters, so it is most useful under TSan.
    @Test func compileSealedModuleWhileSharedAcrossThreads() async {
        await #expect(processExitsWith: .success) {
            compileSealedModuleWhileShared()
        }
    }

    // MARK: (2b) Linear.weight / Linear.bias setters

    // The setters are public: outside a MaterializedModule a Linear is an
    // ordinary mutable Module.  Writing them on a *sealed* module should trap,
    // like every other mutation of a sealed module.

    /// Passes -- the write traps, but in `ParameterInfo.wrappedValue.set`
    /// ("please call update()", for any non-nil current value), not in
    /// `_checkNotSealed()`.
    @Test func writeWeightOfSealedLinearTraps() async {
        await #expect(processExitsWith: .failure) {
            writeWeightOfSealedLinear()
        }
    }

    /// Passes -- `Linear.init` assigns `bias = nil`, which `ParameterInfo`
    /// stores as `.some(nil)`, so this also hits the "please call update()"
    /// trap.  (Only an optional `@ParameterInfo` that `init` never assigns
    /// can be set later.)
    @Test func writeNilBiasOfSealedLinearTraps() async {
        await #expect(processExitsWith: .failure) {
            writeNilBiasOfSealedLinear()
        }
    }
}

// MARK: - (3) compile bakes materialized state but the cache ignores the binding

/// Holds a module that can be rebound -- the indirection that lets a
/// compiled function see a different module than the one it was traced with.
private final class ModuleHolder<M: UnaryLayer>: Updatable {
    var model: M
    init(_ model: M) { self.model = model }
    func innerState() -> [MLXArray] { model.innerState() }
}

/// Holds a single array that can be rebound, e.g. state that is replaced
/// each step.
private final class ArrayHolder: Updatable {
    var value: MLXArray
    init(_ value: MLXArray) { self.value = value }
    func innerState() -> [MLXArray] { [value] }
}

/// A plain module container whose child can be replaced via
/// `update(modules:)`.
private final class Parent: Module, UnaryLayer {
    @ModuleInfo var child: Linear
    init(_ child: Linear) { self._child.wrappedValue = child }
    func callAsFunction(_ x: MLXArray) -> MLXArray { child(x) }
}

/// A `Linear` with every weight and bias element equal to `value`, so that
/// different instances give clearly different results.  If `sealed`, it is
/// the (sealed) base of a MaterializedModule: all of its parameters are
/// `@ParameterInfo` so they are all MaterializedArray.
private func makeLinear(_ value: Float, sealed: Bool) -> Linear {
    let l = Linear(weight: MLXArray.ones([4, 4]) * value, bias: MLXArray.ones([4]) * value)
    return sealed ? MaterializedModule(l)._base : l
}

/// `RMSNorm` (plain `let` weight, stays `MLXArray`) followed by `Linear`
/// (`@ParameterInfo`, becomes MaterializedArray) -- a partly materialized
/// module.  RMSNorm is first so the Linear scale shows in the output.
private func partlySealed(_ value: Float) -> Sequential {
    let model = Sequential(layers: RMSNorm(dimensions: 4), makeLinear(value, sealed: false))
    return MaterializedModule(model)._base
}

/// A fixed (non-random) input for the compiled functions.
private func compileInput() -> MLXArray {
    MLXArray((0 ..< 8).map { Float($0) }, [2, 4])
}

// Access to the mutable state is handed between threads by the test's semaphores.
private final class ReboundCompileState: @unchecked Sendable {
    let holder: ArrayHolder
    let compiled: (MLXArray) -> MLXArray
    var workerBefore: Float = 0
    var workerAfter: Float = 0

    init() {
        let holder = ArrayHolder(MLXArray(Float(1)).materialized())
        self.holder = holder
        self.compiled = compile(inputs: [holder]) { (x: MLXArray) in x * holder.value }
    }

    func value() -> Float {
        compiled(MLXArray(Float(2))).item(Float.self)
    }
}

private final class ReleasedCompileState: @unchecked Sendable {
    var compiled: CompiledFunction?
    let functionID: UInt
    var before: Float = 0
    var after: Float = 0

    init() {
        let scale = MLXArray(Float(2)).materialized()
        let compiled = CompiledFunction(shapeless: false) { [$0[0] * scale] }
        self.compiled = compiled
        self.functionID = UInt(bitPattern: Unmanaged.passUnretained(compiled).toOpaque())
    }

    // Reuse the old address-derived ID without depending on allocator reuse.
    func replacementValue() -> Float {
        let result = withEvalLock {
            let body = new_mlx_closure { [$0[0] * 3] }
            defer { mlx_closure_free(body) }
            var replacement = mlx_closure_new()
            defer { mlx_closure_free(replacement) }
            mlx_detail_compile(&replacement, body, functionID, false, [], 0)
            let input = new_mlx_vector_array([MLXArray(Float(2))])
            defer { mlx_vector_array_free(input) }
            var output = mlx_vector_array_new()
            defer { mlx_vector_array_free(output) }
            mlx_closure_apply(&output, replacement, input)
            return mlx_vector_array_values(output)[0]
        }
        return result.item(Float.self)
    }
}

@Suite("compile with rebound materialized state")
struct CompileReboundMaterializedStateTests {

    @Test func deinitErasesCacheOnTheThreadThatCompiled() {
        let state = ReleasedCompileState()
        let primed = DispatchSemaphore(value: 0)
        let released = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        let worker = Thread {
            state.before = state.compiled!.call([MLXArray(Float(2))])[0].item(Float.self)
            primed.signal()
            if released.wait(timeout: .now() + 30) == .success {
                state.after = state.replacementValue()
            }
            finished.signal()
        }
        worker.start()
        guard primed.wait(timeout: .now() + 30) == .success else {
            released.signal()
            Issue.record("worker did not populate its compilation cache")
            return
        }
        state.compiled = nil
        released.signal()
        guard finished.wait(timeout: .now() + 30) == .success else {
            Issue.record("worker did not finish the replacement call")
            return
        }
        #expect(state.before == 4)
        #expect(state.after == 6)
    }

    @Test func rebindingInvalidatesEveryCallingThreadCache() {
        let state = ReboundCompileState()
        let primed = DispatchSemaphore(value: 0)
        let rebound = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)

        #expect(state.value() == 2)
        let worker = Thread {
            state.workerBefore = state.value()
            primed.signal()
            if rebound.wait(timeout: .now() + 30) == .success {
                state.workerAfter = state.value()
            }
            finished.signal()
        }
        worker.start()
        guard primed.wait(timeout: .now() + 30) == .success else {
            rebound.signal()
            Issue.record("worker did not populate its compilation cache")
            return
        }

        state.holder.value = MLXArray(Float(3)).materialized()
        #expect(state.value() == 6)
        rebound.signal()
        guard finished.wait(timeout: .now() + 30) == .success else {
            Issue.record("worker did not finish its cached call")
            return
        }
        #expect(state.workerBefore == 2)
        #expect(state.workerAfter == 6)
    }

    /// Control: plain modules rebound between calls are threaded through as
    /// inputs and the compiled function follows the rebinding.
    @Test func plainModuleRebound() {
        let input = compileInput()
        let holder = ModuleHolder(makeLinear(1, sealed: false))
        let c = compile(inputs: [holder], outputs: [holder]) { (x: MLXArray) in holder.model(x) }

        #expect(c(input).allClose(holder.model(input)).item(Bool.self))

        holder.model = makeLinear(2, sealed: false)
        #expect(c(input).allClose(holder.model(input)).item(Bool.self))
    }

    /// Control: a plain array rebound between calls.
    @Test func plainArrayRebound() {
        let input = compileInput()
        let holder = ArrayHolder(MLXArray(Float(1)))
        let c = compile(inputs: [holder], outputs: [holder]) { (x: MLXArray) in x + holder.value }

        #expect(c(input).allClose(input + 1).item(Bool.self))

        holder.value = MLXArray(Float(2))
        #expect(c(input).allClose(input + 2).item(Bool.self))
    }

    /// Sealed A -> sealed A2 (same shapes): both have zero threaded state
    /// inputs, so without the identity check the cache would hit the trace
    /// with A's weights baked in.
    @Test func sealedModuleReboundToSealed() {
        let input = compileInput()
        let holder = ModuleHolder(makeLinear(1, sealed: true))
        let c = compile(inputs: [holder], outputs: [holder]) { (x: MLXArray) in holder.model(x) }

        #expect(c(input).allClose(holder.model(input)).item(Bool.self))

        holder.model = makeLinear(2, sealed: true)
        #expect(
            c(input).allClose(holder.model(input)).item(Bool.self),
            "compiled function used the weights of the module it was traced with")
    }

    /// Sealed A (0 state inputs) -> plain B (2) retraces on the input count
    /// alone, but sealed A2 (0 again) would match A's cache entry without the
    /// identity check.
    @Test func sealedPlainSealed() {
        let input = compileInput()
        let holder = ModuleHolder(makeLinear(1, sealed: true))
        let c = compile(inputs: [holder], outputs: [holder]) { (x: MLXArray) in holder.model(x) }

        #expect(c(input).allClose(holder.model(input)).item(Bool.self))

        holder.model = makeLinear(2, sealed: false)
        #expect(c(input).allClose(holder.model(input)).item(Bool.self))

        holder.model = makeLinear(3, sealed: true)
        #expect(
            c(input).allClose(holder.model(input)).item(Bool.self),
            "compiled function reused the trace of an earlier sealed module")
    }

    /// Partly sealed A -> partly sealed B: the same number of threaded inputs
    /// (the RMSNorm weight), so without the identity check the cache would hit
    /// and combine B's RMSNorm weight with A's baked Linear weights.
    @Test func partlySealedRebound() {
        let input = compileInput()
        let holder = ModuleHolder(partlySealed(1))
        let c = compile(inputs: [holder], outputs: [holder]) { (x: MLXArray) in holder.model(x) }

        #expect(c(input).allClose(holder.model(input)).item(Bool.self))

        holder.model = partlySealed(2)
        #expect(
            c(input).allClose(holder.model(input)).item(Bool.self),
            "compiled function mixed baked weights from the traced module")
    }

    /// Same as `sealedModuleReboundToSealed` but through a
    /// plain (unsealed) parent `Module` and `update(modules:)`, the usual way
    /// to swap a layer.
    @Test func sealedChildReplacedViaUpdateModules() {
        let input = compileInput()
        let parent = Parent(makeLinear(1, sealed: true))
        let c = compile(inputs: [parent], outputs: [parent]) { (x: MLXArray) in parent(x) }

        #expect(c(input).allClose(parent(input)).item(Bool.self))

        var u = NestedDictionary<String, Module>()
        u["child"] = .value(makeLinear(2, sealed: true))
        parent.update(modules: u)

        #expect(
            c(input).allClose(parent(input)).item(Bool.self),
            "compiled function used the weights of the child it was traced with")
    }

    /// State that is replaced by a new MaterializedArray each step: without
    /// the identity check the first value is baked in and later ones are
    /// ignored.  (This retraces on every rebinding -- state that changes every
    /// step should be a plain array.)
    @Test func materializedArrayRebound() {
        let input = compileInput()
        let holder = ArrayHolder(MLXArray(Float(1)).materialized())
        let c = compile(inputs: [holder], outputs: [holder]) { (x: MLXArray) in x + holder.value }

        #expect(c(input).allClose(input + 1).item(Bool.self))

        holder.value = MLXArray(Float(2)).materialized()
        #expect(
            c(input).allClose(input + 2).item(Bool.self),
            "compiled function used the MaterializedArray it was traced with")
    }

    /// The identity check must not retrace while the binding is stable, and
    /// must retrace (once) when it changes.
    @Test func sealedRetracesOnlyWhenRebound() {
        let input = compileInput()
        let holder = ModuleHolder(makeLinear(1, sealed: true))

        var traces = 0
        let c = compile(inputs: [holder], outputs: [holder]) { (x: MLXArray) in
            traces += 1
            return holder.model(x)
        }

        for _ in 0 ..< 3 {
            eval(c(input))
        }
        #expect(traces == 1)

        holder.model = makeLinear(2, sealed: true)
        for _ in 0 ..< 3 {
            #expect(c(input).allClose(holder.model(input)).item(Bool.self))
        }
        #expect(traces == 2)
    }
}

// MARK: - (4) @Sendable compile / vmap / CustomFunction

/// (4) `materialize` inside a `@Sendable` compiled function.
private func materializeInsideSendableCompile() {
    let c: @Sendable (MLXArray) -> MLXArray = compile { x in
        (x * 2).materialized()
    }
    eval(c(MLXArray(Float(3))))
}

/// (4) `materialize` inside a `vmapSendable` function.
private func materializeInsideVmapSendable() {
    let f = vmapSendable { (x: MLXArray) -> MLXArray in
        (x * 2).materialized()
    }
    eval(f(MLXArray([Float(1), 2, 3])))
}

/// (4) `materialize` inside a `CustomFunction` forward that is run by `grad`.
private func materializeInsideCustomForwardUnderGrad() {
    let f = CustomFunction {
        Forward { inputs in [(inputs[0] * 2).materialized()] }
        VJP { _, cotangents in [cotangents[0] * 2] }
    }
    let g = grad { (x: MLXArray) -> MLXArray in f([x])[0].sum() }
    eval(g(MLXArray(Float(3))))
}

/// (4) `materialize` inside a `CustomFunction` VJP (run by `grad`).
private func materializeInsideCustomVJPUnderGrad() {
    let f = CustomFunction {
        Forward { inputs in [inputs[0] * 2] }
        VJP { _, cotangents in [(cotangents[0] * 2).materialized()] }
    }
    let g = grad { (x: MLXArray) -> MLXArray in f([x])[0].sum() }
    eval(g(MLXArray(Float(3))))
}

@Suite("@Sendable transforms with materialized state")
struct SendableTransformsMaterializedTests {

    /// A `@Sendable` compiled function may capture a MaterializedModule (its
    /// parameters are sealed constants) and be called concurrently.
    @Test func sendableCompileCapturingMaterializedModule() async {
        let model = MaterializedModule(makeLinear(1, sealed: false))
        let compiled: @Sendable (MLXArray) -> MLXArray = compile { x in
            model(x)
        }

        let input = compileInput().materialized()
        let expected = model(input).materialized()

        await withTaskGroup(of: Bool.self) { group in
            for _ in 0 ..< 8 {
                group.addTask {
                    compiled(input).allClose(expected).item(Bool.self)
                }
            }
            for await ok in group {
                #expect(ok)
            }
        }
    }

    /// A `@Sendable` compiled function may capture a MaterializedArray.
    @Test func sendableCompileCapturingMaterializedArray() {
        let scale = MLXArray(Float(3)).materialized()
        let compiled: @Sendable (MLXArray) -> MLXArray = compile { x in
            x * scale
        }
        for offset in 0 ..< 3 {
            let input = compileInput() + Float(offset)
            #expect(compiled(input).allClose(input * 3).item(Bool.self))
            #expect(isEvaluated(scale))
            #expect(!hasGraph(scale))
            #expect(scale.item(Float.self) == 3)
        }
    }

    @Test func sendableCompileCapturingMaterializedArrayConcurrently() async {
        let scale = MLXArray(Float(3)).materialized()
        let compiled: @Sendable (MLXArray) -> MLXArray = compile { x in
            x * scale
        }

        await withTaskGroup(of: Bool.self) { group in
            for worker in 0 ..< 8 {
                group.addTask {
                    for iteration in 0 ..< 25 {
                        let input = compileInput() + Float(worker * 25 + iteration)
                        if !compiled(input).allClose(input * 3).item(Bool.self) {
                            return false
                        }
                    }
                    return true
                }
            }
            for await ok in group {
                #expect(ok)
            }
        }
        #expect(isEvaluated(scale))
        #expect(!hasGraph(scale))
        #expect(scale.item(Float.self) == 3)
    }

    /// `CustomFunction` closures may capture a MaterializedArray, including
    /// the VJP that is run by `grad`.
    @Test func customFunctionCapturingMaterializedArray() {
        let scale = MLXArray(Float(3)).materialized()
        let f = CustomFunction {
            Forward { inputs in [inputs[0] * scale] }
            VJP { _, cotangents in [cotangents[0] * scale] }
        }
        let g = grad { (x: MLXArray) -> MLXArray in f([x])[0].sum() }
        #expect(g(MLXArray(Float(2))).item(Float.self) == 3)
    }

    /// Allowed: a `CustomFunction` forward that is *not* being run by a
    /// transformation is ordinary code, so `materialize` works.
    @Test func materializeInsideCustomForwardOutsideTransform() {
        let f = CustomFunction {
            Forward { inputs in [(inputs[0] * 2).materialized()] }
        }
        #expect(f([MLXArray(Float(3))])[0].item(Float.self) == 6)
    }

    @Test func materializeInsideSendableCompileTraps() async {
        await #expect(processExitsWith: .failure) {
            materializeInsideSendableCompile()
        }
    }

    @Test func materializeInsideVmapSendableTraps() async {
        await #expect(processExitsWith: .failure) {
            materializeInsideVmapSendable()
        }
    }

    @Test func materializeInsideCustomForwardUnderGradTraps() async {
        await #expect(processExitsWith: .failure) {
            materializeInsideCustomForwardUnderGrad()
        }
    }

    @Test func materializeInsideCustomVJPUnderGradTraps() async {
        await #expect(processExitsWith: .failure) {
            materializeInsideCustomVJPUnderGrad()
        }
    }
}
