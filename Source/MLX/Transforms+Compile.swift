// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Synchronization

// `@unchecked Sendable`: `f`, `inputs`, and `outputs` are plain (non-`@Sendable`) stored
// values used directly outside of `lock` (during `init`), so the compiler can't verify
// this structurally even though `call(_:)` fully serializes access via `lock`.
final class CompiledFunction: @unchecked (Sendable) {

    /// unique (for the lifetime of the object) identifier for the compiled function
    private var id: UInt!

    let lock = Mutex(())

    /// the function to compile
    let f: ([MLXArray]) -> [MLXArray]

    /// any state to be observed
    let inputs: [any Updatable]
    let outputs: [any Updatable]

    let shapeless: Bool

    /// Per-call state that `inner(tracers:)` reads if/when it's actually invoked. This is
    /// updated at the top of every `innerCall`, before `mlx_closure_apply` runs, so that if
    /// a retrace happens synchronously inside that same call, `inner(tracers:)` sees the
    /// current call's arguments/state -- not whatever was captured when the closure was
    /// first built. Safe without extra synchronization: `innerCall` only ever runs while
    /// `lock` is held (see `call(_:)`), and `inner(tracers:)`, when invoked, only ever runs
    /// synchronously within the `innerCall` that triggered it.
    private var currentArgumentsCount = 0
    private var currentStateInputs: [MLXArray] = []

    /// The compiled closure, built once (lazily, on first successful call) and reused
    /// across every subsequent call. `mlx_detail_compile`'s cache is keyed by `id` and
    /// input shape/dtype, not by closure identity -- the closure passed to it is only
    /// invoked lazily, on a cache miss (first call, or a shape/dtype change that forces a
    /// retrace) -- so rebuilding it on every call was pure overhead on every cache hit.
    private var compiled: mlx_closure?

    init(
        inputs: [any Updatable], outputs: [any Updatable], shapeless: Bool,
        _ f: @escaping ([MLXArray]) -> [MLXArray]
    ) {
        self.f = f
        self.inputs = inputs
        self.outputs = outputs
        self.shapeless = shapeless
        self.id = UInt(bitPattern: Unmanaged.passUnretained(self).toOpaque())
    }

    deinit {
        if let compiled {
            mlx_closure_free(compiled)
        }
        // remove the compiled structure from the back end
        mlx_detail_compile_erase(id)
    }

    func call(_ arguments: [MLXArray]) -> [MLXArray] {
        var result: [MLXArray] = []
        lock.withLock { _ in
            result = innerCall(arguments)
        }
        return result
    }

    /// Builds the persistent compiled closure. Returns `nil` on failure (an mlx error was
    /// already dispatched -- see the call site) without caching anything, so a transient
    /// failure (e.g. inside a `withError` scope) doesn't permanently prevent a later,
    /// successful build. Must be called while `lock` is held.
    private func buildCompiledClosure() -> mlx_closure? {
        // inner function to handle the compilation -- invoked lazily, only when a retrace
        // is needed (first call, or a shape/dtype change that invalidates the cache). Reads
        // `currentArgumentsCount`/`currentStateInputs` fresh at invocation time rather than
        // capturing fixed values, since this closure is built once but may be invoked (if
        // ever again) during a much later call than the one that built it.
        func inner(tracers: [MLXArray]) -> [MLXArray] {
            let argumentsCount = self.currentArgumentsCount
            let stateInputs = self.currentStateInputs

            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracerArguments = Array(tracers.prefix(argumentsCount))

            // save a snapshot of the inner state
            let savedStateInputs = stateInputs.map { $0.copyContext() }

            // replace the inner state with the tracers
            for (s, tracer) in zip(stateInputs, tracers[argumentsCount...]) {
                s._updateInternal(tracer)
            }

            // call the function with the tracer arguments
            // and the state holding tracers
            let result = self.f(tracerArguments)

            // recapture the state as it may have changed
            let stateOutputTracers = self.outputs.flatMap { $0.innerState() }.map {
                $0.copyContext()
            }

            // put the original values back in the state
            for (s, saved) in zip(stateInputs, savedStateInputs) {
                s._updateInternal(saved)
            }

            // return the result of the function and the state
            return result + stateOutputTracers
        }

        let innerClosure = new_mlx_closure(inner(tracers:))
        defer { mlx_closure_free(innerClosure) }

        var compiled = mlx_closure_new()
        let compileStatus = mlx_detail_compile(&compiled, innerClosure, id, shapeless, [], 0)

        // mlx_error was already dispatched on failure:
        //   • outside withError — fatalError was called; we won't reach here
        //   • inside withError  — error is stored in the ErrorBox; return nil so
        //                         withError can throw instead of crashing downstream
        guard compileStatus == 0 else {
            mlx_closure_free(compiled)
            return nil
        }

        return compiled
    }

    func innerCall(_ arguments: [MLXArray]) -> [MLXArray] {
        currentStateInputs = inputs.flatMap { $0.innerState() }
        currentArgumentsCount = arguments.count

        evalLock.lock()
        defer { evalLock.unlock() }

        let compiled: mlx_closure
        if let existing = self.compiled {
            compiled = existing
        } else {
            guard let built = buildCompiledClosure() else {
                return []
            }
            compiled = built
            self.compiled = built
        }

        let innerInputs = arguments + currentStateInputs
        let innerInputsVector = new_mlx_vector_array(innerInputs)
        defer { mlx_vector_array_free(innerInputsVector) }

        // will compile the function (if needed) and evaluate the
        // compiled graph
        var resultVector = mlx_vector_array_new()
        let applyStatus = mlx_closure_apply(&resultVector, compiled, innerInputsVector)
        defer { mlx_vector_array_free(resultVector) }

        guard applyStatus == 0 else {
            return []
        }

        let resultsPlusStateOutput = mlx_vector_array_values(resultVector)

        // push the stateOutput into the state
        let stateOutput = outputs.flatMap { $0.innerState() }

        for (s, newValues) in zip(stateOutput, resultsPlusStateOutput.suffix(stateOutput.count)) {
            s._updateInternal(newValues)
        }

        let resultLength = resultsPlusStateOutput.count - stateOutput.count
        let results = Array(resultsPlusStateOutput.prefix(resultLength))
        return results
    }
}

/// Returns a compiled function that produces the same output as `f()`.
///
/// Any mutable state must be provided via the state parameter -- see <doc:compilation> for more
/// information.
///
/// - Parameters:
///   - inputs: input state
///   - outputs: output state
///   - shapeless: A function compiled with the `shapeless`
///     option enabled will not be recompiled when the input shape changes. Not all
///     functions can be compiled with `shapeless` enabled. Attempting to compile
///     such functions with shapeless enabled will throw. Note, changing the number
///     of dimensions or type of any input will result in a recompilation even with
///     `shapeless` set to `true`
///   - f: function to compile
/// - Returns: a new function that produces the same output as `f()`
///
/// ### See Also
/// - <doc:compilation>
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless, f)

    return { arrays in
        compileState.call(arrays)
    }
}

/// Overload of ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` that takes a single ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray) -> MLXArray
) -> @Sendable (MLXArray) -> MLXArray {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0])]
    }

    return { a in
        let r = compileState.call([a])
        // r is empty only when an MLX error fired inside a withError scope — the
        // error is already stored in the ErrorBox.  Return a placeholder so that
        // withError can throw instead of crashing with "Index out of range".
        return r.isEmpty ? MLXArray(0) : r[0]
    }
}

/// Overload of ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` that takes two ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray) -> MLXArray
)
    -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0], $0[1])]
    }

    return { a, b in
        let r = compileState.call([a, b])
        return r.isEmpty ? MLXArray(0) : r[0]
    }
}

/// Overload of ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])`` that takes three ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @Sendable @escaping (MLXArray, MLXArray, MLXArray) -> MLXArray
)
    -> @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0], $0[1], $0[2])]
    }

    return { a, b, c in
        let r = compileState.call([a, b, c])
        return r.isEmpty ? MLXArray(0) : r[0]
    }
}

/// Globally enable or disable ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``.
///
/// Default is enabled.
public func compile(enable: Bool = true) {
    if enable {
        mlx_enable_compile()
    } else {
        mlx_disable_compile()
    }
}
