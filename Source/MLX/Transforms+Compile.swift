// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

private let compileConfigurationGeneration = Mutex(UInt64(0))

/// Mutable values read by the persistent tracer closure. Keeping these in a separate object
/// avoids a retain cycle between `CompiledFunction` and the C closure that it owns.
private final class CompileTraceState: @unchecked Sendable {
    let f: ([MLXArray]) -> [MLXArray]
    let outputs: [any Updatable]

    var argumentsCount = 0
    var stateInputs: [MLXArray] = []

    init(outputs: [any Updatable], _ f: @escaping ([MLXArray]) -> [MLXArray]) {
        self.f = f
        self.outputs = outputs
    }

    /// Called synchronously by MLX while `CompiledFunction.lock` is held.
    func trace(_ tracers: [MLXArray]) -> [MLXArray] {
        let tracerArguments = Array(tracers.prefix(argumentsCount))
        let savedStateInputs = stateInputs.map { $0.copyContext() }

        for (state, tracer) in zip(stateInputs, tracers.dropFirst(argumentsCount)) {
            state._updateInternal(tracer)
        }

        // A trace temporarily installs tracer arrays in caller-owned state. Always restore the
        // original arrays before returning, including when this function gains throwing work in
        // the future.
        defer {
            for (state, saved) in zip(stateInputs, savedStateInputs) {
                state._updateInternal(saved)
            }
        }

        // The function may return one of the mutable state wrappers directly. Snapshot its MLX
        // context before the defer below restores caller-owned state, otherwise the returned
        // wrapper would be rewired to an uncaptured original compile input.
        let result = f(tracerArguments).map { $0.copyContext() }
        let stateOutputTracers = outputs.flatMap { $0.innerState() }.map { $0.copyContext() }
        return result + stateOutputTracers
    }
}

// `@unchecked Sendable`: `f`, `inputs`, and `outputs` are plain (non-`@Sendable`) stored
// values used directly outside of `lock` (during `init`), so the compiler can't verify
// this structurally even though `call(_:)` fully serializes access via `lock`.
// Note: this is all immutable state -- the `id` property is only set at init time
final class CompiledFunction: @unchecked (Sendable) {

    /// unique (for the lifetime of the object) identifier for the compiled function
    private var id: UInt!

    let lock = NSLock()

    /// the function to compile
    let f: ([MLXArray]) -> [MLXArray]

    /// any state to be observed
    let inputs: [any Updatable]
    let outputs: [any Updatable]

    let shapeless: Bool

    private let traceState: CompileTraceState

    /// Persistent wrapper returned by `mlx_detail_compile`. The actual compiled graphs remain
    /// keyed by `id` in MLX's compiler cache; retaining this wrapper avoids rebuilding the Swift
    /// trampoline and C++ `std::function` wrappers on every cache hit.
    private var compiled: mlx_closure?
    private var compiledGeneration: UInt64?

    init(
        inputs: [any Updatable], outputs: [any Updatable], shapeless: Bool,
        _ f: @escaping ([MLXArray]) -> [MLXArray]
    ) {
        self.f = f
        self.inputs = inputs
        self.outputs = outputs
        self.shapeless = shapeless
        self.traceState = CompileTraceState(outputs: outputs, f)
        self.id = UInt(bitPattern: Unmanaged.passUnretained(self).toOpaque())
    }

    deinit {
        // Serialize destruction with application of other MLX transform closures. The tracer
        // closure only retains `traceState` (not `self`), so freeing it here cannot form a cycle.
        evalLock.withLock {
            if let compiled {
                mlx_closure_free(compiled)
            }
            mlx_detail_compile_erase(id)
        }
    }

    func call(_ arguments: [MLXArray]) -> [MLXArray] {
        lock.withLock {
            innerCall(arguments)
        }
    }

    private func buildCompiledClosure() -> mlx_closure? {
        let traceState = traceState
        let innerClosure = new_mlx_closure { tracers in
            traceState.trace(tracers)
        }
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
        traceState.stateInputs = inputs.flatMap { $0.innerState() }
        traceState.argumentsCount = arguments.count

        return evalLock.withLock {
            let generation = compileConfigurationGeneration.withLock { $0 }

            // `mlx_detail_compile` observes whether compilation is enabled when it creates the
            // wrapper. Rebuild only after the public mode setter is used so a cached wrapper does
            // not permanently preserve an earlier enabled/disabled mode.
            if compiledGeneration != generation {
                if let compiled {
                    mlx_closure_free(compiled)
                    self.compiled = nil
                }
                compiledGeneration = generation
            }

            if compiled == nil {
                guard let built = buildCompiledClosure() else {
                    compiledGeneration = nil
                    return []
                }
                compiled = built
            }

            guard let compiled else {
                return []
            }

            let innerInputs = arguments + traceState.stateInputs
            let innerInputsVector = new_mlx_vector_array(innerInputs)
            defer { mlx_vector_array_free(innerInputsVector) }

            // This compiles on a cache miss (including a new shape/dtype) and evaluates the graph.
            var resultVector = mlx_vector_array_new()
            let applyStatus = mlx_closure_apply(&resultVector, compiled, innerInputsVector)
            defer { mlx_vector_array_free(resultVector) }

            guard applyStatus == 0 else {
                // MLX marks a cache entry non-empty before tracing it. If tracing fails, remove
                // that potentially incomplete entry so a later call can retry cleanly.
                mlx_detail_compile_erase(id)
                return []
            }

            let resultsPlusStateOutput = mlx_vector_array_values(resultVector)

            // push the stateOutput into the state
            let stateOutput = outputs.flatMap { $0.innerState() }

            for (state, newValues) in zip(
                stateOutput, resultsPlusStateOutput.suffix(stateOutput.count))
            {
                state._updateInternal(newValues)
            }

            let resultLength = resultsPlusStateOutput.count - stateOutput.count
            return Array(resultsPlusStateOutput.prefix(resultLength))
        }
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
    evalLock.withLock {
        let status: Int32
        if enable {
            status = mlx_enable_compile()
        } else {
            status = mlx_disable_compile()
        }
        if status == 0 {
            compileConfigurationGeneration.withLock { $0 &+= 1 }
        }
    }
}
