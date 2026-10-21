// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

/// A weak handle to MLX's thread-local cache. Retaining this wrapper does not keep
/// the backend cache alive after its thread exits.
private final class ThreadCompilationCache {
    private let ctx: mlx_compile_cache

    private init() {
        var cache = mlx_compile_cache_new()
        mlx_detail_compile_cache(&cache)
        self.ctx = cache
    }

    deinit {
        mlx_compile_cache_free(ctx)
    }

    static var current: ThreadCompilationCache {
        let key = "MLX.ThreadCompilationCache"
        let dictionary = Thread.current.threadDictionary
        if let cache = dictionary[key] as? ThreadCompilationCache {
            return cache
        }
        let cache = ThreadCompilationCache()
        dictionary[key] = cache
        return cache
    }

    func erase(_ functionID: UInt) {
        mlx_detail_compile_erase(ctx, functionID)
    }
}

// Mutable cache bookkeeping is guarded by evalLock. The stateful overloads also
// hold the instance lock while replacing observed inputs and outputs.
// Only the overloads accepting Sendable captures return a Sendable closure.
final class CompiledFunction: @unchecked (Sendable) {

    /// unique (for the lifetime of the object) identifier for the compiled function
    private var id: UInt!

    /// Every backend cache this function has used, guarded by evalLock.
    private var caches: [ObjectIdentifier: ThreadCompilationCache] = [:]

    /// guards the observed state (``inputs`` / ``outputs``) -- see #226
    ///
    /// This is always acquired *inside* `evalLock`; see ``call(_:)`` for the
    /// ordering rule.  It is recursive because a compiled function's body can
    /// legitimately re-enter that same compiled function on the same thread
    /// while it is being traced (a nested call with tracer inputs bypasses the
    /// cache and simply runs the body again).
    let lock = NSRecursiveLock()

    /// the function to compile
    let f: ([MLXArray]) -> [MLXArray]

    /// any state to be observed
    let inputs: [any Updatable]
    let outputs: [any Updatable]

    let shapeless: Bool

    /// The sealed arrays reachable from ``inputs`` when the cached graphs were
    /// traced, or `nil` before the first call.
    ///
    /// Sealed arrays are not threaded through the compiled function as inputs
    /// (that would mean swapping tracers into them), they are baked into the
    /// traced graph as constants.  The compiler cache only keys on the inputs,
    /// so if the state is later rebound to *different* sealed arrays (a
    /// different `MaterializedModule`, a replaced child module, a new
    /// `MaterializedArray`) the cache would happily return a graph holding the
    /// old ones.  ``innerCall(_:)`` compares the current sealed arrays against
    /// these and erases the cache (forcing a retrace) if they differ.
    ///
    /// Identity (`===`) is sufficient: a sealed array's contents can never be
    /// replaced, so the same object always refers to the same node.  The arrays
    /// are retained so that a freed one can't be replaced by a new array at the
    /// same address.  This costs no extra memory: the cached graph holds the
    /// same data.
    ///
    /// Guarded by ``lock``.
    private var tracedConstants: [MLXArray]?

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

    init(
        shapeless: Bool,
        _ f: @escaping @Sendable ([MLXArray]) -> [MLXArray]
    ) {
        self.f = f
        self.inputs = []
        self.outputs = []
        self.shapeless = shapeless
        self.id = UInt(bitPattern: Unmanaged.passUnretained(self).toOpaque())
    }

    // Erase every calling thread's cache before this address can be reused as an ID.
    deinit {
        withEvalLock {
            #if DEBUG
                EvalLockOwnership.requireHeldByCurrentThread(
                    "the compiler cache was erased without holding evalLock; "
                        + "see CompiledFunction.deinit")
                EvalLockOwnership.recordCompileErase()
            #endif
            eraseCaches()
        }
    }

    /// Must be called under evalLock, including when sealed state is rebound.
    private func eraseCaches() {
        for cache in caches.values {
            cache.erase(id)
        }
    }

    /// Evaluate the compiled function.
    ///
    /// ### Lock ordering
    ///
    /// `evalLock` is acquired **outermost**, before the per-instance ``lock``.
    /// That order is not a preference, it is forced:
    ///
    /// - A trace can call back into Swift and mutate observed state, so
    ///   `evalLock` spans the whole trace and cache invalidation (#339).
    /// - The trace calls back into Swift, and the traced body routinely calls
    ///   *other* compiled functions: every `MLXNN` activation is a file-scope
    ///   `CompiledFunction`, so e.g. `silu` inside a traced body enters
    ///   `compiledSilu.call` and takes *its* per-instance lock.
    ///
    /// So `evalLock` → ``lock`` happens whether or not we ask for it.  Taking
    /// ``lock`` first here would add the reverse order ``lock`` → `evalLock` for
    /// any thread entering from the top, and two threads -- one nested inside a
    /// trace, one not -- would deadlock on a cold compiler cache:
    ///
    ///     thread H:  L_cf(outer)  -> evalLock     -> wants L_cf(shared)
    ///     thread W:  L_cf(shared) -> wants evalLock
    ///
    /// Acquiring `evalLock` here gives the process a single global order, which
    /// makes the inversion impossible.  `Source/CompileLockRepro` reproduces the
    /// deadlock deterministically without this.
    ///
    /// The cost is small: ``innerCall(_:)`` used to take `evalLock` itself and
    /// hold it through `mlx_closure_apply` -- i.e. across the actual evaluation
    /// -- so this only moves the acquisition earlier by the state gathering and
    /// closure creation at the top of ``innerCall(_:)``.
    func call(_ arguments: [MLXArray]) -> [MLXArray] {
        withEvalLock {
            // Stateless functions need only evalLock for cache bookkeeping.
            if inputs.isEmpty && outputs.isEmpty {
                return innerCall(arguments)
            }
            return withInstanceLock {
                innerCall(arguments)
            }
        }
    }

    /// Acquire ``lock``, checking the ordering invariant first.
    ///
    /// The check is where the invariant actually lives, not in ``call(_:)``: a
    /// change that moved the per-instance lock back outside `evalLock` -- the
    /// shape that produced the deadlock -- trips this on the first compiled
    /// call, deterministically and without needing to lose a race.
    private func withInstanceLock<R>(_ body: () throws -> R) rethrows -> R {
        #if DEBUG
            EvalLockOwnership.requireHeldByCurrentThread(
                "CompiledFunction.lock was acquired without holding evalLock; "
                    + "that is the lock-order inversion, see CompiledFunction.call")
        #endif
        return try lock.withLock(body)
    }

    /// - Precondition: `evalLock` is held by the caller, and ``lock`` is held
    ///   too unless this function has no observed state; see ``call(_:)``,
    ///   which is the only caller.
    func innerCall(_ arguments: [MLXArray]) -> [MLXArray] {
        let cache = ThreadCompilationCache.current
        caches[ObjectIdentifier(cache)] = cache

        // Sealed arrays (MaterializedArray, the parameters of a sealed module)
        // can never change, and must not be mutated, so they are not threaded
        // through the tracer swap below -- they are simply captured as
        // constants, the same as any other array reachable from the closure
        // that isn't part of `inputs`/`outputs`.
        let state = inputs.flatMap { $0.innerState() }
        let stateInputs = state.filter { !$0.isSealed }
        let argumentsCount = arguments.count

        // The sealed arrays are baked into the cached graphs, so if they have
        // been rebound since those were traced the cache is stale -- see
        // ``tracedConstants``.
        if !inputs.isEmpty {
            let constants = state.filter { $0.isSealed }
            if let traced = tracedConstants {
                let same =
                    traced.count == constants.count
                    && zip(traced, constants).allSatisfy { $0 === $1 }
                if !same {
                    eraseCaches()
                    tracedConstants = constants
                }
            } else {
                tracedConstants = constants
            }
        }

        // inner function to hande the compilation.  this is called
        // once per compile (typically once overall, but can be called
        // again if the conditions for recompile change)
        func inner(tracers: [MLXArray]) -> [MLXArray] {

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
            let result = f(tracerArguments)

            // recapture the state as it may have changed
            let stateOutputTracers = outputs.flatMap { $0.innerState() }
                .filter { !$0.isSealed }
                .map { $0.copyContext() }

            // put the original values back in the state
            for (s, saved) in zip(stateInputs, savedStateInputs) {
                s._updateInternal(saved)
            }

            // return the result of the function and the state
            return result + stateOutputTracers
        }

        let innerClosure = new_mlx_closure(inner(tracers:))
        defer { mlx_closure_free(innerClosure) }

        // note: this will use the cached compile (via the id)
        // but will be able to re-evaluate with fresh state if needed
        //
        // evalLock is already held by call(_:) and covers everything through
        // mlx_closure_apply below, which is where the trace runs.
        var compiled = mlx_closure_new()
        let compileStatus = mlx_detail_compile(&compiled, innerClosure, id, shapeless, [], 0)
        defer {
            mlx_closure_free(compiled)
        }

        // mlx_error was already dispatched on failure:
        //   • outside withError — fatalError was called; we won't reach here
        //   • inside withError  — error is stored in the ErrorBox; return [] so
        //                         withError can throw instead of crashing downstream
        guard compileStatus == 0 else {
            return []
        }

        let innerInputs = arguments + stateInputs
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
        let stateOutput = outputs.flatMap { $0.innerState() }.filter { !$0.isSealed }

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
) -> ([MLXArray]) -> [MLXArray] {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless, f)

    return { arrays in
        compileState.call(arrays)
    }
}

/// Returns a compiled function that produces the same output as `f()`.
///
/// - Parameters:
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
    shapeless: Bool = false,
    _ f: @escaping @Sendable ([MLXArray]) -> [MLXArray]
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    let compileState = CompiledFunction(shapeless: shapeless, f)

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
) -> (MLXArray) -> MLXArray {
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

/// Overload of ``compile(shapeless:_:)-6mb2r`` that takes a single ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-7korq``
public func compile(
    shapeless: Bool = false,
    _ f: @escaping @Sendable (MLXArray) -> MLXArray
) -> @Sendable (MLXArray) -> MLXArray {
    let compileState = CompiledFunction(shapeless: shapeless) {
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
    -> (MLXArray, MLXArray) -> MLXArray
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0], $0[1])]
    }

    return { a, b in
        let r = compileState.call([a, b])
        return r.isEmpty ? MLXArray(0) : r[0]
    }
}

/// Overload of ``compile(shapeless:_:)-6mb2r`` that takes two ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-7korq``
public func compile(
    shapeless: Bool = false,
    _ f: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray
)
    -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    let compileState = CompiledFunction(shapeless: shapeless) {
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
    _ f: @escaping (MLXArray, MLXArray, MLXArray) -> MLXArray
)
    -> (MLXArray, MLXArray, MLXArray) -> MLXArray
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0], $0[1], $0[2])]
    }

    return { a, b, c in
        let r = compileState.call([a, b, c])
        return r.isEmpty ? MLXArray(0) : r[0]
    }
}

/// Overload of ``compile(shapeless:_:)-6mb2r`` that takes three ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-7korq``
public func compile(
    shapeless: Bool = false,
    _ f: @escaping @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray
)
    -> @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray
{
    let compileState = CompiledFunction(shapeless: shapeless) {
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
