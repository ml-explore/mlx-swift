// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation

class CompiledFunction {

    /// unique (for the lifetime of the object) identifier for the compiled function
    var id: Int!

    /// the function to compile
    let f: ([MLXArray]) -> [MLXArray]

    /// any state to be observed
    let inputs: [any Updatable]
    let outputs: [any Updatable]

    let shapeless: Bool

    init(
        inputs: [any Updatable], outputs: [any Updatable], shapeless: Bool,
        _ f: @escaping ([MLXArray]) -> [MLXArray]
    ) {
        self.f = f
        self.inputs = inputs
        self.outputs = outputs
        self.shapeless = shapeless
        self.id = Int(bitPattern: Unmanaged.passUnretained(self).toOpaque())
    }

    deinit {
        // remove the compiled structure from the back end
        mlx_detail_compile_erase(id)
    }

    func call(_ arguments: [MLXArray]) -> [MLXArray] {
        let stateInputs = inputs.flatMap { $0.innerState() }
        let argumentsCount = arguments.count

        // inner function to hande the compilation.  this is called
        // once per compile (typically once overall, but can be called
        // again if the conditions for recompile change)
        func inner(tracers: [MLXArray]) -> [MLXArray] {

            // put the tracers in their appropriate places:
            // - arguments to the function
            // - inner state

            let tracerArguments = Array(tracers.prefix(argumentsCount))

            // save a snapshot of the inner state
            let savedStateInputs = stateInputs.map { $0.copy() }

            // replace the inner state with the tracers
            for (s, tracer) in zip(stateInputs, tracers[argumentsCount...]) {
                s.update(tracer)
            }

            // call the function with the tracer arguments
            // and the state holding tracers
            let result = f(tracerArguments)

            // recapture the state as it may have changed
            let stateOutputTracers = outputs.flatMap { $0.innerState() }.map { $0.copy() }

            // put the original values back in the state
            for (s, saved) in zip(stateInputs, savedStateInputs) {
                s.update(saved)
            }

            // return the result of the function and the state
            return result + stateOutputTracers
        }

        let innerClosure = new_mlx_closure(inner(tracers:))
        defer { mlx_free(innerClosure) }

        // note: this will use the cached compile (via the id)
        // but will be able to re-evaluate with fresh state if needed
        let compiled = mlx_detail_compile(innerClosure, id, shapeless, [], 0)!
        defer { mlx_free(compiled) }

        let innerInputs = arguments + stateInputs
        let innerInputsVector = new_mlx_vector_array(innerInputs)
        defer { mlx_free(innerInputsVector) }

        // will compile the function (if needed) and evaluate the
        // compiled graph
        let resultVector = mlx_closure_apply(compiled, innerInputsVector)!
        defer { mlx_free(resultVector) }

        let resultsPlusStateOutput = mlx_vector_array_values(resultVector)

        // push the stateOutput into the state
        let stateOutput = outputs.flatMap { $0.innerState() }

        for (s, newValues) in zip(stateOutput, resultsPlusStateOutput.suffix(stateOutput.count)) {
            s.update(newValues)
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
/// - ``compile(inputs:outputs:shapeless:_:)-29n3k``
/// - ``compile(inputs:outputs:shapeless:_:)-4msdm``
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> (
    [MLXArray]
) -> [MLXArray] {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless, f)

    return { arrays in
        compileState.call(arrays)
    }
}

/// Overload of ``compile(inputs:outputs:shapeless:_:)-7korq`` that takes a single ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-7korq``
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray) -> MLXArray
) -> (
    MLXArray
) -> MLXArray {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        [f($0[0])]
    }

    return { a in
        compileState.call([a])[0]
    }
}

/// Overload of ``compile(inputs:outputs:shapeless:_:)-7korq`` that takes a two ``MLXArray`` and
/// produces a single ``MLXArray``.
///
/// ### See Also
/// - <doc:compilation>
/// - ``compile(inputs:outputs:shapeless:_:)-7korq``
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
        compileState.call([a, b])[0]
    }
}

/// Globally enable or disable ``compile(inputs:outputs:shapeless:_:)-7korq``.
///
/// Default is enabled.
public func compile(enable: Bool = true) {
    if enable {
        mlx_enable_compile()
    } else {
        mlx_disable_compile()
    }
}
