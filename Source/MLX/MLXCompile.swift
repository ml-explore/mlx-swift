// Copyright © 2024-2025 Apple Inc.

/// Macro that produces a compiled function from a typed closure.
///
/// Use the freestanding form to create a compiled function directly:
///
/// ```swift
/// let square = #MLXCompile({ (x: MLXArray) -> MLXArray in x * x })
///
/// let add = #MLXCompile({ (a: MLXArray, b: MLXArray) -> MLXArray in a + b })
/// ```
///
/// The closure must have explicit parameter types and return type.
/// Supported return types: `MLXArray`, tuples of `MLXArray`, or `[MLXArray]`.
///
/// Compile parameters can be passed before the closure:
///
/// ```swift
/// let forward = #MLXCompile(inputs: [model], outputs: [model], {
///     (x: MLXArray) -> MLXArray in model(x)
/// })
/// ```
///
/// The attached form can also be used to create a named function:
///
/// ```swift
/// @MLXCompile({ (x: MLXArray) -> MLXArray in x * x })
/// func square(x: MLXArray) -> MLXArray
/// ```
///
/// ### See Also
/// - ``compile(inputs:outputs:shapeless:_:)-([Updatable],[Updatable],Bool,([MLXArray])->[MLXArray])``
/// - <doc:compilation>

// MARK: - Freestanding form: let f = #MLXCompile({ ... })

@freestanding(expression)
public macro MLXCompile<T>(
    inputs: [any Updatable] = [],
    outputs: [any Updatable] = [],
    shapeless: Bool = false,
    _ body: T
) -> T = #externalMacro(module: "MLXMacros", type: "MLXCompileMacro")

// MARK: - Attached form: @MLXCompile({ ... }) func f(...)

@attached(peer, names: prefixed(_mlxc_))
@attached(body)
public macro MLXCompile(
    inputs: [any Updatable] = [],
    outputs: [any Updatable] = [],
    shapeless: Bool = false,
    _ body: Any
) = #externalMacro(module: "MLXMacros", type: "MLXCompileMacro")

// MARK: - MLXValueAndGrad

/// Macro that produces a value-and-gradient function from a typed loss closure.
///
/// Use the freestanding form to create a value-and-gradient function directly:
///
/// ```swift
/// let lg = #MLXValueAndGrad(model: m, { (model: M, x: MLXArray, y: MLXArray) -> MLXArray in
///     crossEntropy(logits: model(x), targets: y, reduction: .mean)
/// })
/// // lg: (M, MLXArray, MLXArray) -> (MLXArray, ModuleParameters)
///
/// let (loss, grads) = lg(m, x, y)
/// ```
///
/// The closure must have explicit parameter types and return type.
/// The first parameter is the model (any `Module`-conforming type).
/// Remaining parameters must all be `MLXArray`.
/// Supported return types: `MLXArray` or `[MLXArray]`.
///
/// The attached form can also be used:
///
/// ```swift
/// @MLXValueAndGrad(model: m, { (model: M, x: MLXArray, y: MLXArray) -> MLXArray in
///     crossEntropy(logits: model(x), targets: y, reduction: .mean)
/// })
/// func lossGrad(model: M, x: MLXArray, y: MLXArray) -> (MLXArray, ModuleParameters)
/// ```
///
/// ### See Also
/// - ``valueAndGrad(model:_:)-548r7``
/// - <doc:training>

// MARK: Freestanding form: let f = #MLXValueAndGrad(model: m, { ... })

@freestanding(expression)
public macro MLXValueAndGrad<T, R>(model: Any, _ body: T) -> R =
    #externalMacro(module: "MLXMacros", type: "MLXValueAndGradMacro")

// MARK: Attached form: @MLXValueAndGrad(model: m, { ... }) func f(...)

@attached(peer, names: prefixed(_mlxvg_))
@attached(body)
public macro MLXValueAndGrad(model: Any, _ body: Any) =
    #externalMacro(module: "MLXMacros", type: "MLXValueAndGradMacro")
