// Copyright © 2026 Apple Inc.

import Cmlx

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray) -> (MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0])
        return [r.0, r.1]
    }
    return { a in
        let r = compileState.call([a])
        return (r[0], r[1])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray) -> (MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0])
        return [r.0, r.1, r.2]
    }
    return { a in
        let r = compileState.call([a])
        return (r[0], r[1], r[2])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0])
        return [r.0, r.1, r.2, r.3]
    }
    return { a in
        let r = compileState.call([a])
        return (r[0], r[1], r[2], r[3])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1])
        return [r.0, r.1]
    }
    return { a, b in
        let r = compileState.call([a, b])
        return (r[0], r[1])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1])
        return [r.0, r.1, r.2]
    }
    return { a, b in
        let r = compileState.call([a, b])
        return (r[0], r[1], r[2])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1])
        return [r.0, r.1, r.2, r.3]
    }
    return { a, b in
        let r = compileState.call([a, b])
        return (r[0], r[1], r[2], r[3])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2])
        return [r.0, r.1]
    }
    return { a, b, c in
        let r = compileState.call([a, b, c])
        return (r[0], r[1])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2])
        return [r.0, r.1, r.2]
    }
    return { a, b, c in
        let r = compileState.call([a, b, c])
        return (r[0], r[1], r[2])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2])
        return [r.0, r.1, r.2, r.3]
    }
    return { a, b, c in
        let r = compileState.call([a, b, c])
        return (r[0], r[1], r[2], r[3])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        [f(args[0], args[1], args[2], args[3])]
    }
    return { a, b, c, d in
        compileState.call([a, b, c, d])[0]
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3])
        return [r.0, r.1]
    }
    return { a, b, c, d in
        let r = compileState.call([a, b, c, d])
        return (r[0], r[1])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3])
        return [r.0, r.1, r.2]
    }
    return { a, b, c, d in
        let r = compileState.call([a, b, c, d])
        return (r[0], r[1], r[2])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3])
        return [r.0, r.1, r.2, r.3]
    }
    return { a, b, c, d in
        let r = compileState.call([a, b, c, d])
        return (r[0], r[1], r[2], r[3])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        [f(args[0], args[1], args[2], args[3], args[4])]
    }
    return { a, b, c, d, e in
        compileState.call([a, b, c, d, e])[0]
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4])
        return [r.0, r.1]
    }
    return { a, b, c, d, e in
        let r = compileState.call([a, b, c, d, e])
        return (r[0], r[1])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4])
        return [r.0, r.1, r.2]
    }
    return { a, b, c, d, e in
        let r = compileState.call([a, b, c, d, e])
        return (r[0], r[1], r[2])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray, MLXArray
    )
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4])
        return [r.0, r.1, r.2, r.3]
    }
    return { a, b, c, d, e in
        let r = compileState.call([a, b, c, d, e])
        return (r[0], r[1], r[2], r[3])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        [f(args[0], args[1], args[2], args[3], args[4], args[5])]
    }
    return { a, b, c, d, e, g in
        compileState.call([a, b, c, d, e, g])[0]
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5])
        return [r.0, r.1]
    }
    return { a, b, c, d, e, g in
        let r = compileState.call([a, b, c, d, e, g])
        return (r[0], r[1])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    )
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5])
        return [r.0, r.1, r.2]
    }
    return { a, b, c, d, e, g in
        let r = compileState.call([a, b, c, d, e, g])
        return (r[0], r[1], r[2])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray, MLXArray
    )
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5])
        return [r.0, r.1, r.2, r.3]
    }
    return { a, b, c, d, e, g in
        let r = compileState.call([a, b, c, d, e, g])
        return (r[0], r[1], r[2], r[3])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        [f(args[0], args[1], args[2], args[3], args[4], args[5], args[6])]
    }
    return { a, b, c, d, e, g, h in
        compileState.call([a, b, c, d, e, g, h])[0]
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray
    )
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        return [r.0, r.1]
    }
    return { a, b, c, d, e, g, h in
        let r = compileState.call([a, b, c, d, e, g, h])
        return (r[0], r[1])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    )
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        return [r.0, r.1, r.2]
    }
    return { a, b, c, d, e, g, h in
        let r = compileState.call([a, b, c, d, e, g, h])
        return (r[0], r[1], r[2])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray, MLXArray
    )
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        return [r.0, r.1, r.2, r.3]
    }
    return { a, b, c, d, e, g, h in
        let r = compileState.call([a, b, c, d, e, g, h])
        return (r[0], r[1], r[2], r[3])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
        -> MLXArray
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
    MLXArray
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        [f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])]
    }
    return { a, b, c, d, e, g, h, i in
        compileState.call([a, b, c, d, e, g, h, i])[0]
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
        -> (MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
    (MLXArray, MLXArray)
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
        return [r.0, r.1]
    }
    return { a, b, c, d, e, g, h, i in
        let r = compileState.call([a, b, c, d, e, g, h, i])
        return (r[0], r[1])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
        -> (MLXArray, MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
    (MLXArray, MLXArray, MLXArray)
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
        return [r.0, r.1, r.2]
    }
    return { a, b, c, d, e, g, h, i in
        let r = compileState.call([a, b, c, d, e, g, h, i])
        return (r[0], r[1], r[2])
    }
}

@_documentation(visibility: internal)
public func compile(
    inputs: [any Updatable] = [], outputs: [any Updatable] = [], shapeless: Bool = false,
    _ f:
        @escaping (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
        -> (MLXArray, MLXArray, MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
    (MLXArray, MLXArray, MLXArray, MLXArray)
{
    let compileState = CompiledFunction(inputs: inputs, outputs: outputs, shapeless: shapeless) {
        args in
        let r = f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])
        return [r.0, r.1, r.2, r.3]
    }
    return { a, b, c, d, e, g, h, i in
        let r = compileState.call([a, b, c, d, e, g, h, i])
        return (r[0], r[1], r[2], r[3])
    }
}
