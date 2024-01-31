// Copyright © 2024 Apple Inc.

import Foundation

@main
struct GenerateGrad {

    /// up to how many MLXArray tuples should we generate, e.g. 3 == `MLXArray, MLXArray, MLXArray`
    static let inputTupleCount = 1
    static let outputTupleCount = 1

    static func indentLines(_ text: String, lead: String) -> String {
        lead
            + text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .joined(separator: "\n\(lead)")
    }

    struct MethodInfo {
        let methodName: String
        let methodDescription: String
        let internalDocumentation: String
        let seeAlso: String
        let arguments: (String, String) -> String
        let returnValue: (String, String) -> String
        let body: (String) -> String
    }

    static let methodInfo = [
        "grad": MethodInfo(
            methodName: "grad",
            methodDescription: "Returns a function which computes the gradient of `f`.",
            internalDocumentation:
                """
                Converts the given function `f()` into canonical types, e.g.
                (MLXArray) -> MLXArray into the canonical form ([MLXArray]) -> [MLXArray].

                First use the wrapArguments() and wrapResult() function to transform
                it into that form.  Then call buildValueAndGradient() to produce a new
                function with the same canonical form.

                Finally use unwrapArguments() and unwrapResult() to transform the function
                back into the original signature.

                Note: this particular form of the function is already in the canonical
                form and the wrap/unwrap calls are identity functions.
                """,
            seeAlso: "See ``grad(_:)-r8dv``",
            arguments: { input, returnValue in
                if input == "MLXArray" {
                    return "(_ f: @escaping (\(input)) -> \(returnValue))"
                } else {
                    return
                        "(_ f: @escaping (\(input)) -> \(returnValue), argumentNumbers: [Int] = [0])"
                }
            },
            returnValue: { input, returnValue in
                "(\(input)) -> \(returnValue)"
            },
            body: { input in
                let argumentNumbersUse: String
                if input == "MLXArray" {
                    argumentNumbersUse = "[0]"
                } else {
                    argumentNumbersUse = "argumentNumbers"
                }
                return
                    """
                    let wrappedFunction = wrapResult(wrapArguments(f))
                    let gradientFunction = buildGradient(wrappedFunction, argumentNumbers: \(argumentNumbersUse))
                    let uag: (\(input)) -> [MLXArray] = unwrapArguments(gradientFunction)
                    return unwrapResult(uag)
                    """
            }
        ),
        "valueAndGrad": MethodInfo(
            methodName: "valueAndGrad",
            methodDescription: "Returns a function which computes the value and gradient of `f`.",
            internalDocumentation: "",
            seeAlso: "See ``valueAndGrad(_:)``",
            arguments: { input, returnValue in
                "(_ f: @escaping (\(input)) -> \(returnValue), argumentNumbers: [Int] = [0])"
            },
            returnValue: { input, returnValue in
                "(\(input)) -> (\(returnValue), \(returnValue))"
            },
            body: { input in
                """
                return buildValueAndGradient(f, argumentNumbers: argumentNumbers)
                """
            }
        ),
    ]

    static func emitFunction(name: String, input: String, output: String) -> String {
        var result = ""

        let info = methodInfo[name]!

        let firstMethod = input == "[MLXArray]" && output == "[MLXArray]"
        let documentationText = firstMethod ? info.methodDescription : info.seeAlso

        result += indentLines(documentationText, lead: "// ")
        result += "\n"

        let returnValue: String
        if output.contains(",") {
            returnValue = "(\(output))"
        } else {
            returnValue = output
        }

        result +=
            """
            public func \(name)\(info.arguments(input, returnValue)) -> \(info.returnValue(input, returnValue)) {

            """

        if firstMethod {
            result += indentLines(info.internalDocumentation, lead: "    // ")
            result += "\n"
        }

        result += indentLines(info.body(input), lead: "    ")
        result += "\n}\n"

        return result
    }

    /// Tool to generate `Transforms+Variants.swift`.
    ///
    /// Either:
    /// - run this and paste the output into `Transforms+Grad.swift`
    /// - or `swift run GenerateGrad > Sources/MLX/Transforms+Grad.swift`
    static func main() {
        print(
            """
            import Foundation
            import Cmlx

            // This file is generated by GenerateGrad.

            """
        )

        // emit the `grad()` variants -- these are the public functions that can be called.
        // we emit a variant for each combination of inputs and outputs below

        let baseTypes = [
            "[MLXArray]",
            "MLXArray",
        ]
        var inputs = baseTypes
        var outputs = baseTypes

        for i in 2 ..< (inputTupleCount + 1) {
            inputs.append(Array(repeating: "MLXArray", count: i).joined(separator: ", "))
        }
        for i in 2 ..< (outputTupleCount + 1) {
            outputs.append(Array(repeating: "MLXArray", count: i).joined(separator: ", "))
        }

        for input in inputs {
            for output in outputs {
                print(emitFunction(name: "grad", input: input, output: output))
            }
        }

        print(emitFunction(name: "valueAndGrad", input: "[MLXArray]", output: "[MLXArray]"))

        // functions for converting to and from canonical types.  For example this function:
        //
        // public func grad(_ f: @escaping (MLXArray, MLXArray) -> [MLXArray]) -> (MLXArray, MLXArray) -> [MLXArray] {
        //
        // takes a (MLXArray, MLXArray) -> [MLXArray].  We need to convert that to ([MLXArray]) -> [MLXArray]
        // and can compute the gradient:
        //
        //     let gradientFunction = buildValueAndGradient(wrapResult(wrapArguments(f)))
        //
        // the result is ([MLXArray]) -> [MLXArray] and we need to convert that back to
        // (MLXArray, MLXArray) -> [MLXArray]:
        //
        //     let uag: (MLXArray, MLXArray) -> [MLXArray] = unwrapArguments(gradientFunction)
        //     return unwrapResult(uag)
        //
        // These are all the different wrap/unwrap functions.

        // these are the special cases (NOP and single element tuple)
        print(
            """

            // MARK: - Functions to wrap and unwrap types in closures

            @inline(__always)
            private func wrapArguments<Result>(_ f: @escaping ([MLXArray]) -> Result) -> ([MLXArray]) -> Result {
                f
            }

            @inline(__always)
            private func wrapResult(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
                f
            }

            @inline(__always)
            private func wrapResult(_ f: @escaping ([MLXArray]) -> MLXArray) -> ([MLXArray]) -> [MLXArray] {
                { (arrays: [MLXArray]) in
                    [f(arrays)]
                }
            }

            @inline(__always)
            private func unwrapArguments(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
                f
            }

            @inline(__always)
            private func unwrapResult(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
                f
            }

            """
        )

        for c in 1 ..< (inputTupleCount + 1) {
            let args = Array(repeating: "MLXArray", count: c).joined(separator: ", ")
            let wrapArguments = (0 ..< c).map { "arrays[\($0)]" }.joined(separator: ", ")

            print(
                """
                @inline(__always)
                private func wrapArguments<Result>(_ f: @escaping (\(args)) -> Result) -> ([MLXArray]) -> Result {
                    { (arrays: [MLXArray]) in
                        f(\(wrapArguments))
                    }
                }

                """
            )
        }

        // note: from 2 since we have the 1 special case above
        for c in 2 ..< (inputTupleCount + 1) {
            let args = Array(repeating: "MLXArray", count: c).joined(separator: ", ")
            let wrapResult = (0 ..< c).map { "v.\($0)" }.joined(separator: ", ")

            print(
                """
                @inline(__always)
                private func wrapResult(_ f: @escaping ([MLXArray]) -> (\(args))) -> ([MLXArray]) -> [MLXArray] {
                    { (arrays: [MLXArray]) in
                        let v = f(arrays)
                        return [\(wrapResult)]
                    }
                }

                """
            )
        }

        for c in 1 ..< (inputTupleCount + 1) {
            let args = Array(repeating: "MLXArray", count: c).joined(separator: ", ")
            let unrwapArguments1 = (0 ..< c).map { "a\($0): MLXArray" }.joined(separator: ", ")
            let unrwapArguments2 = (0 ..< c).map { "a\($0)" }.joined(separator: ", ")
            print(
                """
                @inline(__always)
                private func unwrapArguments(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> (\(args)) -> [MLXArray] {
                    { (\(unrwapArguments1)) in
                        f([\(unrwapArguments2)])
                    }
                }

                """
            )
        }

        // unwrapResult is a little more complicated because we have to handle all the
        // input/output pairs.

        // [MLXArray] -> (MLXArray...)
        for c in 1 ..< (inputTupleCount + 1) {
            let args = Array(repeating: "MLXArray", count: c).joined(separator: ", ")
            let unwrapResult = (0 ..< c).map { "v[\($0)]" }.joined(separator: ", ")

            print(
                """
                @inline(__always)
                private func unwrapResult(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> (\(args)) {
                    { (a0: [MLXArray]) in
                        let v = f(a0)
                        return (\(unwrapResult))
                    }
                }

                """
            )
        }

        // (MLXArray...) -> ([MLXArray])
        for c in 1 ..< (inputTupleCount + 1) {
            let args = Array(repeating: "MLXArray", count: c).joined(separator: ", ")
            let unwrapInputs1 = (0 ..< c).map { "a\($0): MLXArray" }.joined(separator: ", ")
            let unwrapInputs2 = (0 ..< c).map { "a\($0)" }.joined(separator: ", ")

            print(
                """
                @inline(__always)
                private func unwrapResult(_ f: @escaping (\(args)) -> [MLXArray]) -> (\(args)) -> [MLXArray] {
                    { (\(unwrapInputs1)) in
                        f(\(unwrapInputs2))
                    }
                }

                """
            )
        }

        // (MLXArray...) -> (MLXArray...)
        for c in 1 ..< (inputTupleCount + 1) {
            let output = Array(repeating: "MLXArray", count: c).joined(separator: ", ")
            let unwrapResult = (0 ..< c).map { "v[\($0)]" }.joined(separator: ", ")

            for argc in 1 ..< (inputTupleCount + 1) {
                let inputArgs = Array(repeating: "MLXArray", count: argc).joined(separator: ", ")
                let unwrapInputs1 = (0 ..< argc).map { "a\($0): MLXArray" }.joined(separator: ", ")
                let unwrapInputs2 = (0 ..< argc).map { "a\($0)" }.joined(separator: ", ")

                print(
                    """
                    @inline(__always)
                    private func unwrapResult(_ f: @escaping (\(inputArgs)) -> [MLXArray]) -> (\(inputArgs)) -> (\(output)) {
                        { (\(unwrapInputs1)) in
                            let v = f(\(unwrapInputs2))
                            return (\(unwrapResult))
                        }
                    }

                    """
                )
            }
        }
    }

}
