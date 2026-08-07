// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

struct BenchmarkResult {
    let name: String
    let iterations: Int
    let minMs: Double
    let medianMs: Double
    let meanMs: Double
}

func measure(
    name: String, warmup: Int = 5, iterations: Int = 100, _ body: () -> Void
) -> BenchmarkResult {
    for _ in 0 ..< warmup { body() }

    var samplesMs: [Double] = []
    samplesMs.reserveCapacity(iterations)
    for _ in 0 ..< iterations {
        let start = DispatchTime.now()
        body()
        let end = DispatchTime.now()
        samplesMs.append(Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
    }

    samplesMs.sort()
    return BenchmarkResult(
        name: name, iterations: iterations,
        minMs: samplesMs.first!,
        medianMs: samplesMs[samplesMs.count / 2],
        meanMs: samplesMs.reduce(0, +) / Double(iterations))
}

func pad(_ s: String, _ width: Int) -> String {
    s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
}

func padLeft(_ s: String, _ width: Int) -> String {
    s.count >= width ? s : String(repeating: " ", count: width - s.count) + s
}

func printResults(_ results: [BenchmarkResult]) {
    let nameWidth = max(30, (results.map { $0.name.count }.max() ?? 0) + 2)
    print(
        pad("Benchmark", nameWidth) + padLeft("Iters", 8) + padLeft("Min(ms)", 12)
            + padLeft("Median(ms)", 12) + padLeft("Mean(ms)", 12))
    for r in results {
        print(
            pad(r.name, nameWidth) + padLeft("\(r.iterations)", 8)
                + padLeft(String(format: "%.4f", r.minMs), 12)
                + padLeft(String(format: "%.4f", r.medianMs), 12)
                + padLeft(String(format: "%.4f", r.meanMs), 12))
    }
}

/// Case 1: graph construction only, no eval() -- isolates per-op MLXArray
/// allocation/ARC overhead from any real compute.
func benchmarkGraphConstruction() -> BenchmarkResult {
    let chainLength = 200

    // sanity check: confirm the chain actually runs end-to-end and produces
    // the expected shape, so a bug here isn't mistaken for a real finding.
    var sanity = full([4, 4], values: Float(0))
    for _ in 0 ..< chainLength {
        sanity = sanity + 1
    }
    precondition(sanity.shape == [4, 4], "graph construction sanity check failed")

    return measure(name: "Graph construction (\(chainLength) ops, no eval)") {
        var x = full([4, 4], values: Float(0))
        for _ in 0 ..< chainLength {
            x = x + 1
        }
    }
}

/// Case 2: eval() in isolation on an already-realized array -- isolates
/// evalLock + Cmlx+Util.swift marshalling cost from any real compute, since
/// re-evaluating a realized array does effectively no C++-side work.
func benchmarkEvalOverhead() -> BenchmarkResult {
    let x = full([4, 4], values: Float(1))
    eval(x)

    // sanity check: confirm the array is already realized before timing --
    // re-eval should be a no-op at the compute level.
    precondition(x.shape == [4, 4], "eval overhead sanity check failed")

    return measure(name: "eval() overhead (already-realized array)") {
        eval(x)
    }
}

/// Case 3: a small MLP forward pass, compiled vs. uncompiled -- the one case
/// that quantifies what compile() actually saves on a realistic-shaped
/// workload, rather than reporting an abstract overhead number with no real
/// compute alongside it to contextualize significance.
func benchmarkMLPForwardPass() -> [BenchmarkResult] {
    let layer1 = Linear(128, 256)
    let layer2 = Linear(256, 256)
    let layer3 = Linear(256, 10)

    func forward(_ x: MLXArray) -> MLXArray {
        var y = relu(layer1(x))
        y = relu(layer2(y))
        return layer3(y)
    }

    let input = MLXRandom.normal([32, 128])

    // Linear's weights are captured state, not a function argument -- compile()
    // must be told about them via `inputs:` or it bakes in trace-time placeholder
    // values instead of the real weights (Module conforms to Updatable for exactly
    // this purpose; see <doc:compilation> "Compiling Training Graphs").
    let compiledForward = compile(inputs: [layer1, layer2, layer3], forward)

    // sanity check: compiled and uncompiled paths must produce numerically
    // close outputs, or the comparison below is meaningless.
    let uncompiledOutput = forward(input)
    let compiledOutput = compiledForward(input)
    eval(uncompiledOutput, compiledOutput)
    precondition(
        allClose(uncompiledOutput, compiledOutput).item(Bool.self),
        "compiled and uncompiled MLP outputs diverged")

    let uncompiledResult = measure(name: "MLP forward + eval (uncompiled)") {
        eval(forward(input))
    }
    let compiledResult = measure(name: "MLP forward + eval (compiled)") {
        eval(compiledForward(input))
    }

    return [uncompiledResult, compiledResult]
}

@main
struct Benchmarks {
    static func main() {
        func getDeviceFromArgs() -> Device? {
            guard let index = CommandLine.arguments.firstIndex(of: "--device") else {
                return nil
            }

            let valueIndex = index + 1
            guard valueIndex < CommandLine.arguments.count else {
                print("Error: Missing value for option '--device'.")
                exit(1)
            }

            let value = CommandLine.arguments[valueIndex]
            switch value.lowercased() {
            case "cpu":
                return .cpu
            case "gpu":
                return .gpu
            default:
                print("Error: Invalid device: '\(value)'. Please use 'cpu' or 'gpu'.")
                exit(1)
            }
        }

        let specifiedDevice = getDeviceFromArgs()

        let defaultDevice: Device
        #if os(Linux)
            defaultDevice = .cpu
        #else
            defaultDevice = .gpu
        #endif

        let selectedDevice = specifiedDevice ?? defaultDevice

        print("Using device: \(selectedDevice).")

        Stream.withNewDefaultStream(device: selectedDevice) {
            var results: [BenchmarkResult] = []
            results.append(benchmarkGraphConstruction())
            results.append(benchmarkEvalOverhead())
            results.append(contentsOf: benchmarkMLPForwardPass())

            print("")
            printResults(results)
        }
    }
}
