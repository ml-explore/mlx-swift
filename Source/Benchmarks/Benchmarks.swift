// Copyright © 2026 Apple Inc.

import Dispatch
import Foundation
import MLX
import MLXNN

private struct BenchmarkResult {
    let name: String
    let iterations: Int
    let minMs: Double
    let medianMs: Double
    let meanMs: Double
}

private func measure(
    name: String, warmup: Int = 5, iterations: Int = 100, _ body: () -> Void
) -> BenchmarkResult {
    for _ in 0 ..< warmup { body() }

    var samples = [Double]()
    samples.reserveCapacity(iterations)
    for _ in 0 ..< iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        body()
        let end = DispatchTime.now().uptimeNanoseconds
        samples.append(Double(end - start) / 1_000_000)
    }
    samples.sort()

    return BenchmarkResult(
        name: name,
        iterations: iterations,
        minMs: samples.first!,
        medianMs: samples[samples.count / 2],
        meanMs: samples.reduce(0, +) / Double(iterations))
}

private func printResults(_ results: [BenchmarkResult]) {
    print("Benchmark                                      Iters     Min ms  Median ms    Mean ms")
    for result in results {
        let name = result.name.padding(toLength: 44, withPad: " ", startingAt: 0)
        let values = String(
            format: "%6d %10.4f %10.4f %10.4f", result.iterations, result.minMs,
            result.medianMs, result.meanMs)
        print("\(name) \(values)")
    }
}

private func graphConstruction() -> BenchmarkResult {
    measure(name: "Graph construction (200 elementwise ops)") {
        var x = MLXArray.zeros([4, 4])
        for _ in 0 ..< 200 { x = x + 1 }
        precondition(x.shape == [4, 4])
    }
}

private func realizedEval() -> BenchmarkResult {
    let x = MLXArray.ones([4, 4])
    eval(x)
    return measure(name: "eval() of an already-realized array") { eval(x) }
}

private func mlpForward() -> [BenchmarkResult] {
    let l1 = Linear(128, 256)
    let l2 = Linear(256, 256)
    let l3 = Linear(256, 10)
    let input = MLXRandom.normal([32, 128])

    func forward(_ x: MLXArray) -> MLXArray {
        l3(relu(l2(relu(l1(x)))))
    }

    let compiled = compile(inputs: [l1, l2, l3], forward)
    let expected = forward(input)
    let actual = compiled(input)
    eval(expected, actual)
    precondition(allClose(expected, actual).item(Bool.self))

    return [
        measure(name: "MLP forward + eval (uncompiled)") { eval(forward(input)) },
        measure(name: "MLP forward + eval (compiled)") { eval(compiled(input)) },
    ]
}

private func evalContention(device: Device) -> [BenchmarkResult] {
    let workerCount = 4
    let workPerWorker = 8
    let streams = (0 ..< workerCount).map { _ in Stream(device) }

    let runWorker: @Sendable (Int) -> Void = { worker in
        let stream = StreamOrDevice.stream(streams[worker])
        var x = full([64, 64], values: Float(worker), stream: stream)
        for _ in 0 ..< workPerWorker {
            x = add(x, 1, stream: stream)
            x = multiply(x, 1.0001, stream: stream)
        }
        eval(x)
    }

    let sequential = measure(name: "4 graph + eval workers (sequential)", iterations: 25) {
        for worker in 0 ..< workerCount { runWorker(worker) }
    }
    let concurrent = measure(name: "4 graph + eval workers (concurrent)", iterations: 25) {
        DispatchQueue.concurrentPerform(iterations: workerCount, execute: runWorker)
    }
    return [sequential, concurrent]
}

@main
private struct Benchmarks {
    static func main() {
        let requested = CommandLine.arguments.dropFirst().first
        let device: Device
        switch requested {
        case nil: device = Device.defaultDevice()
        case "gpu": device = .gpu
        case "cpu": device = .cpu
        default:
            fatalError("usage: Benchmarks [cpu|gpu]")
        }

        Device.withDefaultDevice(device) {
            var results = [graphConstruction(), realizedEval()]
            results.append(contentsOf: mlpForward())
            results.append(contentsOf: evalContention(device: device))
            printResults(results)
        }
    }
}
