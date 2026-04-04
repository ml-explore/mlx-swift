// Copyright © 2024 Apple Inc.

import Darwin
import Foundation
import MLX
import MLXNN

private enum DistributedWorkerOperation: String {
    case allSum
    case sendRecv
    case split
    case shardLinearForward
    case shardLinearBackward
    case averageGradients
}

enum DistributedWorkerRunner {
    static func main() {
        let environment = ProcessInfo.processInfo.environment

        guard let rankString = environment["MLX_RANK"], let rank = Int(rankString) else {
            fail("MLX_RANK not set")
        }
        guard environment["MLX_HOSTFILE"] != nil else {
            fail("MLX_HOSTFILE not set")
        }
        guard let rawOperation = environment["MLX_TEST_OP"],
            let operation = DistributedWorkerOperation(rawValue: rawOperation)
        else {
            fail("Unknown test operation: \(environment["MLX_TEST_OP"] ?? "<missing>")")
        }

        fputs("Worker rank=\(rank) starting operation=\(operation.rawValue)\n", stderr)

        // Distributed operations are CPU-only; keep the worker pinned to CPU.
        MLX.Device.withDefaultDevice(.cpu) {
            run(rank: rank, operation: operation)
        }
    }

    private static func run(rank: Int, operation: DistributedWorkerOperation) {
        guard let group = MLXDistributed.`init`(strict: true, backend: .ring) else {
            fail("Failed to initialize distributed group (strict=true)")
        }

        fputs(
            "Worker rank=\(rank) initialized: group.rank=\(group.rank) group.size=\(group.size)\n",
            stderr)

        guard group.rank == rank else {
            fail("group.rank (\(group.rank)) != expected rank (\(rank))")
        }
        guard group.size == 2 else {
            fail("group.size (\(group.size)) != 2")
        }

        switch operation {
        case .allSum:
            runAllSum(rank: rank, group: group)
        case .sendRecv:
            runSendRecv(rank: rank, group: group)
        case .split:
            runSplit(rank: rank, group: group)
        case .shardLinearForward:
            runShardLinearForward(rank: rank, group: group)
        case .shardLinearBackward:
            runShardLinearBackward(rank: rank, group: group)
        case .averageGradients:
            runAverageGradients(rank: rank, group: group)
        }

        finish(rank: rank)
    }
}

private func runAllSum(rank: Int, group: DistributedGroup) {
    let input =
        rank == 0
        ? MLXArray(converting: [1.0, 2.0, 3.0])
        : MLXArray(converting: [4.0, 5.0, 6.0])

    let result = MLXDistributed.allSum(input, group: group)
    eval(result)

    let values = result.asArray(Float.self)
    let expected: [Float] = [5.0, 7.0, 9.0]
    assertClose(values, expected, tolerance: 1e-5, context: "allSum")

    emitJSON([
        "shape": result.shape,
        "values": values.map(Double.init),
    ])
}

private func runSendRecv(rank: Int, group: DistributedGroup) {
    if rank == 0 {
        let data = MLXArray(converting: [10.0, 20.0, 30.0])
        let token = MLXDistributed.send(data, to: 1, group: group)
        eval(token)
        emitJSON(["sent": [10.0, 20.0, 30.0]])
        return
    }

    let received = MLXDistributed.recv(shape: [3], dtype: .float32, from: 0, group: group)
    eval(received)

    let values = received.asArray(Float.self)
    let expected: [Float] = [10.0, 20.0, 30.0]
    guard received.shape == [3] else {
        fail("recv shape mismatch: got \(received.shape), expected [3]")
    }
    assertClose(values, expected, tolerance: 1e-5, context: "sendRecv")

    emitJSON([
        "shape": received.shape,
        "values": values.map(Double.init),
    ])
}

private func runSplit(rank: Int, group: DistributedGroup) {
    var splitErrorCaught = false
    do {
        try withError {
            _ = group.split(color: 0, key: rank)
        }
    } catch {
        fputs("Worker rank=\(rank) split error (expected): \(error)\n", stderr)
        splitErrorCaught = true
    }

    if !splitErrorCaught {
        fputs("Worker rank=\(rank) split unexpectedly succeeded\n", stderr)
    }

    let input =
        rank == 0
        ? MLXArray(converting: [1.0, 2.0, 3.0])
        : MLXArray(converting: [4.0, 5.0, 6.0])

    let result = MLXDistributed.allSum(input, group: group)
    eval(result)

    let values = result.asArray(Float.self)
    let expected: [Float] = [5.0, 7.0, 9.0]
    assertClose(values, expected, tolerance: 1e-5, context: "split")

    emitJSON([
        "shape": result.shape,
        "splitErrorCaught": splitErrorCaught,
        "values": values.map(Double.init),
    ])
}

private func runShardLinearForward(rank: Int, group: DistributedGroup) {
    let count = group.size

    MLXRandom.seed(0xF0F0_F0F0)

    let x = MLXRandom.normal([4, 1024])
    let linear = Linear(1024, 1024, bias: true)
    eval(x, linear)

    let reference = linear(x)
    eval(reference)

    let allToSharded = shardLinear(
        module: linear, sharding: .allToSharded, group: group
    ) as! UnaryLayer
    let shardedToAll = shardLinear(
        module: linear, sharding: .shardedToAll, group: group
    ) as! UnaryLayer
    eval(allToSharded, shardedToAll)

    let shardedOutput = allToSharded(x)
    eval(shardedOutput)

    let columnStart = rank * 1024 / count
    let columnEnd = (rank + 1) * 1024 / count
    let shardedInput = x[0..., columnStart ..< columnEnd]
    eval(shardedInput)
    let fullOutput = shardedToAll(shardedInput)
    eval(fullOutput)

    let rowStart = rank * 1024 / count
    let rowEnd = (rank + 1) * 1024 / count
    let referenceShard = reference[0..., rowStart ..< rowEnd]
    eval(referenceShard)

    let allToShardedMatch = referenceShard.allClose(
        shardedOutput, rtol: 1e-4, atol: 1e-5
    ).item(Bool.self)
    let shardedToAllMatch = reference.allClose(
        fullOutput, rtol: 1e-4, atol: 1e-5
    ).item(Bool.self)

    if !allToShardedMatch {
        let diff = abs(referenceShard - shardedOutput).max().item(Float.self)
        fail("AllToSharded forward parity failed (max diff: \(diff))")
    }
    if !shardedToAllMatch {
        let diff = abs(reference - fullOutput).max().item(Float.self)
        fail("ShardedToAll forward parity failed (max diff: \(diff))")
    }

    emitJSON([
        "allToShardedMatch": allToShardedMatch,
        "shardedToAllMatch": shardedToAllMatch,
        "y1Shape": shardedOutput.shape,
        "y2Shape": fullOutput.shape,
    ])
}

private func runShardLinearBackward(rank: Int, group: DistributedGroup) {
    let count = group.size

    MLXRandom.seed(0xF0F0_F0F0)

    let model = Sequential(
        layers:
            Linear(128, 128, bias: true),
        Linear(128, 128, bias: true),
        Linear(128, 128, bias: true),
        Linear(128, 128, bias: true)
    )
    eval(model)

    let shardedModel = Sequential(
        layers:
            shardLinear(module: model.layers[0], sharding: .allToSharded, group: group) as! UnaryLayer,
        shardLinear(module: model.layers[1], sharding: .shardedToAll, group: group) as! UnaryLayer,
        shardLinear(module: model.layers[2], sharding: .allToSharded, group: group) as! UnaryLayer,
        shardLinear(module: model.layers[3], sharding: .shardedToAll, group: group) as! UnaryLayer
    )
    eval(shardedModel)

    let x = MLXRandom.normal([4, 128])
    let target = MLXRandom.normal([4, 128])
    eval(x, target)

    func loss(model: Sequential, x: MLXArray, y: MLXArray) -> MLXArray {
        (model(x) * y).sum()
    }

    let fullGrad = valueAndGrad(model: model, loss)
    let (fullLoss, fullGradients) = fullGrad(model, x, target)
    eval(fullLoss, fullGradients)

    let shardedGrad = valueAndGrad(model: shardedModel, loss)
    let (shardedLoss, shardedGradients) = shardedGrad(shardedModel, x, target)
    eval(shardedLoss, shardedGradients)

    let part = rank * 128 / count ..< (rank + 1) * 128 / count

    let lossMatch = fullLoss.allClose(shardedLoss).item(Bool.self)

    let fullFlat = Dictionary(uniqueKeysWithValues: fullGradients.flattened())
    let shardedFlat = Dictionary(uniqueKeysWithValues: shardedGradients.flattened())

    func full(_ key: String) -> MLXArray { fullFlat[key]! }
    func sharded(_ key: String) -> MLXArray { shardedFlat[key]! }

    let l0WeightMatch = full("layers.0.weight")[part].allClose(
        sharded("layers.0.weight"), rtol: 1e-4, atol: 1e-6
    ).item(Bool.self)
    let l0BiasMatch = full("layers.0.bias")[part].allClose(
        sharded("layers.0.bias"), rtol: 1e-4, atol: 1e-6
    ).item(Bool.self)
    let l1WeightMatch = full("layers.1.weight")[0..., part].allClose(
        sharded("layers.1.weight"), rtol: 1e-4, atol: 1e-6
    ).item(Bool.self)
    let l1BiasMatch = full("layers.1.bias").allClose(
        sharded("layers.1.bias"), rtol: 1e-4, atol: 1e-5
    ).item(Bool.self)
    let l2WeightMatch = full("layers.2.weight")[part].allClose(
        sharded("layers.2.weight"), rtol: 1e-4, atol: 1e-6
    ).item(Bool.self)
    let l2BiasMatch = full("layers.2.bias")[part].allClose(
        sharded("layers.2.bias"), rtol: 1e-4, atol: 1e-6
    ).item(Bool.self)
    let l3WeightMatch = full("layers.3.weight")[0..., part].allClose(
        sharded("layers.3.weight"), rtol: 1e-4, atol: 1e-6
    ).item(Bool.self)
    let l3BiasMatch = full("layers.3.bias").allClose(
        sharded("layers.3.bias"), rtol: 1e-4, atol: 1e-5
    ).item(Bool.self)

    let checks: [(String, Bool)] = [
        ("loss", lossMatch),
        ("layer0 weight", l0WeightMatch),
        ("layer0 bias", l0BiasMatch),
        ("layer1 weight", l1WeightMatch),
        ("layer1 bias", l1BiasMatch),
        ("layer2 weight", l2WeightMatch),
        ("layer2 bias", l2BiasMatch),
        ("layer3 weight", l3WeightMatch),
        ("layer3 bias", l3BiasMatch),
    ]
    for (name, passed) in checks where !passed {
        fail("\(name) gradient parity failed")
    }

    emitJSON([
        "l0BiasMatch": l0BiasMatch,
        "l0WeightMatch": l0WeightMatch,
        "l1BiasMatch": l1BiasMatch,
        "l1WeightMatch": l1WeightMatch,
        "l2BiasMatch": l2BiasMatch,
        "l2WeightMatch": l2WeightMatch,
        "l3BiasMatch": l3BiasMatch,
        "l3WeightMatch": l3WeightMatch,
        "lossMatch": lossMatch,
    ])
}

private func runAverageGradients(rank: Int, group: DistributedGroup) {
    let weight: MLXArray
    let bias: MLXArray
    if rank == 0 {
        weight = MLXArray(converting: [2.0, 4.0, 6.0])
        bias = MLXArray(converting: [10.0])
    } else {
        weight = MLXArray(converting: [4.0, 8.0, 12.0])
        bias = MLXArray(converting: [20.0])
    }
    eval(weight, bias)

    var gradients = ModuleParameters()
    gradients["weight"] = .value(weight)
    gradients["bias"] = .value(bias)

    let expectedWeight: [Float] = [3.0, 6.0, 9.0]
    let expectedBias: [Float] = [15.0]

    let defaultAverage = averageGradients(gradients: gradients, group: group)
    let defaultFlat = Dictionary(uniqueKeysWithValues: defaultAverage.flattened())
    let defaultWeight = defaultFlat["weight"]!.asArray(Float.self)
    let defaultBias = defaultFlat["bias"]!.asArray(Float.self)
    let defaultMatch =
        arraysClose(defaultWeight, expectedWeight, tolerance: 1e-4)
        && arraysClose(defaultBias, expectedBias, tolerance: 1e-4)

    let unbatchedAverage = averageGradients(
        gradients: gradients, group: group, allReduceSize: 0
    )
    let unbatchedFlat = Dictionary(uniqueKeysWithValues: unbatchedAverage.flattened())
    let unbatchedWeight = unbatchedFlat["weight"]!.asArray(Float.self)
    let unbatchedBias = unbatchedFlat["bias"]!.asArray(Float.self)
    let unbatchedMatch =
        arraysClose(unbatchedWeight, expectedWeight, tolerance: 1e-4)
        && arraysClose(unbatchedBias, expectedBias, tolerance: 1e-4)

    let communicationAverage = averageGradients(
        gradients: gradients, group: group, communicationType: .float16
    )
    let communicationFlat = Dictionary(uniqueKeysWithValues: communicationAverage.flattened())
    let communicationWeight = communicationFlat["weight"]!
    let communicationBias = communicationFlat["bias"]!
    let communicationMatch =
        arraysClose(communicationWeight.asArray(Float.self), expectedWeight, tolerance: 0.1)
        && arraysClose(communicationBias.asArray(Float.self), expectedBias, tolerance: 0.1)
    let communicationTypeDtype = String(describing: communicationWeight.dtype)

    let mixedFlat: [String: MLXArray] = [
        "weight_f32": MLXArray(rank == 0 ? [2.0, 4.0] as [Float] : [4.0, 8.0] as [Float]),
        "weight_f16": MLXArray(
            rank == 0 ? [10.0, 20.0] as [Float] : [30.0, 40.0] as [Float]
        ).asType(.float16),
    ]
    let mixedGradients = ModuleParameters.unflattened(mixedFlat)
    let mixedAverage = averageGradients(gradients: mixedGradients, group: group)
    eval(mixedAverage)

    let mixedResult = Dictionary(uniqueKeysWithValues: mixedAverage.flattened())
    let mixedF32 = mixedResult["weight_f32"]!
    let mixedF16 = mixedResult["weight_f16"]!
    let mixedDtypeMatch =
        arraysClose(mixedF32.asArray(Float.self), [3.0, 6.0], tolerance: 0.1)
        && arraysClose(mixedF16.asType(.float32).asArray(Float.self), [20.0, 30.0], tolerance: 1.0)
    let mixedDtypePreserved = mixedF16.dtype == .float16

    emitJSON([
        "commTypeDtype": communicationTypeDtype,
        "commTypeMatch": communicationMatch,
        "defaultMatch": defaultMatch,
        "mixedDtypeMatch": mixedDtypeMatch,
        "mixedDtypePreserved": mixedDtypePreserved,
        "unbatchedMatch": unbatchedMatch,
    ])
}

private func emitJSON(_ object: [String: Any]) {
    do {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
    } catch {
        fail("Failed to encode JSON: \(error)")
    }
}

private func assertClose(
    _ actual: [Float], _ expected: [Float], tolerance: Float, context: String
) {
    guard arraysClose(actual, expected, tolerance: tolerance) else {
        fail("\(context) mismatch: got \(actual), expected \(expected)")
    }
}

private func arraysClose(_ actual: [Float], _ expected: [Float], tolerance: Float) -> Bool {
    guard actual.count == expected.count else {
        return false
    }
    return zip(actual, expected).allSatisfy { abs($0 - $1) <= tolerance }
}

private func finish(rank: Int) -> Never {
    fputs("Worker rank=\(rank) completed successfully\n", stderr)
    fflush(stdout)
    fflush(stderr)
    _exit(0)
}

private func fail(_ message: String) -> Never {
    fputs("ERROR: \(message)\n", stderr)
    fflush(stderr)
    exit(1)
}
