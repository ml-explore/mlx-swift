// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@Suite
struct MaterializedTests {
    @Test
    func testMaterialize() async {
        // a materialized array is Sendable
        let x = MLXArray(10) + 5
        let m = materialize(x)
        let t = Task {
            print(m + 3)
        }
        _ = await t.result
    }

    @Test
    func testCompileMaterialized() async {
        // compile manipulates the input arrays (tracers)
        // make sure MaterializedArray doesn't run into problems here
        func f(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            square(a * b)
        }

        let c = compile(f)

        let i1 = MLXRandom.normal([20, 20]).materialized()
        let i2 = MLXRandom.normal([20, 20]).materialized()

        let t = Task {
            let s = sum(c(i1, i2))
            let s2 = sum(f(i1, i2))
            #expect(s.allClose(s2).item(Bool.self))
            print(s, s2)
        }
        _ = await t.result
    }

    @Test
    func testMaterializedLinear() async {
        let l = Linear(10, 10)
        let lm = MaterializedModule(l)

        // this will have been materialized in the call
        #expect(l.weight is MaterializedArray)

        let t = Task {
            let i = MLXRandom.normal([10, 10])
            let r = lm(i)
            print(sum(r))
            print(lm)
        }
        _ = await t.result

    }

    @Test
    func testMaterializedMultithreadedEval() async throws {
        // Exercise concurrent evaluation: produce MaterializedArrays on the
        // main task, fan them out to many child tasks, and have each task do
        // its own work (creates new arrays, evals, reads scalars) at the same
        // time.  This stresses the evalLock and the Sendable contract of
        // MaterializedArray.

        let taskCount = 16
        let iterations = 8
        let shape = [32, 32]

        // shared inputs created up-front and materialized so they can cross
        // task boundaries
        let a = MLXRandom.normal(shape).materialized()
        let b = MLXRandom.normal(shape).materialized()

        // expected reference value computed serially on the main task
        let expected = sum(square(a * b)).item(Float.self)

        try await withThrowingTaskGroup(of: Float.self) { group in
            for i in 0 ..< taskCount {
                group.addTask {
                    var last: Float = 0
                    for _ in 0 ..< iterations {
                        // mix the shared inputs with task-local arrays so that
                        // each task is producing fresh graphs and evaluating
                        // them concurrently
                        let local = MLXRandom.normal(shape)
                        let r = sum(square(a * b) + (local - local))
                        last = r.item(Float.self)

                        // also exercise materialize from inside a task
                        let m = (a + MLXArray(Float(i))).materialized()
                        _ = (m - MLXArray(Float(i))).sum().item(Float.self)
                    }
                    return last
                }
            }

            for try await value in group {
                #expect(abs(value - expected) < 1e-2)
            }
        }
    }

    @Test
    func testMaterializedHighContention() async throws {
        // High-contention variant: many tasks producing and consuming
        // MaterializedArrays through a shared actor-protected pool.  Every
        // task in a tight loop pulls two arrays from the pool, computes a
        // new one, materializes it (forcing an eval), reads a scalar, and
        // pushes the result back into the pool.  Lots of arrays flowing
        // between tasks, lots of concurrent eval calls hammering the
        // evalLock.

        actor Pool {
            var arrays: [MaterializedArray]
            init(_ initial: [MaterializedArray]) { self.arrays = initial }

            func take() -> MaterializedArray {
                arrays.randomElement()!
            }

            func replace(_ a: MaterializedArray) {
                arrays[Int.random(in: 0 ..< arrays.count)] = a
            }

            func snapshot() -> [MaterializedArray] {
                arrays
            }
        }

        let shape = [16, 16]
        let poolSize = 16
        let taskCount = 32
        let iterations = 50

        let initial = (0 ..< poolSize).map { _ in
            MLXRandom.normal(shape).materialized()
        }
        let pool = Pool(initial)

        // sum + count tracker so we can assert no task silently dropped work
        actor Counter {
            var n = 0
            func bump() { n += 1 }
            func value() -> Int { n }
        }
        let counter = Counter()

        try await withThrowingTaskGroup(of: Void.self) { group in
            for t in 0 ..< taskCount {
                group.addTask {
                    for k in 0 ..< iterations {
                        // pull two arrays from the shared pool — these may be
                        // referenced concurrently by other tasks at the same time
                        let a = await pool.take()
                        let b = await pool.take()

                        // build a new graph, materialize it (forces eval inside
                        // the task), and read a scalar (forces another eval).
                        // tanh keeps values bounded to [-1, 1] so the pool can
                        // recycle results across iterations without blowing up.
                        let mixin = MLXArray(Float(t * 31 + k) * 1e-3)
                        let r = tanh((a * b) + (a - b) + mixin).materialized()
                        let s = r.sum().item(Float.self)
                        #expect(s.isFinite)

                        // put the result back so other tasks see fresh arrays
                        await pool.replace(r)
                        await counter.bump()
                    }
                }
            }
            try await group.waitForAll()
        }

        #expect(await counter.value() == taskCount * iterations)

        // every entry in the final pool should still be a valid, finite array
        let final = await pool.snapshot()
        #expect(final.count == poolSize)
        for a in final {
            #expect(a.shape == shape)
            #expect(sum(a).item(Float.self).isFinite)
        }
    }

    @Test func testMaterializedModule() {
        let inner = Linear(10, 10)
        let mm = MaterializedModule(inner)

        #expect(mm.description.contains("Linear"))
        #expect(mm.parameterNBytes == mm.parameters().reduce(0) { $0 + $1.nbytes })

        // callable
        let x = uniform(0 ..< 1, [10])
        let _ = mm(x)
    }
}
