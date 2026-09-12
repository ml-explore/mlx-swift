import Dispatch
import MLX

@main
struct SwiftPMMetallibConsumer {
    static func main() {
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            Stream(Device.gpu).synchronize()
        }

        let result = MLXArray([Float(1), 2, 3]) + 1
        eval(result)
        precondition(result.asArray(Float.self) == [2, 3, 4])
        print("GPU computation succeeded")
    }
}
