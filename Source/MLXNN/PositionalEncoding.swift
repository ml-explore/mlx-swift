import Foundation
import MLX

final public class RoPE : Module, UnaryModel {

    let dimensions: Int
    let traditional: Bool
    let base: Float
    let scale: Float
    
    struct Key : Hashable {
        let N: Int
        let D: Int
        let offset: Int
        let base: Float
        let scale: Float
        let dtype: DType
    }
    
    static let cache = Cache<Key, (MLXArray, MLXArray)>()
    
    public init(dimensions: Int, traditional: Bool = false, base: Float = 10_000, scale: Float = 1) {
        self.dimensions = dimensions
        self.traditional = traditional
        self.base = base
        self.scale = scale
    }
    
    static func cosSinTheta(key: Key) -> (MLXArray, MLXArray) {
        if let values = cache[key] {
            return values
        }
        
        let D = key.D / 2
        let positions = MLXArray(key.offset ..< key.N).asType(key.dtype) * key.scale
        let freqs = exp(-MLXArray(0 ..< D).asType(key.dtype)) * (log(key.base) / Float(D))
        let theta = positions.reshape(-1, 1) * freqs.reshape(1, -1)
        
        let result = (cos(theta), sin(theta))
        cache[key] = result
        
        return result
    }
    
    func rope(costheta: MLXArray, sintheta: MLXArray, x: MLXArray) -> MLXArray {
        let x1 = x[0 ..< (self.dimensions / 2), axis: -1]
        let x2 = x[(self.dimensions / 2) ..< self.dimensions, axis: -1]
        
        let rx1 = x1 * costheta - x2 * sintheta
        let rx2 = x1 * sintheta + x2 * costheta
        
        let rx: MLXArray
        if self.dimensions < x.dim(-1) {
            rx = concatenate([rx1, rx2, x[self.dimensions... , axis: -1]], axis: -1)
        } else {
            rx = concatenate([rx1, rx2], axis: -1)
        }
        return rx
    }
    
    func traditionalRope(costheta: MLXArray, sintheta: MLXArray, x: MLXArray) -> MLXArray {
        let x1 = x[stride: 2, axis: -1]
        let x2 = x[from: 1, stride: 2, axis: -1]
        
        let rx1 = x1 * costheta - x2 * sintheta
        let rx2 = x1 * sintheta + x2 * costheta

        if dimensions < x.dim(-1) {
            fatalError("RoPE doesn't implement partial traditional application")
        }
        
        let rx = concatenate([expandDimensions(rx1, axis: -1), expandDimensions(rx2, axis: -1)], axis: -1)
        
        return rx
    }
    
    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let shape = x.shape
        let x = x.reshape(-1, shape[shape.endIndex - 2], shape[shape.endIndex - 1])
        let N = x.dim(1) + offset
        
        let key = Key(N: N, D: dimensions, offset: offset, base: base, scale: scale, dtype: x.dtype)
        let (costheta, sintheta) = Self.cosSinTheta(key: key)
        
        let f = traditional ? traditionalRope : rope
        let rx = f(costheta, sintheta, x)
        
        return rx.reshape(shape)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, offset: 0)
    }
}

