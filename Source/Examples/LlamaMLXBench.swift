import Foundation
import Mlx

@main
struct LlamaMLXBench {
    static func measure(model: LlamaEncoderLayer, x: MLXArray, cache: (MLXArray, MLXArray)) -> Int {
        for _ in 0 ..< 5 {
            let (y, c) = model(x, mask: nil, cache: cache)
            eval(y, c)
        }
        
        let start = Date.timeIntervalSinceReferenceDate
        var rs = [MLXArray]()
        for _ in 0 ..< 5 {
            let (y, c) = model(x, mask: nil, cache: cache)
            rs.append(y)
            rs.append(c.0)
            rs.append(c.1)
        }
        eval(rs)
        let end = Date.timeIntervalSinceReferenceDate
        
        return Int((end - start) * 1000 / 5)
    }
    
    static func main() {
        let H = 32
        let D = 4096
        let F = 43 * 256
        let C = 1000

        // TODO: set dtype as float16
        
        let layer = LlamaEncoderLayer(dimensions: D, mlpDimensions: F, numHeads: H)
        
        // TODO: keys not used?  perhaps should be used in 3 calls below?
        let keys = MLXRandom.split(key: MLXRandom.key(0), into: 3)
        let x = MLXRandom.normal([1, 1, D], type: Float.self)
        let cache = (
            MLXRandom.normal([1, H, C, D / H], type: Float.self),
            MLXRandom.normal([1, H, C, D / H], type: Float.self)
        )
        eval(x, cache)
        
        let t = measure(model: layer, x: x, cache: cache)
        
        print("Time per layer per token: \(t) ms")
        print("Lower bound total time per token: \(t * 32) ms")

    }
}

public class LlamaAttention : Module {
    
    let dimensions: Int
    let numHeads: Int

    let rope: RoPE
    let queryProjection: Linear
    let keyProjection: Linear
    let valueProjection: Linear
    let outProjection: Linear

    public init(dimensions: Int, numHeads: Int) {
        
        self.dimensions = dimensions
        self.numHeads = numHeads
        
        self.rope = RoPE(dimensions: dimensions / numHeads, traditional: true)
        self.queryProjection = Linear(dimensions, dimensions, bias: false)
        self.keyProjection = Linear(dimensions, dimensions, bias: false)
        self.valueProjection = Linear(dimensions, dimensions, bias: false)
        self.outProjection = Linear(dimensions, dimensions, bias: false)
        
        super.init()
    }
    
    public func callAsFunction(queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray? = nil, cache: (MLXArray, MLXArray)? = nil) -> (MLXArray, (MLXArray, MLXArray)) {
        var queries = queryProjection(queries)
        var keys = keyProjection(keys)
        var values = valueProjection(values)
        
        let B = queries.dim(0)
        let L = queries.dim(1)
        
        queries = queries.reshape(B, L, numHeads, -1).transpose(axes: [0, 2, 1, 3])
        keys = keys.reshape(B, L, numHeads, -1).transpose(axes: [0, 2, 1, 3])
        values = values.reshape(B, L, numHeads, -1).transpose(axes: [0, 2, 1, 3])

        if let (keyCache, valueCache) = cache {
            queries = rope(queries, offset: keyCache.dim(2))
            keys = rope(keys, offset: keyCache.dim(2))
            keys = concatenate([keyCache, keys], axis: 2)
            values = concatenate([valueCache, values], axis: 2)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }
        
        // Dimensions are [batch x num heads x sequence x hidden dim]
        let scale = MLXArray(sqrt(1 / Float(queries.dim(-1)))).asType(queries.dtype)
        var scores = (queries * scale).matmul(keys.transpose(axes: [0, 1, 3, 2]))
        if let mask {
           scores = scores + mask
        }
        scores = softMax(scores, axis: -1)
        let valuesHat = scores.matmul(values).transpose(axes: [0, 2, 1, 3]).reshape(B, L, -1)
        
        return (outProjection(valuesHat), (keys, values))
    }
}

public class LlamaEncoderLayer : Module {
    
    let attention: LlamaAttention
    let norm1: RMSNorm
    let norm2: RMSNorm
    
    let linear1: Linear
    let linear2: Linear
    let linear3: Linear
    
    public init(dimensions: Int, mlpDimensions: Int, numHeads: Int) {
        self.attention = LlamaAttention(dimensions: dimensions, numHeads: numHeads)
        
        self.norm1 = RMSNorm(dimensions)
        self.norm2 = RMSNorm(dimensions)
        
        self.linear1 = Linear(dimensions, mlpDimensions, bias: false)
        self.linear2 = Linear(dimensions, mlpDimensions, bias: false)
        self.linear3 = Linear(mlpDimensions, dimensions, bias: false)
        
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: (MLXArray, MLXArray)? = nil) -> (MLXArray, (MLXArray, MLXArray)) {
        var y = norm1(x)
        var resultCache: (MLXArray, MLXArray)
        (y, resultCache) = attention(queries: y, keys: y, values: y, mask: mask, cache: cache)
        var x = x + y
        
        y = norm2(x)
        let a = linear1(y)
        let b = linear2(y)
        y = a * sigmoid(a) * b
        y = linear3(y)
        x = x + y
        
        return (x, resultCache)
    }
}
