import Foundation

/// A placeholder identity operator that is argument-insensitive.
public class Identity : Module {
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x
    }
}

/// Applies an affine transformation to the input.
///
/// Concretely:
///
/// ```swift
/// matmul(x, W) + b
/// ```
///
/// where `W` has shape `[inputDimensions, outputDimensions]` and `b` has
/// shape `[outputDimensions]`.
///
/// The values are initialized from the uniform distribution:
///
/// ```swift
/// scale = sqrt(1.0 / inputDimensions)
/// W = MLXRandom.uniform(-scale ..< scale, [outputDimensions, inputDimensions])
/// b = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
/// ```
public class Linear : Module {
    
    let weight: MLXArray
    let bias: MLXArray?

    /// See ``Linear``
    public init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true) {
        let scale = sqrt(1.0 / Float(inputDimensions))
        self.weight = MLXRandom.uniform(-scale ..< scale, [outputDimensions, inputDimensions])
        if bias {
            self.bias = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
        } else {
            self.bias = nil
        }
        super.init()
    }
    
    public override func describeParameters(_ indent: Int) -> String {
        "(inputDimensions=\(weight.dim(1)), outputDimensions=\(weight.dim(0)), bias=\(bias == nil ? "false" : "true"))"
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x.matmul(weight.T)
        if let bias {
            result = result + bias
        }
        return result
    }
}

public class Bilinear : Module {
    
    let weight: MLXArray
    let bias: MLXArray?
 
    public init(_ inputDimensions1: Int, _ inputDimensions2: Int, _ outputDimensions: Int, bias: Bool = true) {
        let scale = sqrt(1.0 / Float(inputDimensions1))
        self.weight = MLXRandom.uniform(-scale ..< scale, [outputDimensions, inputDimensions2, inputDimensions1])
        if bias {
            self.bias = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
        } else {
            self.bias = nil
        }
        super.init()
    }
    
    public override func describeParameters(_ indent: Int) -> String {
        "(inputDimensions1=\(weight.dim(2)), inputDimensions2=\(weight.dim(1)), outputDimensions=\(weight.dim(0)), bias=\(bias == nil ? "false" : "true"))"
    }
    
    public func callAsFunction(_ x1: MLXArray, _ x2: MLXArray) -> MLXArray {
        // normalize shapes
        let out = weight.dim(0)
        let in1 = weight.dim(2)
        let in2 = weight.dim(1)
        let xShape = x1.shape.dropLast()
        let x1 = x1.reshape(-1, in1)
        let x2 = x2.reshape(-1, 1, in2)

        // perform the bilinear transform
        var y: MLXArray
        let w = weight.reshape(out * in2, in1)
        y = x1.matmul(w.T)
        y = y.reshape(-1, out, in2).swapAxes(-2, -1)
        y = x2.matmul(y)
        y = y.squeeze(axis: 1)
        
        // reset the shape
        y = y.reshape(xShape + [out])
        
        if let bias {
            y = y + bias
        }
        return y
    }
}

