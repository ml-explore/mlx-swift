import Foundation
import Cmlx

// TODO: element-wise
public func abs(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_abs(array.ctx, stream.ctx))
}

// TODO: binary-arithmetic
public func add(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_add(a.ctx, b.ctx, stream.ctx))
}

// TODO: element-wise
public func acos(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_arccos(array.ctx, stream.ctx))
}

// TODO: element-wise
public func acosh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_arccosh(array.ctx, stream.ctx))
}

// TODO: element-wise
public func asin(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_arcsin(array.ctx, stream.ctx))
}

// TODO: element-wise
public func asinh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_arcsinh(array.ctx, stream.ctx))
}

// TODO: element-wise
public func atan(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_arctan(array.ctx, stream.ctx))
}

// TODO: element-wise
public func atanh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_arctanh(array.ctx, stream.ctx))
}

// TODO: indexes
public func argPartition(_ array: MLXArray, kth: Int, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_argpartition(array.ctx, kth.int32, axis.int32, stream.ctx))
}

// TODO: indexes
public func argPartition(_ array: MLXArray, kth: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_argpartition_all(array.ctx, kth.int32, stream.ctx))
}

// TODO: indexes
public func argSort(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_argsort(array.ctx, axis.int32, stream.ctx))
}

// TODO: indexes
public func argSort(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_argsort_all(array.ctx, stream.ctx))
}

public func asStrided(_ array: MLXArray, _ shape: [Int]? = nil, strides: [Int]? = nil, offset: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    let shape = shape ?? array.shape

    let resolvedStrides: [Int]
    if let strides {
        resolvedStrides = strides
    } else {
        var result = [Int]()
        var cum = 1
        for v in shape.reversed() {
            result.append(cum)
            cum = cum * v
        }
        resolvedStrides = result.reversed()
    }
    
    return MLXArray(mlx_as_strided(array.ctx, shape.asInt32, shape.count, resolvedStrides, resolvedStrides.count, offset, stream.ctx))
}

public func broadcast(_ array: MLXArray, to shape: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_broadcast_to(array.ctx, shape.asInt32, shape.count, stream.ctx))
}

// TODO: element-wise
public func ceil(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_ceil(array.ctx, stream.ctx))
}

public func clip(_ array: MLXArray, min: MLXArray, max: MLXArray? = nil, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_clip(array.ctx, min.ctx, max?.ctx, stream.ctx))
}

public func clip(_ array: MLXArray, max: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_clip(array.ctx, nil, max.ctx, stream.ctx))
}

// TODO: shapes
public func concatenate(_ arrays: [MLXArray], axis: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_concatenate(arrays.map { $0.ctx }, arrays.count, axis.int32, stream.ctx))
}

// TODO: shapes
public func concatenate(_ arrays: [MLXArray], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_concatenate_all(arrays.map { $0.ctx }, arrays.count, stream.ctx))
}

// TODO: convolution
public func conv1d(_ array: MLXArray, _ weight: MLXArray, stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_conv1d(array.ctx, weight.ctx, stride.int32, padding.int32, dilation.int32, groups.int32, stream.ctx))
}

public struct IntOrPair : ExpressibleByIntegerLiteral, ExpressibleByArrayLiteral {
    let values: (Int, Int)
    
    var first: Int { values.0 }
    var second: Int { values.1 }
    
    public init(integerLiteral value: Int) {
        self.values = (value, value)
    }
    
    public init(arrayLiteral elements: Int...) {
        precondition(elements.count == 2)
        self.values = (elements[0], elements[1])
    }
    
    public init(_ values: [Int]) {
        precondition(values.count == 2)
        self.values = (values[0], values[1])
    }
    
    public init(_ values: (Int, Int)) {
        self.values = values
    }
}

// TODO: convolution
public func conv2d(_ array: MLXArray, _ weight: MLXArray, stride: IntOrPair = 1, padding: IntOrPair = 0, dilation: IntOrPair = 1, groups: Int = 1, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_conv2d(array.ctx, weight.ctx, stride.first.int32, stride.second.int32, padding.first.int32, padding.second.int32, dilation.first.int32, dilation.second.int32, groups.int32, stream.ctx))
}

public enum ConvolveMode {
    case full
    case valid
    case same
}

// TODO: convolution
public func convolve(_ a: MLXArray, _ b: MLXArray, mode: ConvolveMode, stream: StreamOrDevice = .default) -> MLXArray {
    precondition(a.ndim == 1, "inputs must be 1d (a)")
    precondition(b.ndim == 1, "inputs must be 1d (b)")
    
    var (input, weight) = a.size < b.size ? (b, a) : (a, b)
    
    weight = MLXArray(mlx_slice(weight.ctx, [weight.dim(0) - 1].asInt32, 1, [-weight.dim(0) - 1].asInt32, 1, [-1], 1, stream.ctx))

    weight = weight.reshape([1, -1, 1], stream: stream)
    input = input.reshape([1, -1, 1], stream: stream)

    let weightSize = weight.size
    var padding = 0
        
    switch mode {
    case .full:
        padding = weightSize - 1
    case .valid:
        padding = 0
    case .same:
        if weightSize % 2 == 1 {
            padding = weightSize / 2
        } else {
            let padLeft = weightSize / 2
            let padRight = max(0, padLeft / 2 - 1)
            
            input = pad(input, widths: [0, [padLeft, padRight], 0], stream: stream)
        }
    }

    return MLXArray(mlx_conv1d(input.ctx, weight.ctx, 1, padding.int32, 1, 1, stream.ctx))
}

// TODO: element-wise
public func cosh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_cosh(array.ctx, stream.ctx))
}

public func dequantize(_ array: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int = 64, bits: Int = 4, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_dequantize(array.ctx, scales.ctx, biases.ctx, groupSize.int32, bits.int32, stream.ctx))
}

// TODO: binary-arithmetic
public func divide(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_divide(a.ctx, b.ctx, stream.ctx))
}

// TODO: logical
public func equal(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_equal(a.ctx, b.ctx, stream.ctx))
}

// TODO: element-wise
public func erf(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_erf(array.ctx, stream.ctx))
}

// TODO: element-wise
public func erfInverse(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_erfinv(array.ctx, stream.ctx))
}

// TODO: shapes
public func expandDimensions(_ array: MLXArray, axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_expand_dims(array.ctx, axes.asInt32, axes.count, stream.ctx))
}

// TODO: shapes
public func expandDimensions(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_expand_dims(array.ctx, [axis.int32], 1, stream.ctx))
}

// TODO: logical
public func greater(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_greater(a.ctx, b.ctx, stream.ctx))
}

// TODO: logical
public func greaterEqual(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_greater_equal(a.ctx, b.ctx, stream.ctx))
}

// TODO: logical
public func less(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_less(a.ctx, b.ctx, stream.ctx))
}

// TODO: logical
public func lessEqual(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_less_equal(a.ctx, b.ctx, stream.ctx))
}

enum LoadError : Error {
    case unableToOpen(URL, String)
}

public func load(url: URL, stream: StreamOrDevice = .default) throws -> MLXArray {
    let path = url.path(percentEncoded: false)
    if let fp = fopen(path, "r") {
        defer { fclose(fp) }
        return MLXArray(mlx_load(fp, stream.ctx))

    } else {
        let message = String(cString: strerror(errno))
        throw LoadError.unableToOpen(url, message)
    }
}

// TODO: binary-arithmetic
public func logAddExp(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_logaddexp(a.ctx, b.ctx, stream.ctx))
}

// TODO: element-wise
// TODO: logical
public func logicalNot(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_logical_not(array.ctx, stream.ctx))
}

// TODO: binary-arithmetic
public func maximum(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_maximum(a.ctx, b.ctx, stream.ctx))
}

// TODO: binary-arithmetic
public func minimum(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_minimum(a.ctx, b.ctx, stream.ctx))
}

// TODO: binary-arithmetic
public func multiply(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_multiply(a.ctx, b.ctx, stream.ctx))
}

// TODO: element-wise
public func negative(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_negative(array.ctx, stream.ctx))
}

// TODO: logical
public func notEqual(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_not_equal(a.ctx, b.ctx, stream.ctx))
}

// TODO: shapes
public func pad(_ array: MLXArray, width: IntOrPair, value: MLXArray = 0, stream: StreamOrDevice = .default) -> MLXArray {
    let ndim = array.ndim
    let axes = Array(Int32(0) ..< Int32(ndim))
    let lowPads = (0 ..< ndim).map { _ in width.first.int32 }
    let highPads = (0 ..< ndim).map { _ in width.second.int32 }

    return MLXArray(mlx_pad(array.ctx, axes, ndim, lowPads, ndim, highPads, ndim, value.ctx, stream.ctx))
}

// TODO: shapes
public func pad(_ array: MLXArray, widths: [IntOrPair], value: MLXArray = 0, stream: StreamOrDevice = .default) -> MLXArray {
    let ndim = array.ndim
    let axes = Array(Int32(0) ..< Int32(ndim))
    let lowPads = widths.map { $0.first.int32 }
    let highPads = widths.map { $0.second.int32 }

    return MLXArray(mlx_pad(array.ctx, axes, ndim, lowPads, ndim, highPads, ndim, value.ctx, stream.ctx))
}

// TODO: sorting
public func partition(_ array: MLXArray, kth: Int, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_partition(array.ctx, kth.int32, axis.int32, stream.ctx))
}

// TODO: sorting
public func partition(_ array: MLXArray, kth: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_partition_all(array.ctx, kth.int32, stream.ctx))
}

public func quantize(_ w: MLXArray, groupSize: Int = 64, bits: Int = 4, stream: StreamOrDevice = .default) -> (wq: MLXArray, scales: MLXArray, biases: MLXArray) {
    let result = mlx_quantize(w.ctx, groupSize.int32, bits.int32, stream.ctx)
    defer { mlx_vector_array_free(result) }
    
    return (MLXArray(result.arrays[0]!), MLXArray(result.arrays[1]!), MLXArray(result.arrays[2]!))
}

public func quantizedMatmul(_ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray, transpose: Bool = true, groupSize: Int = 64, bits: Int = 4, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_quantized_matmul(x.ctx, w.ctx, scales.ctx, biases.ctx, transpose, groupSize.int32, bits.int32, stream.ctx))
}

// TODO: binary-arithmetic
public func remainder(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_remainder(a.ctx, b.ctx, stream.ctx))
}

public func save(_ a: MLXArray, url: URL, stream: StreamOrDevice = .default) throws {
    let path = url.path(percentEncoded: false)
    if let fp = fopen(path, "w") {
        defer { fclose(fp) }
        mlx_save(fp, a.ctx)
        
    } else {
        let message = String(cString: strerror(errno))
        throw LoadError.unableToOpen(url, message)
    }
}

// TODO: element-wise
public func sigmoid(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sigmoid(array.ctx, stream.ctx))
}

// TODO: element-wise
public func sign(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sign(array.ctx, stream.ctx))
}

// TODO: element-wise
public func sinh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sinh(array.ctx, stream.ctx))
}

public func softMax(_ array: MLXArray, axes: [Int], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_softmax(array.ctx,  axes.asInt32, axes.count, stream.ctx))
}

public func softMax(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_softmax(array.ctx,  [axis.int32], 1, stream.ctx))
}

public func softMax(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_softmax_all(array.ctx,  stream.ctx))
}

// TODO: sorting
public func sort(_ array: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sort(array.ctx,  axis.int32, stream.ctx))
}

// TODO: sorting
public func sort(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_sort_all(array.ctx,  stream.ctx))
}

// TODO: shapes
public func stack(_ arrays: [MLXArray], axis: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_stack(arrays.map { $0.ctx }, arrays.count, axis.int32, stream.ctx))
}

// TODO: shapes
public func stack(_ arrays: [MLXArray], stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_stack_all(arrays.map { $0.ctx }, arrays.count, stream.ctx))
}

public func stopGradient(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_stop_gradient(array.ctx, stream.ctx))
}

// TODO: binary-arithmetic
public func subtract(_ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_subtract(a.ctx, b.ctx, stream.ctx))
}

// TODO: Functions returning an index along an axis, like argsort and argpartition, produce suitable indices for this function.
public func takeAlong(_ array: MLXArray, _ indices: MLXArray, axis: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_take_along_axis(array.ctx, indices.ctx, axis.int32, stream.ctx))
}

public func takeAlong(_ array: MLXArray, _ indices: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    let array = array.reshape([-1], stream: stream)
    return MLXArray(mlx_take_along_axis(array.ctx, indices.ctx, 0, stream.ctx))
}

// TODO: element-wise
public func tan(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_tan(array.ctx, stream.ctx))
}

// TODO: element-wise
public func tanh(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_tanh(array.ctx, stream.ctx))
}

// TODO: sorting
public func top(_ array: MLXArray, k: Int, axis: Int = -1, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_topk(array.ctx, k.int32, axis.int32, stream.ctx))
}

// TODO: sorting
public func top(_ array: MLXArray, k: Int, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_topk_all(array.ctx, k.int32, stream.ctx))
}

public func tril(_ array: MLXArray, k: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_tril(array.ctx, k.int32, stream.ctx))
}

public func triu(_ array: MLXArray, k: Int = 0, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_triu(array.ctx, k.int32, stream.ctx))
}

// TODO: logical
public func `where`(_ condition: MLXArray, _ a: MLXArray, _ b: MLXArray, stream: StreamOrDevice = .default) -> MLXArray {
    MLXArray(mlx_where(condition.ctx, a.ctx, b.ctx, stream.ctx))
}

