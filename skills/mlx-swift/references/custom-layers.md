# Custom Layers Reference

Guide to creating custom neural network layers in MLX Swift.

## Basic Custom Layer

```swift
import MLX
import MLXNN

class MyLayer: Module {
    let weight: MLXArray
    let bias: MLXArray

    init(inputDim: Int, outputDim: Int) {
        // Initialize parameters
        let scale = sqrt(1.0 / Float(inputDim))
        self.weight = MLXRandom.uniform(-scale ..< scale, [outputDim, inputDim])
        self.bias = MLXArray.zeros([outputDim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T) + bias
    }
}
```

## Using @ModuleInfo

For layers containing sub-modules, use `@ModuleInfo` to enable updates and quantization:

```swift
class TransformerBlock: Module {
    @ModuleInfo var attention: MultiHeadAttention
    @ModuleInfo var ffn1: Linear
    @ModuleInfo var ffn2: Linear
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm

    init(dimensions: Int, numHeads: Int, hiddenDim: Int? = nil) {
        let hidden = hiddenDim ?? dimensions * 4
        self.attention = MultiHeadAttention(dimensions: dimensions, numHeads: numHeads)
        self.ffn1 = Linear(dimensions, hidden)
        self.ffn2 = Linear(hidden, dimensions)
        self.norm1 = LayerNorm(dimensions: dimensions)
        self.norm2 = LayerNorm(dimensions: dimensions)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + attention(norm1(x), keys: norm1(x), values: norm1(x), mask: mask)
        let ffnOut = ffn2(gelu(ffn1(norm2(h))))
        h = h + ffnOut
        return h
    }
}
```

### Custom Parameter Keys

```swift
class Example: Module {
    @ModuleInfo(key: "weights_1") var w1: Linear
    @ModuleInfo(key: "weights_2") var w2: Linear

    init() {
        self._w1.wrappedValue = Linear(10, 20)
        self._w2.wrappedValue = Linear(20, 10)
        super.init()
    }
}
```

## Conforming to Protocols

### UnaryLayer

For layers with single input/output:

```swift
class MyActivation: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        maximum(x, 0)  // ReLU
    }
}

// Can be used in Sequential
let model = Sequential {
    Linear(10, 20)
    MyActivation()
    Linear(20, 10)
}
```

### Quantizable

For layers that can be quantized:

```swift
class MyLinear: Module, Quantizable {
    let weight: MLXArray
    let bias: MLXArray?

    // ... init and forward ...

    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}
```

## Parameter Management

### Accessing Parameters

```swift
let model = MyModel()

// All parameters
let params = model.parameters()

// Trainable only
let trainable = model.trainableParameters()

// Access specific child modules via children()
let children = model.children()
```

### Updating Parameters

```swift
// Update from dictionary
model.update(parameters: newParams)

// Update modules
model.update(modules: newModules)
```

### Freezing Parameters

```swift
// Freeze all
model.freeze()

// Freeze specific
model.freeze(keys: ["encoder"])

// Unfreeze
model.unfreeze()
model.unfreeze(keys: ["decoder"])
```

## Custom Description

Override `describeExtra` for better debugging output:

```swift
class MyConv: Module {
    let weight: MLXArray

    override func describeExtra(_ indent: Int) -> String {
        let (out, inp, kH, kW) = weight.shape4
        return "(in=\(inp), out=\(out), kernel=\(kH)x\(kW))"
    }
}

// Prints: MyConv(in=3, out=64, kernel=3x3)
```

## Train/Eval Mode

```swift
class MyDropoutLayer: Module {
    let p: Float

    init(p: Float = 0.1) {
        self.p = p
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if training {
            // Apply dropout during training
            let mask = MLXRandom.bernoulli(p: 1 - p, x.shape)
            return x * mask / (1 - p)
        } else {
            // Pass through during eval
            return x
        }
    }
}

// Usage
model.train(true)   // Training mode
model.train(false)  // Eval mode
```

### Responding to Mode Changes

```swift
class MyLayer: Module {
    override func didSetTrain(_ train: Bool) {
        // Called when training mode changes
        // Useful for updating internal state
    }
}
```

## Layer with Multiple Outputs

```swift
class EncoderDecoder: Module {
    @ModuleInfo var encoder: Encoder
    @ModuleInfo var decoder: Decoder

    func callAsFunction(_ x: MLXArray) -> (encoded: MLXArray, decoded: MLXArray) {
        let encoded = encoder(x)
        let decoded = decoder(encoded)
        return (encoded, decoded)
    }
}
```

## Layer with Optional Components

```swift
class ConditionalLayer: Module {
    @ModuleInfo var main: Linear
    @ModuleInfo var optional: Linear?

    init(inputDim: Int, outputDim: Int, useOptional: Bool) {
        self.main = Linear(inputDim, outputDim)
        self.optional = useOptional ? Linear(outputDim, outputDim) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = main(x)
        if let opt = optional {
            h = opt(h)
        }
        return h
    }
}
```

## Layer with Arrays of Modules

```swift
class MultiHeadLayer: Module {
    @ModuleInfo var heads: [Linear]

    init(numHeads: Int, inputDim: Int, outputDim: Int) {
        self.heads = (0..<numHeads).map { _ in
            Linear(inputDim, outputDim)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        heads.map { $0(x) }
    }
}
```

## Complete Example: Residual Block

```swift
class ResidualBlock: Module, UnaryLayer {
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var norm1: BatchNorm
    @ModuleInfo var norm2: BatchNorm
    @ModuleInfo var downsample: Conv2d?

    init(inChannels: Int, outChannels: Int, stride: Int = 1) {
        self.conv1 = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 3,
            stride: stride,
            padding: 1
        )
        self.conv2 = Conv2d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
        self.norm1 = BatchNorm(featureCount: outChannels)
        self.norm2 = BatchNorm(featureCount: outChannels)

        // Downsample shortcut if dimensions change
        if stride != 1 || inChannels != outChannels {
            self.downsample = Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: 1,
                stride: stride
            )
        } else {
            self.downsample = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = downsample?(x) ?? x

        var out = conv1(x)
        out = norm1(out)
        out = relu(out)

        out = conv2(out)
        out = norm2(out)

        out = out + identity
        return relu(out)
    }
}
```
