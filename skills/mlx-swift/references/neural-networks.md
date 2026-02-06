# Neural Networks Reference

The MLXNN module provides neural network layers built on top of MLX arrays.

## Module Base Class

All layers inherit from `Module`:

```swift
import MLXNN

class MyLayer: Module {
    // Parameters are automatically discovered via reflection
    let weight: MLXArray
    let bias: MLXArray

    init() {
        self.weight = MLXArray.zeros([10, 10])
        self.bias = MLXArray.zeros([10])
        super.init()
    }
}
```

### Key Module Methods

```swift
// Get all parameters
let params = model.parameters()

// Get trainable parameters only
let trainable = model.trainableParameters()

// Get child modules
let children = model.children()

// Freeze/unfreeze for fine-tuning
model.freeze()
model.unfreeze()
model.freeze(keys: ["encoder"])  // Freeze specific parts

// Train/eval mode
model.train(true)
model.train(false)  // Eval mode
model.training  // Check current mode
```

## Property Wrappers

### @ModuleInfo

Use for sub-modules to enable updates and quantization:

```swift
class Transformer: Module {
    @ModuleInfo var attention: MultiHeadAttention
    @ModuleInfo var ffn: Linear

    init() {
        self.attention = MultiHeadAttention(dimensions: 512, numHeads: 8)
        self.ffn = Linear(512, 512)
        super.init()
    }
}
```

### @ModuleInfo with Custom Keys

```swift
class Example: Module {
    @ModuleInfo(key: "weights") var w: Linear

    init() {
        self._w.wrappedValue = Linear(10, 20)
        super.init()
    }
}
```

## Linear Layers

### Linear

```swift
// Basic linear: y = xW^T + b
let layer = Linear(inputDimensions, outputDimensions, bias: true)
let output = layer(input)

// Without bias
let noBias = Linear(inputDimensions, outputDimensions, bias: false)
```

### Bilinear

```swift
// Bilinear: y = x1 W x2^T + b
let layer = Bilinear(inputDim1, inputDim2, outputDim)
let output = layer(x1, x2)
```

### Identity

```swift
// Pass-through layer
let layer = Identity()
let output = layer(input)  // output == input
```

## Convolution Layers

### Conv1d

```swift
let layer = Conv1d(
    inputChannels: 3,
    outputChannels: 16,
    kernelSize: 3,
    stride: 1,
    padding: 1
)

// Input shape: [batch, length, channels]
let output = layer(input)
```

### Conv2d

```swift
let layer = Conv2d(
    inputChannels: 3,
    outputChannels: 64,
    kernelSize: 3,      // or (3, 3)
    stride: 1,          // or (1, 1)
    padding: 1          // or (1, 1)
)

// Input shape: [batch, height, width, channels]
let output = layer(input)
```

### Conv3d

```swift
let layer = Conv3d(
    inputChannels: 3,
    outputChannels: 64,
    kernelSize: 3,
    stride: 1,
    padding: 1
)

// Input shape: [batch, depth, height, width, channels]
let output = layer(input)
```

### Transposed Convolutions

```swift
let layer = ConvTransposed1d(inputChannels: 64, outputChannels: 3, kernelSize: 3)
let layer = ConvTransposed2d(inputChannels: 64, outputChannels: 3, kernelSize: 3)
let layer = ConvTransposed3d(inputChannels: 64, outputChannels: 3, kernelSize: 3)
```

## Normalization Layers

### LayerNorm

```swift
let layer = LayerNorm(dimensions: 512, eps: 1e-5, affine: true)
let output = layer(input)
```

### RMSNorm

```swift
let layer = RMSNorm(dimensions: 512, eps: 1e-5)
let output = layer(input)
```

### BatchNorm

```swift
let layer = BatchNorm(
    featureCount: 64,
    eps: 1e-5,
    momentum: 0.1,
    affine: true,
    trackRunningStats: true
)
let output = layer(input)
```

### GroupNorm

```swift
let layer = GroupNorm(
    groupCount: 32,
    dimensions: 512,
    eps: 1e-5,
    affine: true,
    pytorchCompatible: true
)
let output = layer(input)
```

### InstanceNorm

```swift
let layer = InstanceNorm(dimensions: 64, eps: 1e-5, affine: false)
let output = layer(input)
```

## Attention

### MultiHeadAttention

```swift
let attention = MultiHeadAttention(
    dimensions: 512,
    numHeads: 8,
    queryInputDimensions: nil,  // defaults to dimensions
    keyInputDimensions: nil,
    valueInputDimensions: nil,
    valueDimensions: nil,
    valueOutputDimensions: nil,
    bias: false
)

// Self-attention (first parameter is unlabeled)
let output = attention(x, keys: x, values: x, mask: nil)

// Cross-attention
let output = attention(q, keys: k, values: v, mask: mask)
```

### Scaled Dot Product Attention (Fast)

```swift
// Optimized attention implementation
let output = MLXFast.scaledDotProductAttention(
    queries: queries,
    keys: keys,
    values: values,
    scale: 1.0 / sqrt(Float(headDim)),
    mask: mask
)
```

## Recurrent Layers

### RNN

```swift
let rnn = RNN(
    inputSize: 128,
    hiddenSize: 256,
    bias: true,
    nonLinearity: .tanh  // or .relu
)

let (output, hiddenState) = rnn(input, hiddenState: nil)
```

### LSTM

```swift
let lstm = LSTM(
    inputSize: 128,
    hiddenSize: 256,
    bias: true
)

let (output, (h, c)) = lstm(input, hiddenState: nil, cellState: nil)
```

### GRU

```swift
let gru = GRU(
    inputSize: 128,
    hiddenSize: 256,
    bias: true
)

let (output, hiddenState) = gru(input, hiddenState: nil)
```

## Embedding

```swift
let embedding = Embedding(embeddingCount: 10000, dimensions: 512)
let embedded = embedding(tokenIds)

// Access embedding weight directly
let weight = embedding.weight  // Shape: [embeddingCount, dimensions]
```

## Dropout

```swift
let dropout = Dropout(p: 0.1)

// Only active during training
model.train(true)
let output = dropout(input)  // Some values zeroed

model.train(false)
let output = dropout(input)  // Pass-through
```

### Dropout2d / Dropout3d

```swift
let dropout2d = Dropout2d(p: 0.1)  // Drops entire channels
let dropout3d = Dropout3d(p: 0.1)
```

## Pooling

### Average Pooling

```swift
let pool = AvgPool1d(kernelSize: 2, stride: 2, padding: 0)
let pool = AvgPool2d(kernelSize: (2, 2), stride: (2, 2), padding: (0, 0))
```

### Max Pooling

```swift
let pool = MaxPool1d(kernelSize: 2, stride: 2, padding: 0)
let pool = MaxPool2d(kernelSize: (2, 2), stride: (2, 2), padding: (0, 0))
```

## Upsampling

```swift
let upsample = Upsample(scaleFactor: 2, mode: .nearest)
let upsample = Upsample(scaleFactor: 2, mode: .linear)

let output = upsample(input)
```

## Activation Layers

```swift
// As layers (for Sequential)
ReLU()
GELU()
SiLU()
Mish()
LeakyReLU(negativeSlope: 0.01)
PReLU()
ELU(alpha: 1.0)
CELU(alpha: 1.0)
ReLU6()
Softmax()      // Note: NOT SoftMax (deprecated)
LogSoftmax()   // Note: NOT LogSoftMax (deprecated)
Softplus()     // Note: NOT SoftPlus (deprecated)
Softsign()
Sigmoid()
HardSwish()
Step()
SELU()
Tanh()
GLU(axis: -1)
```

### Activation Functions

```swift
// As functions
relu(x)
gelu(x)
silu(x)
sigmoid(x)
softmax(x, axis: -1)
logSoftmax(x, axis: -1)  // Note: NOT logSoftMax (deprecated)
tanh(x)
```

## Positional Encoding

### RoPE (Rotary Position Embedding)

```swift
let rope = RoPE(
    dimensions: 64,
    traditional: false,
    base: 10000.0,
    scale: 1.0
)

let output = rope(input, offset: 0)
```

### SinusoidalPositionalEncoding

```swift
let posEnc = SinusoidalPositionalEncoding(
    dimensions: 512,
    minFrequency: 0.0001,
    maxFrequency: 1.0,
    scale: 1.0
)

let output = posEnc(input)
```

### ALiBi (Attention with Linear Biases)

```swift
// ALiBi provides position-dependent attention bias
// Use the ALiBi layer which creates the bias internally
let alibi = ALiBi()

// Or create bias manually for custom implementations:
// Slopes decrease geometrically: 2^(-8/n), 2^(-16/n), ...
// Bias matrix is slopes * (query_pos - key_pos)
```

## Containers

### Sequential

```swift
let model = Sequential {
    Linear(784, 256)
    ReLU()
    Linear(256, 128)
    ReLU()
    Linear(128, 10)
}

let output = model(input)
```

## Loss Functions

```swift
// Cross entropy (classification)
crossEntropy(logits: logits, targets: targets, reduction: .mean)
crossEntropy(logits: logits, targets: targets, weights: classWeights)

// Binary cross entropy
binaryCrossEntropy(logits: logits, targets: targets, reduction: .mean)

// MSE (regression)
mseLoss(predictions: predictions, targets: targets, reduction: .mean)

// L1 loss
l1Loss(predictions: predictions, targets: targets, reduction: .mean)

// Smooth L1 (Huber)
smoothL1Loss(predictions: predictions, targets: targets, beta: 1.0, reduction: .mean)

// KL Divergence
klDivLoss(inputs, targets, reduction: .mean)

// Hinge losses
hingeLoss(inputs, targets, reduction: .mean)
marginRankingLoss(inputs1, inputs2, targets, margin: 1.0)

// Triplet loss
tripletLoss(anchors, positives, negatives, margin: 1.0)

// Contrastive loss
cosineSimilarityLoss(x1, x2, axis: -1)
```

### Reduction Options

```swift
.none   // No reduction, return per-element losses
.mean   // Mean of losses
.sum    // Sum of losses
```

## Quantization

```swift
// Quantize a model using the top-level function
quantize(model: model, groupSize: 64, bits: 4)

// Check if layer is quantizable
if let quantizable = layer as? Quantizable {
    let quantized = quantizable.toQuantized(groupSize: 64, bits: 4)
}
```
