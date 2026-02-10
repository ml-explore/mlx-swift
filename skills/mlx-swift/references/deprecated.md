# Deprecated APIs and Migration Guide

This guide covers deprecated APIs in MLX Swift and their modern replacements.

## Module Consolidation

The following standalone modules have been deprecated. Their functionality is now available in the main `MLX` module through namespaces.

### MLXRandom Module Import

The separate `MLXRandom` module import is deprecated. Use `import MLX` instead:

```swift
// OLD (deprecated)
import MLXRandom
let values = MLXRandom.uniform(0.0 ..< 1.0, [3, 3])
MLXRandom.seed(42)

// NEW - use import MLX, then MLXRandom namespace or free functions
import MLX
let values = MLXRandom.uniform(0.0 ..< 1.0, [3, 3])
MLXRandom.seed(42)

// Alternative: use free functions
let values = uniform(0.0 ..< 1.0, [3, 3])
```

The `MLXRandom` enum is still the correct namespace for random functions. The deprecation only affects the separate module import.

### MLXFFT → FFT namespace

```swift
// OLD (deprecated)
import MLXFFT
let spectrum = MLXFFT.fft(signal)

// NEW
import MLX
let spectrum = FFT.fft(signal)
```

Deprecated functions:
- `MLXFFT.fft()` → `FFT.fft()`
- `MLXFFT.ifft()` → `FFT.ifft()`
- `MLXFFT.fft2()` → `FFT.fft2()`
- `MLXFFT.ifft2()` → `FFT.ifft2()`
- `MLXFFT.fftn()` → `FFT.fftn()`
- `MLXFFT.ifftn()` → `FFT.ifftn()`
- `MLXFFT.rfft()` → `FFT.rfft()`
- `MLXFFT.irfft()` → `FFT.irfft()`
- `MLXFFT.rfft2()` → `FFT.rfft2()`
- `MLXFFT.irfft2()` → `FFT.irfft2()`
- `MLXFFT.rfftn()` → `FFT.rfftn()`
- `MLXFFT.irfftn()` → `FFT.irfftn()`

### MLXLinalg → Linalg namespace

```swift
// OLD (deprecated)
import MLXLinalg
let inverse = MLXLinalg.inv(matrix)

// NEW
import MLX
let inverse = Linalg.inv(matrix)
```

Deprecated functions:
- `MLXLinalg.norm()` → `Linalg.norm()`
- `MLXLinalg.qr()` → `Linalg.qr()`
- `MLXLinalg.svd()` → `Linalg.svd()`
- `MLXLinalg.inv()` → `Linalg.inv()`
- `MLXLinalg.triInv()` → `Linalg.triInv()`
- `MLXLinalg.cholesky()` → `Linalg.cholesky()`
- `MLXLinalg.choleskyInv()` → `Linalg.choleskyInv()`
- `MLXLinalg.cross()` → `Linalg.cross()`
- `MLXLinalg.lu()` → `Linalg.lu()`
- `MLXLinalg.luFactor()` → `Linalg.luFactor()`
- `MLXLinalg.solve()` → `Linalg.solve()`
- `MLXLinalg.solveTriangular()` → `Linalg.solveTriangular()`

### MLXFast → MLXFast (some functions moved)

```swift
// OLD (deprecated) - top-level MLXFast functions
import MLXFast
let result = MLXFast.rmsNorm(x, weight: w, eps: 1e-5)

// NEW - still in MLXFast namespace but in main MLX module
import MLX
let result = MLXFast.rmsNorm(x, weight: w, eps: 1e-5)
```

Deprecated:
- `MLXFast.RoPE()` in MLXFast module → `MLXFast.RoPE()` in MLX module
- `MLXFast.rmsNorm()` in MLXFast module → `MLXFast.rmsNorm()` in MLX module
- `MLXFast.layerNorm()` in MLXFast module → `MLXFast.layerNorm()` in MLX module

## GPU → Memory Class

The `GPU` class for memory management has been renamed to `Memory`:

```swift
// OLD (deprecated)
let active = GPU.activeMemory
let cache = GPU.cacheMemory
let peak = GPU.peakMemory
GPU.set(cacheLimit: 1024 * 1024 * 1024)
GPU.clearCache()

// NEW
let active = Memory.activeMemory
let cache = Memory.cacheMemory
let peak = Memory.peakMemory
Memory.cacheLimit = 1024 * 1024 * 1024
Memory.clearCache()
```

Full list:
- `GPU.activeMemory` → `Memory.activeMemory`
- `GPU.cacheMemory` → `Memory.cacheMemory`
- `GPU.peakMemory` → `Memory.peakMemory`
- `GPU.snapshot()` → `Memory.snapshot()`
- `GPU.cacheLimit()` → `Memory.cacheLimit`
- `GPU.set(cacheLimit:)` → `Memory.cacheLimit = ...`
- `GPU.memoryLimit()` → `Memory.memoryLimit`
- `GPU.set(memoryLimit:)` → `Memory.memoryLimit = ...`
- `GPU.withWiredLimit()` → use `WiredMemoryTicket.withWiredLimit(...)` with `WiredMemoryManager`
- `GPU.clearCache()` → `Memory.clearCache()`

## Wired Limit API Migration

`withWiredLimit` APIs are now deprecated in favor of ticket-based coordination.

```swift
// OLD (deprecated)
try await GPU.withWiredLimit(bytes) {
    try await runInference()
}

try await Memory.withWiredLimit(bytes) {
    try await runInference()
}

// NEW
let ticket = WiredMemoryTicket(
    size: bytes,
    policy: WiredSumPolicy(),
    manager: .shared,
    kind: .active
)
try await ticket.withWiredLimit {
    try await runInference()
}
```

Important details:

- `GPU.withWiredLimit(...)` is deprecated with a migration message.
- `Memory.withWiredLimit(...)` async still works as a compatibility wrapper, but is deprecated.
- `Memory.withWiredLimit(...)` sync is deprecated and a no-op.
- For long-lived allocations (for example, model weights), use `.reservation` tickets.

## Function Renames

### Capitalization Changes

```swift
// OLD (deprecated)
addmm(c, a, b)
logSoftMax(x)
SoftMax()
SoftPlus()
Softsign()  // (layer version)
LogSoftMax() // (layer version)

// NEW
addMM(c, a, b)
logSoftmax(x)
Softmax()
Softplus()
Softsign()
LogSoftmax()
```

### repeat → repeated

```swift
// OLD (deprecated)
repeat(array, count: 3, axis: 0)
array.repeat(count: 3)

// NEW
repeated(array, count: 3, axis: 0)
array.repeated(count: 3)
```

### gatherMM Parameter Order

```swift
// OLD (deprecated)
gatherMM(a, b, lhsIndices, rhsIndices)

// NEW
gatherMM(a, b, lhsIndices: lhsIndices, rhsIndices: rhsIndices)
```

### softmax with precise Parameter

```swift
// OLD (deprecated)
softmax(x, axes: [1])
softmax(x, axis: 1)

// NEW (with optional precise parameter)
softmax(x, axes: [1], precise: false)
softmax(x, axis: 1, precise: false)
```

## Indexing API Changes

Use `.ellipsis` for advanced indexing:

```swift
array[.ellipsis, 0]        // Access across all dimensions
array[.newAxis, .ellipsis] // Add dimension at front
```

## Device API Changes

```swift
// Get default device
Device.defaultDevice()

// Temporarily use a different device
Device.withDefaultDevice(.cpu) {
    // Operations here use CPU
}
```

## Stream API Changes

```swift
// OLD (deprecated)
Stream(index: 0, device: .gpu)

// NEW
Stream(Device.gpu)
```

## Error Handling Changes

```swift
// OLD (deprecated)
useExceptionHandler()
clearExceptionHandler()

// NEW
withErrorHandler { ... }
withError { ... }
```

## MLXArray Changes

### strides Property

```swift
// OLD (deprecated)
let strides = array.strides

// NEW
let data = array.asData(access: .copy)
let strides = data.strides
```

### asData Without Access

```swift
// OLD (deprecated)
let bytes = array.asData()

// NEW
let bytes = array.asData(access: .copy)
```

## Quantization API Changes

```swift
// OLD (deprecated)
QuantizedLinear(linear)
QuantizedLinear.quantize(model: model, groupSize: 64, bits: 4)

// NEW - use top-level quantize function
quantize(model: model, groupSize: 64, bits: 4)
```

## MLXFast Kernel Changes

```swift
// OLD (deprecated)
MLXFastKernel(...)

// NEW
MLXFast.MLXFastKernel(...)
// Or use the factory method:
MLXFast.metalKernel(...)
```

## Scaled Dot Product Attention

```swift
// OLD (deprecated)
MLXFast.scaledDotProductAttention(..., mask: .arrays([mask1, mask2]))

// NEW
MLXFast.scaledDotProductAttention(..., mask: .array(mask))
```

## Activation Layers (Class Names)

```swift
// OLD (deprecated)
SoftMax()
SoftPlus()
LogSoftMax()

// NEW (proper capitalization)
Softmax()
Softplus()
LogSoftmax()
```

## Migration Checklist

When updating code:

1. **Replace module imports**:
   - Remove `import MLXRandom`, `import MLXFFT`, `import MLXLinalg`
   - Use `import MLX` with namespaced calls

2. **Update memory management**:
   - Replace `GPU.` with `Memory.`

3. **Fix function names**:
   - `addmm` → `addMM`
   - `repeat` → `repeated`
   - `logSoftMax` → `logSoftmax`

4. **Update layer class names**:
   - `SoftMax` → `Softmax`
   - `SoftPlus` → `Softplus`
   - `LogSoftMax` → `LogSoftmax`

5. **Fix Device usage**:
   - `Device.default` → `Device.defaultDevice()`
   - `Device.setDefault()` → `Device.withDefaultDevice() { }`

6. **Update error handling**:
   - `useExceptionHandler()` → `withErrorHandler { }`

7. **Fix quantization calls**:
   - Add required parameters to `QuantizedLinear.quantize()`
