# Running on iOS

Considerations for running on iOS and general memory tuning.

## Controlling Memory Use

iOS devices have a feature called [jetsam](https://developer.apple.com/documentation/xcode/identifying-high-memory-use-with-jetsam-event-reports) 
which will terminate processes if they use too much memory.

Models may take several gigabytes for their weights and they need memory on 
top of that for evaluation.  The size of weights can be controlled by using
narrower types, e.g. `Float16` instead of `Float32` or qantizing the weights with 
the [QuantizeLinear](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlxnn/quantizedlinear) layer.

The LLM models in the <doc:examples> already make use of these techniques.

As MLX evaluates graphs (models) it produces temporary buffers and results.
These are managed automatically and for efficiency MLX will recycle the
buffers after they are disposed.  The limit on this cache is determined by
Metal's [recommendedMaxWorkingSetSize()](https://developer.apple.com/documentation/metal/mtldevice/2369280-recommendedmaxworkingsetsize)
but you may wish to limit this further.

For example, to evaluate an LLM you might allow up to 20 megabytes of buffer cache via ``GPU/set(cacheLimit:)``.

```swift
MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
```

``GPU/snapshot()`` can be used to monitor memory use over time:

```swift
// load model & weights
...

let startMemory = GPU.snapshot()

// work
...

let endMemory = GPU.snapshot()

// what stats are interesting to you?

print("=======")
print("Memory size: \(GPU.memoryLimit / 1024)K")
print("Cache size:  \(GPU.cacheLimit / 1024)K")

print("")
print("=======")
print("Starting memory")
print(startMemory.description)

print("")
print("=======")
print("Ending memory")
print(endMemory.description)

print("")
print("=======")
print("Growth")
print(startMemory.delta(endMemory).description)
```

It may be interesting to print the current memory statistics during evaluation if
you want to see performance over time.

Decreasing this number to 0 will result in decreased performance due to the
lack of buffer reuse, but it will also result in smaller memory use.
Tune this value for your needs.

Finally, if the device your code runs on has more RAM than the jetsam limit would
normally allow, you can use the [Increased Memory Limit](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_increased-memory-limit) entitlement.
