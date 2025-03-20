# ``MLXRandom``

Collection of functions related to random number generation.

Random sampling functions in MLX use an implicit global PRNG state by default. However, all 
functions take an optional key keyword argument for when more fine-grained control or explicit state management is needed.

For example, you can generate random numbers with:

```swift
for _ in 0 ..< 3:
  print(MLXRandom.uniform())
```

which will print a sequence of unique pseudo random numbers. Alternatively you can explicitly set the key:

```swift
let key = MLXRandom.key(0)
for _ in 0 ..< 3:
  print(MLXRandom.uniform(key: key))
```

which will yield the same pseudo random number at each iteration.

Following [JAX’s PRNG design](https://jax.readthedocs.io/en/latest/jep/263-prng.html) we use a
splittable version of Threefry, which is a counter-based PRNG.

## Other MLX Packages

- [MLX](mlx)
- [MLXRandom](mlxrandom)
- [MLXNN](mlxnn)
- [MLXOptimizers](mlxoptimizers)
- [MLXFFT](mlxfft)
- [MLXLinalg](mlxlinalg)
- [MLXFast](mlxfast)

- [Python `mlx`](https://ml-explore.github.io/mlx/build/html/index.html)

## Topics

### Keys and Seeds

- ``key(_:)``
- ``split(key:into:stream:)``
- ``seed(_:)``

