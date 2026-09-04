# SwiftPM metallib consumer smoke test

This nested package verifies the downstream macOS CLI use case rather than an
in-package test target. It builds a release executable against the parent
`mlx-swift` package, confirms that SwiftPM emitted
`mlx-swift_Cmlx.bundle/default.metallib`, copies the executable and resource
bundle into a clean deployment directory, and launches the executable from a
different working directory. The executable races several first GPU stream
initializations, then evaluates and verifies a GPU array operation.

Run it from the repository root:

```sh
Tests/SwiftPMMetallibConsumer/run.sh
```
