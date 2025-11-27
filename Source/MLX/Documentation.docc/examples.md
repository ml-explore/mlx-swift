# MLX Swift Examples

Swift example code for MLX and MLXNN.

@Metadata {
    @CallToAction(
        purpose: download,
        url: "https://github.com/ml-explore/mlx-swift-examples/archive/refs/heads/main.zip")
    @PageKind(sampleCode)
}

## Overview

Examples using ``MLX`` and `MLXNN` are available in
[mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) (example applications and tools)
and [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) (implementations of various LLM and VLM model architectures).
The examples include:

- [MNISTTrainer](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/MNISTTrainer/README.md): An example that runs on
  both iOS and macOS that downloads MNIST training data and trains a
  [LeNet](https://en.wikipedia.org/wiki/LeNet).

- [MLXChatExample](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/MLXChatExample/README.md): An example chat app that runs on both iOS and macOS that supports LLMs and VLMs.

- [LLMEval](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/LLMEval/README.md): A simple example that runs on both iOS
  and macOS that downloads an LLM and tokenizer from Hugging Face and
  generates text from a given prompt.

- [StableDiffusionExample](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/StableDiffusionExample/README.md): An
  example that runs on both iOS and macOS that downloads a stable diffusion model
  from Hugging Face and generates an image from a given prompt.

- [llm-tool](https://github.com/ml-explore/mlx-swift-examples/blob/main/Tools/llm-tool/README.md): A command line tool for generating text
  using a variety of LLMs available on the Hugging Face hub.

and several more.  Much of the code is also available as a SwiftPM package -- use this as a starting point to build your own applications.
