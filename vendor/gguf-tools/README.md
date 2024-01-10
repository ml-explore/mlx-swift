# GGUF tools

This is a work in progress library to manipulate GGUF files.
While the library aims to be useful, one of the main goals is to provide
an accessible code base that as a side effect documents the GGUF
files used by the awesome [llama.cpp](https://github.com/ggerganov/llama.cpp) project: GGUF files are becoming increasingly more used and central in
the _local_ machine learning scene, so to have multiple implementations
of parsers and files generators may be useful.

The program **gguf-tools** uses the library to implement both useful and
useless stuff, to show the library usage in the real world. For now
the utility implements the following subcommands:

### gguf-tools show file.gguf

shows detailed info about the GGUF file. This will include all the key-value pairs, including arrays, and detailed tensors informations. Tensor offsets will be relative to the start *of the file* (so they are actually absolute offsets), not the start of the data section like in the GGUF format.

Example output:

```
./gguf-tools show models/phi-2.Q8_0.gguf | head -20        :main*: ??
models/phi-2.Q8_0.gguf (ver 3): 20 key-value pairs, 325 tensors
general.architecture: [string] phi2
general.name: [string] Phi2
phi2.context_length: [uint32] 2048
phi2.embedding_length: [uint32] 2560
phi2.feed_forward_length: [uint32] 10240
phi2.block_count: [uint32] 32
phi2.attention.head_count: [uint32] 32
phi2.attention.head_count_kv: [uint32] 32
phi2.attention.layer_norm_epsilon: [float32] 0.000010
phi2.rope.dimension_count: [uint32] 32
general.file_type: [uint32] 7
tokenizer.ggml.add_bos_token: [bool] false
tokenizer.ggml.model: [string] gpt2
tokenizer.ggml.tokens: [array] [!, ", #, $, %, &, ', (, ), *, +, ,, -, ., /, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, :, ;, <, =, >, ... 51170 more items of 51200]

... many more key-value pairs ...

q8_0 tensor token_embd.weight @1806176, 131072000 weights, dims [2560,51200], 139264000 bytes
f32 tensor blk.0.attn_norm.bias @141070176, 2560 weights, dims [2560], 10240 bytes
f32 tensor blk.0.attn_norm.weight @141080416, 2560 weights, dims [2560], 10240 bytes
f32 tensor blk.0.attn_qkv.bias @141090656, 7680 weights, dims [7680], 30720 bytes
q8_0 tensor blk.0.attn_qkv.weight @141121376, 19660800 weights, dims [2560,7680], 20889600 bytes
f32 tensor blk.0.attn_output.bias @162010976, 2560 weights, dims [2560], 10240 bytes
q8_0 tensor blk.0.attn_output.weight @162021216, 6553600 weights, dims [2560,2560], 6963200 bytes
f32 tensor blk.0.ffn_up.bias @168984416, 10240 weights, dims [10240], 40960 bytes

... many more tensors ...
```

### gguf-tools compare file1.gguf file2.gguf

This tool is useful to understand if two LLMs (or other models distributed as GGUF files) are related, for instance if one is the finetune of another, or if both are fine-tuned from the same parent model.

For each matching tensor (same name and parameters count), the command computes the average weights difference (in percentage, so that a random distribution in the interval -N, +N would be on average 100% different than another random distribution in the same interval). This is useful to see if a model is a finetune of another model, how much it was finetuned, which layers were frozen while finetuning and so forth. Note that because of quantization, even tensors that are functionally equivalent may have some small average difference.

Example output:

```
./gguf-tools compare mistral-7b-instruct-v0.2.Q8_0.gguf \
                     solar-10.7b-instruct-v1.0-uncensored.Q8_0.gguf
[token_embd.weight]: avg weights difference: 44.539944%
[blk.0.attn_q.weight]: avg weights difference: 48.717736%
[blk.0.attn_k.weight]: avg weights difference: 56.201885%
[blk.0.attn_v.weight]: avg weights difference: 47.087249%
[blk.0.attn_output.weight]: avg weights difference: 47.663048%
[blk.0.ffn_gate.weight]: avg weights difference: 37.508761%
[blk.0.ffn_up.weight]: avg weights difference: 39.061584%
[blk.0.ffn_down.weight]: avg weights difference: 39.632648%
...
```

### gguf-tools inspect-tensor file.gguf tensor.name [count]

Show all (if count is not specified, otherwise only the first _count_) weights values of the specified tensor. This is useful for low level stuff, like checking if quantization is working as expected, see the introduced error, model fingerprinting and so forth.

### gguf-tools split-mixtral 65230776370407150546470161412165 mixtral.gguf out.gguf

Extracts a 7B model `out.gguf` from Mixtral 7B MoE using the specified MoE ID for each layer (there are 32 digits in the sequence 652...).

Note that split-mixtral is quite useless as models obtained in this way will not perform any useful work. This is just an experiment and a non trivial task to show how to use the library. Likely it will be removed soon, once I have more interesting and useful examples to show, like models merging.

## gufflib API

For now the only documentation is the implementation itself: see the
gguf-tools.c for usage information. This may change later, but for now
the library is under active development.

The code is well commented, and the API so far is extremely simple to understand and use.

## Limitations

Many quantization formats are missing.

## Specification documents

* [Official GGUF specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md), where the file layout and meta-data is described.
* [Quantization formats](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.h) used in quantized GGUF models.
