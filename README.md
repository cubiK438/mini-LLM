# Mini-LLM

**A decoder-only Transformer language model implemented and pretrained from scratch in PyTorch.**

Mini-LLM is an experimental language-modeling project focused on understanding and implementing the pipeline behind a small pretrained language model. It covers text preprocessing, byte-level BPE tokenization, Transformer architecture, training-data preparation, autoregressive pretraining, performance profiling, and downstream evaluation.

The project began with studying how raw internet data is filtered and prepared. It progressed through tokenizer development, implementation of Transformer components, data-pipeline construction, GPU performance experiments, pretraining, and benchmark evaluation.

> **Project status:** Experimental pretrained language model. Text-generation quality remains limited, and outputs can be incoherent or repetitive. This project is not a production-ready conversational assistant.

## Model Specifications

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Approximate parameter count | 100.97 million |
| Hidden dimension | 768 |
| Transformer layers | 5 |
| Attention heads | 12 |
| Feed-forward dimension | 3,072 |
| Maximum context length | 1,024 tokens |
| Tokenizer | Byte-level BPE |
| Original vocabulary size | 35,000 tokens |
| Training corpus | FineWeb-Edu |
| Training objective | Autoregressive next-token prediction |
| Framework | PyTorch |

The implementation includes multi-head attention, Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU feed-forward layers, and KV caching.

## Data Preparation

An important part of this project was understanding how raw internet text becomes usable language-model training data.

### Data filtration and quality

The development and learning process covered the following techniques:

- English-language filtering and content-quality filtering.
- Removal of unwanted content and spam-related material.
- Shingling and Jaccard similarity.
- MinHash and Locality-Sensitive Hashing (LSH) for approximate deduplication.
- C4-style text filters.
- Statistical text-quality filters, including line-length and character-distribution checks.
- Perplexity-based and classifier-based filtering concepts.

These techniques informed the project's understanding of web-scale dataset preparation. The development log does not establish that every listed filtering technique was applied to the final training corpus.

### Deduplication

Deduplication was explored and implemented using techniques including:

- Four-gram shingling.
- Jaccard similarity.
- MinHash.
- Locality-Sensitive Hashing.

The goal was to understand how similar documents can be detected efficiently without comparing every document against every other document directly.

## Tokenizer

Two approaches were explored:

1. A basic BPE tokenizer implemented from scratch.
2. A Hugging Face ByteLevelBPE tokenizer.

The Hugging Face tokenizer was trained on approximately 2 GB of FineWeb data with a vocabulary size of 35,000 tokens.

The recorded experiment achieved approximately **1.4 tokens per word**. A separate experiment with the custom BPE implementation achieved approximately 1.25 tokens per word on a much smaller dataset of roughly 1 MB. Since these experiments used different dataset sizes, they should not be interpreted as a controlled tokenizer comparison.

### Tokenization pipeline

```text
Raw text
   ↓
Text preprocessing
   ↓
Byte-level BPE tokenization
   ↓
Token IDs
   ↓
Binary token dataset (train.bin)
   ↓
Memory-mapped data access
   ↓
Training batches
   ↓
Transformer pretraining
```

The final data-preparation process produced approximately **2.02 billion tokens**, stored in a binary training file of roughly 4 GB.

## Transformer Implementation

The Transformer was implemented incrementally, with individual components tested during development.

### Multi-head attention

Implemented the attention mechanism in PyTorch and added unit tests for the attention implementation.

### Rotary Position Embeddings (RoPE)

Implemented RoPE and verified its integration with multi-head attention.

### RMSNorm

Implemented RMSNorm and compared its execution time with LayerNorm.

| Normalization | Recorded execution time |
|---|---:|
| RMSNorm | 0.06727 seconds |
| LayerNorm | 0.10948 seconds |

In the recorded experiment, RMSNorm was approximately **1.63× faster** than LayerNorm. This is a measurement from the tested setup, not a universal performance guarantee.

### KV cache

Implemented key-value caching to reduce repeated computation during autoregressive inference.

| Configuration | Recorded execution time |
|---|---:|
| Without KV cache | 0.03679 seconds |
| With KV cache | 0.01050 seconds |

The measured speedup was approximately **3.50×** in the tested setup.

### SwiGLU feed-forward network

Implemented a SwiGLU feed-forward block using the SiLU activation. The block uses separate gate and value projections, multiplies their outputs, and passes the result through a final projection.

The implementation also explored removing biases. A recorded experiment found a 0.14% difference in token count between the tested variants. This is an observation from that experiment, not a general conclusion about bias removal.

## Pretraining

The model was pretrained on approximately 2.02 billion tokens prepared from FineWeb-Edu.

The training implementation includes:

- Tokenized datasets and memory-mapped data loading.
- Batching and causal attention masks.
- Next-token prediction with cross-entropy loss.
- AdamW optimization.
- Learning-rate scheduling and warmup.
- Gradient accumulation and gradient clipping.
- Checkpoint saving and restoration.
- Validation-loss tracking and training metrics.

### GPU smoke test

A 20-step smoke test was performed on an NVIDIA L4 GPU to estimate throughput and memory requirements.

| Metric | Recorded result |
|---|---:|
| Model size | 100.97M parameters |
| Training steps | 20 |
| Total runtime | 7.42 seconds |
| Throughput | 11,028.69 tokens/second |
| Peak memory | 3,771.35 MB |
| Final logged training loss | 8.30747 |

This was a short smoke test for performance estimation, not the final pretraining run.

### Precision experiments

Separate FP32 and BF16 training runs were conducted to
