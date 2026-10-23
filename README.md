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

Separate FP32 and BF16 training runs were conducted to investigate training behavior under different numerical precisions. Metrics were recorded for both runs.

### Pretraining outcome

The training log records a lowest validation loss of **3.3178** across the pretraining runs. Training was resumed across separate runs, with checkpoints used to continue the experiment.

The original log records 61,707 epochs; this terminology should be verified against the training configuration before interpreting it as the number of full dataset passes or optimization steps.

## Baseline Evaluation

On 10 October 2026, the pretrained model was evaluated before supervised fine-tuning to establish a baseline for subsequent post-training experiments.

The evaluation covered ARC-Easy, ARC-Challenge, HellaSwag, and WinoGrande.

| Benchmark | Samples | Accuracy | Standard error |
|---|---:|---:|---:|
| ARC-Easy | 2,376 | 50.29% | 1.03 percentage points |
| ARC-Challenge | 1,172 | 20.90% | 1.19 percentage points |
| HellaSwag | 10,042 | 27.32% | 0.44 percentage points |
| WinoGrande | 1,267 | 51.46% | 1.40 percentage points |

The accuracy values use the recorded `acc,none` metric. Standard errors are expressed in percentage points.

These results represent the model's recorded baseline before supervised fine-tuning. They provide a reference for future experiments, but comparisons with other models require consistent evaluation settings and metrics.

Benchmark accuracy and free-form generation quality measure different aspects of model behavior. These results do not establish that the model can generate reliable or coherent conversational responses.

## Development Timeline

The following timeline summarizes the major milestones recorded during development.

| Date | Milestone |
|---|---|
| June 2026 | Studied web-data filtration, deduplication, MinHash, LSH, and Jaccard similarity |
| 3–5 June | Built a basic BPE tokenizer and trained a Hugging Face ByteLevelBPE tokenizer |
| June 2026 | Studied Transformer fundamentals and the text-to-token training pipeline |
| 6–8 July | Implemented attention, RoPE, RMSNorm, KV caching, and SwiGLU |
| 9 August | Integrated the data pipeline and tested it on 1,000 documents |
| 15–16 August | Conducted initial training, FP32, and BF16 experiments |
| 29 August | Ran an NVIDIA L4 smoke test and recorded throughput and memory usage |
| 31 August | Completed preparation of the approximately 2.02-billion-token training dataset |
| September | Completed the recorded pretraining runs and tracked validation loss |
| 10 October | Evaluated the pretrained model on four downstream benchmarks |

### Collaboration

The initial development involved BNS Kartheek and Harsha Nerella. Harsha's development log records contributions to data deduplication and normalization experiments, along with work on understanding data-filtering techniques. The log states that Harsha stopped working on the project in August, after which development continued independently.

## Limitations

Mini-LLM is a small experimental model trained under a limited compute and data budget.

Known limitations include:

- Incoherent or repetitive free-form generations.
- Limited instruction-following ability.
- No claim of competitive general-purpose language-model performance.
- Tokenizer and vocabulary changes that require careful checkpoint-compatibility checks.
- Performance measurements that depend on hardware, workload, and implementation.
- Benchmark results that do not establish production readiness.

The repository preserves the implementation and experiments, including unsuccessful or incomplete areas, as part of the research and learning process.

## Future Work

This repository focuses on implementing and pretraining a language model from scratch.

A separate research project will investigate post-training and preference optimization using established pretrained models. The intended direction is to compare:

- Supervised Fine-Tuning (SFT).
- Reinforcement Learning from Human Feedback (RLHF).
- Direct Preference Optimization (DPO).

The goal is to conduct controlled experiments using comparable datasets, evaluation methods, and training configurations, with selected experiments repeated on models of different sizes.

Keeping the projects separate will make it easier to distinguish language-model pretraining experiments from research on alignment and preference optimization.

## Acknowledgements

Thanks to Harsha Nerella for contributions to the early data-processing and model-development work.

## License

See the repository's `LICENSE` file for the applicable terms.
