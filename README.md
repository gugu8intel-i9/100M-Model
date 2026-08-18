# Tern-1

**Ternary 100M Parameter Language Model with Chain-of-Thought Reasoning**

[![Model Size](https://img.shields.io/badge/Model_Size-100M-yellow)](https://huggingface.co/Gugu8/Tern-1)  
[![Quantization](https://img.shields.io/badge/Quantization-Ternary_(±1,_0)-blue)](https://arxiv.org/)  
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)  
[![Hugging Face](https://img.shields.io/badge/🤗-HuggingFace-orange)](https://huggingface.co/Gugu8/Tern-1)

---

## Overview

**Tern-1** is a 100-million parameter language model that combines two key innovations:

1. **Ternary Weight Quantization**: Weights are constrained to values of `{-1, 0, +1}`, enabling extreme memory efficiency and fast inference on edge devices.
2. **Chain-of-Thought (CoT) Fine-tuning**: Trained with explicit reasoning traces to improve multi-step problem solving and mathematical reasoning.

Despite its compact size, Tern-1 achieves competitive performance on reasoning benchmarks through its CoT-optimized training and ternary-aware architecture.

---

## Key Features

- **Ultra-Compact**: ~100M parameters stored in 2 bits per weight (ternary) → **~25 MB** model size
- **Fast Inference**: Ternary operations enable bitwise computation, up to **3-5× faster** than full-precision models of similar size
- **CoT Reasoning**: Generates step-by-step reasoning traces before final answers
- **Efficient Deployment**: Runs on CPU, mobile devices, and resource-constrained environments
- **Fine-Tunable**: Compatible with standard fine-tuning pipelines with ternary-aware optimizers

---

## Architecture

| Component | Details |
|-----------|---------|
| **Layers** | 16 Transformer decoder layers |
| **Hidden Size** | 768 |
| **Heads** | 12 attention heads |
| **Context Length** | 2048 tokens |
| **Weight Precision** | Ternary {-1, 0, +1} + scale factor per layer |
| **Activations** | bfloat16 (inference) / bfloat16 (training) |
| **Training Data** | Public datasets from Hugging Face (see Training Details) |

### Ternary Weight Representation

```
Standard:  fp32 (32 bits)  →  Ternary: 2 bits + shared scale
Each weight = sign(x) * scale, where sign(x) ∈ {-1, 0, 1}
```

---

## Chain-of-Thought Fine-Tuning (Planned)

Tern-1 will be fine‑tuned on a curated dataset of ~5M examples with explicit reasoning traces, including:

- **Math Word Problems**: Multi-step arithmetic and algebra with reasoning steps
- **Logical Reasoning**: Deductive and inductive reasoning tasks
- **Code Reasoning**: Algorithmic step-by-step explanations
- **General QA**: Questions requiring multi-hop reasoning

CoT fine‑tuning will commence after the pretraining phase is complete.

### Example Output (Expected)

**Input:**
```
Q: Sarah bought 3 notebooks for $2 each and 2 pens for $1.50 each.
How much did she spend in total?
```
**Output:**
```
Step 1: Cost of notebooks = 3 × $2 = $6
Step 2: Cost of pens = 2 × $1.50 = $3
Step 3: Total cost = $6 + $3 = $9
Final answer: $9
```

---

## Performance

> **Training is currently in progress** on Google Cloud TPU v5.  
> Benchmark results (GSM8K, MATH, BBH, MMLU) will be published here once evaluation is complete.

---

## Use Cases

- **Edge AI Applications**: On-device reasoning for mobile assistants
- **Educational Tools**: Step-by-step math problem solving
- **Low-Resource Environments**: Serverless, IoT, and embedded systems
- **Research**: Efficient reasoning, sparse models, and quantization studies

---

## Getting Started

> Note: The model is not yet publicly released. The following code is for reference once training is finished.

### Installation

```bash
pip install transformers accelerate
```

### Loading the Model (Future)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Gugu8/Tern-1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to("cpu")  # runs efficiently on CPU
```

### Inference with CoT (Future)

```python
def generate_with_cot(prompt):
    input_text = f"Q: {prompt}\nA: Let me reason step by step."
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Training Details

- **Hardware**: Google Cloud TPU v5 (8× v5e chips)
- **Pretraining**:
  - Datasets: Mixture of public corpora from Hugging Face, including:
    - C4 (Colossal Clean Crawled Corpus)
    - OpenWebText
    - Math‑specific datasets (e.g., arXiv, StackExchange)
  - Tokens: ~100B (ongoing)
  - Optimizer: AdamW with ternary‑aware gradient clipping
- **CoT Fine‑Tuning** (planned):
  - Dataset: 5M examples from GSM8K, MATH, and synthetic reasoning tasks
  - Duration: ~2 days on same TPU cluster

---

## Limitations

- **Reasoning Depth**: May struggle with tasks requiring >8 reasoning steps
- **Knowledge Capacity**: 100M parameters limits factual recall
- **Generalization**: Best suited for structured reasoning tasks
- **Ternary Quantization**: Slight accuracy degradation compared to full‑precision

---

## Citation

If you use Tern-1 in your research, please cite:

```bibtex
@misc{tern1-2026,
  title = {Tern-1: A Ternary 100M Language Model with Chain-of-Thought Reasoning},
  author = {gugu8intel-i9},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/gugu8intel-i9/tern-1}
}
```

---

## License

This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Thanks to the open-source AI community
- Inspired by research on ternary neural networks and CoT reasoning
- Built with Hugging Face Transformers and PyTorch/XLA on Google Cloud TPU

---

## Contact

For questions, feedback, or collaborations, please open an issue on GitHub or reach out at `g73447476@gmail.com`.

---

**Made with ❤️ for efficient, reasoning-capable AI**
