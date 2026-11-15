[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/4ka3wsjgxe/badge.svg)](https://msrit.ask.thili.ai/b/4ka3wsjgxe)
# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/jdzxfjrys2/badge.svg)](https://ask.thili.ai/b/jdzxfjrys2)

**Adventure Story Weaver** is a character-driven adventure story generator powered by a custom GPT language model implemented and trained using PyTorch.

The project explores how a GPT-style transformer learns patterns from adventure narratives and generates new text continuations from user-provided prompts. It includes a BPE tokenizer, causal self-attention, transformer blocks, a training pipeline, and an interactive command-line interface.

> This is an educational, small-scale language model built from scratch at the architecture level using PyTorch. It does not use an external LLM API for story generation.

## Features

- Custom GPT-style transformer architecture.
- Byte Pair Encoding (BPE) tokenizer.
- Token and positional embeddings.
- Multi-head causal self-attention.
- Feed-forward neural network with GELU activation.
- Layer normalization and residual connections.
- Training and validation pipeline.
- AdamW optimizer with cosine learning-rate scheduling.
- Text generation with temperature, Top-K sampling, and repetition penalty.
- Interactive CLI with Conservative, Balanced, and Creative generation styles.
- Automated tests using pytest.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| PyTorch | GPT architecture and model training |
| Tokenizers | BPE tokenization |
| pytest | Automated testing |
| Git and GitHub | Version control |

## Model Architecture

The model follows a GPT-style decoder-only transformer architecture.

**Processing pipeline:**

1. Input text is converted into token IDs using the BPE tokenizer.
2. Token embeddings and positional embeddings are combined.
3. The sequence passes through multiple transformer blocks.
4. Each block applies multi-head causal self-attention.
5. A feed-forward network processes the attention output.
6. Residual connections and layer normalization stabilize the network.
7. A final linear layer produces vocabulary logits.
8. The generation module samples the next token and continues the sequence.

### Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 1,000 |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Context window | 128 tokens |
| Batch size | 8 |
| Training iterations | 10,000 |
| Optimizer | AdamW |
| Learning-rate schedule | Cosine annealing |
| Device | CPU |

## Dataset

The model was trained on two public-domain adventure and fantasy books from Project Gutenberg:

1. **Treasure Island** — Robert Louis Stevenson ([Project Gutenberg #120](https://www.gutenberg.org/ebooks/120))
2. **The Wonderful Wizard of Oz** — L. Frank Baum ([Project Gutenberg #55](https://www.gutenberg.org/ebooks/55))

The preprocessing pipeline removes Gutenberg headers and footers, normalizes whitespace, preserves paragraph boundaries, and creates separate training and validation text files.

| Dataset statistic | Value |
|---|---:|
| Total characters | 570,612 |
| Total words | 108,286 |
| Total paragraphs | 2,576 |
| Training characters | 514,264 |
| Validation characters | 56,346 |

The two books are split independently at paragraph boundaries to avoid mixing their training and validation sections.

Dataset sources and preprocessing details are documented in `data/SOURCES.md`.

## Training Results

The final model was trained for 10,000 iterations on CPU.

**Final V4 training results:**

| Metric | Result |
|---|---:|
| Training loss | 2.9938 |
| Best validation loss | 3.7227 |
| Best checkpoint iteration | 10,000 |
| Final learning rate | 0.0001 |

The training pipeline saves the best validation-loss checkpoint.

The model was also compared with an earlier training experiment. V4 achieved a validation loss of 3.7227, compared with 3.7738 for V3.

## Text Generation

The generation module supports:

- **Temperature:** Controls the randomness of token selection.
- **Top-K sampling:** Restricts sampling to the highest-scoring candidate tokens.
- **Repetition penalty:** Reduces repeated token selection.
- **Context window:** Limits the input sequence to the model's supported context length.

### Generation Styles

| Style | Temperature | Top-K | Repetition Penalty |
|---|---:|---:|---:|
| Conservative | 0.5 | 10 | 1.15 |
| Balanced | 0.8 | 30 | 1.15 |
| Creative | 1.0 | 50 | 1.10 |

## Project Structure

```text
adventure-story-weaver/
│
├── app/
│   ├── attention.py
│   ├── batch_loader.py
│   ├── bpe_tokenizer.py
│   ├── cli.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── feed_forward.py
│   ├── generate.py
│   ├── layer_norm.py
│   ├── model.py
│   ├── position_embeddings.py
│   ├── residual.py
│   ├── tokenizer.py
│   ├── train.py
│   └── transformer_block.py
│
├── checkpoints/
│   └── adventure_gpt_bpe_v4_best.pt
│
├── data/
│   └── SOURCES.md
│
├── scripts/
│   ├── prepare_dataset.py
│   └── ...
│
├── tests/
│   ├── test_generate.py
│   ├── test_smoke.py
│   └── test_core.py
│
├── tokenizer/
│   └── adventure_bpe_v2.json
│
├── .gitignore
├── README.md
└── requirements.txt
```

Raw and processed datasets, virtual environments, and intermediate checkpoints are excluded from version control.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/prathameshj2306-cpu/adventure-story-weaver.git
cd adventure-story-weaver
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m app.cli
```

The application loads the included trained checkpoint and tokenizer, displays the generation-style menu, and asks for a story prompt.

## Training the Model

To retrain the model from scratch, first prepare the dataset and train or generate the required BPE tokenizer.

The training script expects:

- `data/processed/train_corpus.txt`
- `data/processed/val_corpus.txt`
- `tokenizer/adventure_bpe_v2.json`

Then run:

```bash
python -m app.train
```

Training is CPU-compatible but can take time depending on the hardware.

The included checkpoint allows users to run the application without repeating the training process.

## Testing

Run the automated test suite:

```bash
pytest -v
```

The project includes tests for generation behaviour, input validation, batch generation, context-window handling, storage, and CLI startup.

## Current Limitations

This project is a small educational GPT trained on a limited corpus of two books. Although the model learns some narrative patterns, its generated text is not consistently grammatical or coherent.

The model may produce malformed words, inconsistent character references, or abrupt narrative transitions.

The current implementation demonstrates the GPT architecture, training pipeline, and text-generation process rather than production-level language quality.

## Future Improvements

- Expand the training corpus with more public-domain narratives.
- Experiment with larger model configurations.
- Improve training-data quality and diversity.
- Evaluate generation quality using additional metrics.
- Explore prompt formatting and improved sampling strategies.
- Add options to save generated stories to text files.

## Author

Developed as an independent machine-learning and deep-learning portfolio project.
