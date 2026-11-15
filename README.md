# Adventure Story Weaver

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/jdzxfjrys2/badge.svg)](https://ask.thili.ai/b/jdzxfjrys2)

This project guides you through building a custom GPT from scratch, specifically designed to generate text adventure story segments. By training on existing adventure game narratives, your GPT will create dynamic and coherent story continuations. This demonstrates your ability to apply core transformer concepts to a creative text generation task.

## What you are building

**Goal.** You will have built a custom GPT-based text adventure story segment generator that creates dynamic story continuations based on adventure game narratives.

**Done when.** The command-line interface (CLI) allows a user to input a story prompt and receives generated story segments that are coherent and follow the style of adventure narratives.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Assemble a small corpus of text from public domain adventure stories or fantasy novels.**  
      Hint: Gather and preprocess your text corpus.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings from the story text.**  
      Hint: Apply byte-pair tokenization and create embeddings.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Implement causal self-attention and build the transformer block.
- [ ] **4. Set up the training loop and train the GPT model on the narrative data.**  
      Hint: Set up and run the model training loop.
- [ ] **5. Implement a text generation feature that creates coherent story continuations.**  
      Hint: Develop the text generation logic.
- [ ] **6. Design a CLI where users can input a story prompt and receive generated segments.**  
      Hint: Create a user-friendly command-line interface.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify parameters like generation length or creativity temperature.
- Implement a feature to save and load trained model checkpoints.
- Expand the CLI to support multi-turn story generation, where previous segments influence new ones.
