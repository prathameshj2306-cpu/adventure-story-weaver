
from pathlib import Path

import torch
import torch.nn.functional as F

from app.dataset import AdventureDataset
from app.batch_loader import get_batch
from app.model import GPTLanguageModel


# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

torch.manual_seed(42)


# ==========================================
# 2. TRAINING CONFIGURATION
# ==========================================

BATCH_SIZE = 8
BLOCK_SIZE = 128

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4

LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.0001

MAX_ITERS = 10000

EVAL_INTERVAL = 1000
EVAL_ITERS = 30

DEVICE = "cpu"


# ==========================================
# 3. FILE PATHS
# ==========================================

CORPUS_PATH = "data/processed/train_corpus.txt"

VAL_CORPUS_PATH = "data/processed/val_corpus.txt"

TOKENIZER_PATH = "tokenizer/adventure_bpe_v2.json"

# New checkpoint: does not overwrite v3
CHECKPOINT_PATH = Path(
    "checkpoints/adventure_gpt_bpe_v4_best.pt"
)


# ==========================================
# 4. LOAD DATASET
# ==========================================

dataset = AdventureDataset(
    train_corpus_path=CORPUS_PATH,
    val_corpus_path=VAL_CORPUS_PATH,
    tokenizer_path=TOKENIZER_PATH
)

train_data = dataset.train_data
val_data = dataset.val_data

vocab_size = dataset.tokenizer.vocab_size


print("========== BPE DATASET LOADED ==========")

print("Vocabulary size:", vocab_size)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Device:", DEVICE)


# ==========================================
# 5. CREATE GPT MODEL
# ==========================================

model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    block_size=BLOCK_SIZE
).to(DEVICE)


# ==========================================
# 6. OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 7. COSINE ANNEALING SCHEDULER
# ==========================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_ITERS,
    eta_min=MIN_LEARNING_RATE
)


# ==========================================
# 8. VALIDATION LOSS FUNCTION
# ==========================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):

            x, y = get_batch(
                data,
                BATCH_SIZE,
                BLOCK_SIZE,
                DEVICE
            )

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# ==========================================
# 9. TRAINING
# ==========================================

best_val_loss = float("inf")

print("\n========== BPE GPT V4 TRAINING STARTED ==========")

for iteration in range(MAX_ITERS + 1):

    # --------------------------------------
    # Evaluate model
    # --------------------------------------

    if iteration % EVAL_INTERVAL == 0:

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Step {iteration:5d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------
        # Save best validation checkpoint
        # ----------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            CHECKPOINT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),

                    "model_config": {
                        "vocab_size": vocab_size,
                        "embedding_dim": EMBEDDING_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "block_size": BLOCK_SIZE
                    },

                    "tokenizer_type": "BPE",
                    "tokenizer_path": TOKENIZER_PATH,

                    "train_loss": train_loss,
                    "val_loss": val_loss,

                    "iteration": iteration,

                    "learning_rate": current_lr
                },
                CHECKPOINT_PATH
            )

            print(
                f"  Best checkpoint saved! "
                f"Validation Loss: {best_val_loss:.4f}"
            )

    # --------------------------------------
    # Stop after final evaluation
    # --------------------------------------

    if iteration == MAX_ITERS:
        break

    # --------------------------------------
    # Get training batch
    # --------------------------------------

    x, y = get_batch(
        train_data,
        BATCH_SIZE,
        BLOCK_SIZE,
        DEVICE
    )

    # --------------------------------------
    # Forward pass
    # --------------------------------------

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    # --------------------------------------
    # Backpropagation
    # --------------------------------------

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    # --------------------------------------
    # Gradient clipping
    # --------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # --------------------------------------
    # Update model parameters
    # --------------------------------------

    optimizer.step()

    # --------------------------------------
    # Update learning rate
    # --------------------------------------

    scheduler.step()


# ==========================================
# 10. TRAINING COMPLETED
# ==========================================

print("\n========== TRAINING COMPLETED ==========")

print(f"Best validation loss: {best_val_loss:.4f}")

print("Best checkpoint saved to:", CHECKPOINT_PATH)

print("\nBPE GPT V4 TRAINING COMPLETED!")
