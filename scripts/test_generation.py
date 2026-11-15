
import torch

from app.model import GPTLanguageModel
from app.generate import generate
from app.bpe_tokenizer import BPETokenizer


# ==========================================
# 1. CONFIGURATION
# ==========================================

CHECKPOINT_PATH = "checkpoints/adventure_gpt_bpe_v4_best.pt"

MAX_NEW_TOKENS = 200

PROMPT = "Once upon a time"


# ==========================================
# 2. LOAD TRAINED CHECKPOINT
# ==========================================

print("========== LOADING TRAINED GPT ==========")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

config = checkpoint["model_config"]

tokenizer_path = checkpoint["tokenizer_path"]

tokenizer = BPETokenizer.load(tokenizer_path)

model = GPTLanguageModel(**config)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint loaded successfully!")
print("Tokenizer vocabulary:", tokenizer.vocab_size)
print("Training iteration:", checkpoint["iteration"])
print("Best validation loss:", checkpoint["val_loss"])


# ==========================================
# 3. DEFINE SAMPLING EXPERIMENTS
# ==========================================

experiments = [
    {
        "name": "Experiment A - Conservative",
        "temperature": 0.5,
        "top_k": 10,
        "repetition_penalty": 1.15
    },
    {
        "name": "Experiment B - Balanced",
        "temperature": 0.8,
        "top_k": 30,
        "repetition_penalty": 1.15
    },
    {
        "name": "Experiment C - Creative",
        "temperature": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
]


# ==========================================
# 4. RUN GENERATION EXPERIMENTS
# ==========================================

print("\n========== GENERATION EXPERIMENTS ==========")

for experiment in experiments:

    # Reset random seed for reproducibility
    torch.manual_seed(42)

    # Encode input prompt
    encoded_prompt = tokenizer.encode(PROMPT)

    input_ids = torch.tensor(
        [encoded_prompt],
        dtype=torch.long
    )

    # Generate story
    with torch.no_grad():

        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            block_size=config["block_size"],
            temperature=experiment["temperature"],
            top_k=experiment["top_k"],
            repetition_penalty=experiment["repetition_penalty"]
        )

    # Decode generated tokens
    generated_text = tokenizer.decode(
        output_ids[0].tolist()
    )

    # Display results
    print("\n" + "=" * 65)
    print(experiment["name"])
    print("=" * 65)

    print("Temperature:", experiment["temperature"])
    print("Top-K:", experiment["top_k"])
    print("Repetition Penalty:", experiment["repetition_penalty"])

    print("\nGenerated Story:\n")
    print(generated_text)

print("\n========== ALL EXPERIMENTS COMPLETED ==========")
