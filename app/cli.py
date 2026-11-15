
import torch
from pathlib import Path

from app.model import GPTLanguageModel
from app.generate import generate
from app.bpe_tokenizer import BPETokenizer


# ==========================================
# 1. CONFIGURATION
# ==========================================

CHECKPOINT_PATH = Path(
    "checkpoints/adventure_gpt_bpe_v4_best.pt"
)

MAX_NEW_TOKENS = 300

DEVICE = "cpu"


# ==========================================
# 2. GENERATION STYLES
# ==========================================

GENERATION_STYLES = {
    "1": {
        "name": "Conservative",
        "temperature": 0.5,
        "top_k": 10,
        "repetition_penalty": 1.15
    },

    "2": {
        "name": "Balanced",
        "temperature": 0.8,
        "top_k": 30,
        "repetition_penalty": 1.15
    },

    "3": {
        "name": "Creative",
        "temperature": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
}


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

def load_model():

    print("\nLoading Adventure Story Weaver...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    config = checkpoint["model_config"]

    tokenizer_path = checkpoint["tokenizer_path"]

    tokenizer = BPETokenizer.load(tokenizer_path)

    model = GPTLanguageModel(**config).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully!")

    print("Vocabulary size:", tokenizer.vocab_size)

    print("Training iteration:", checkpoint["iteration"])

    print(
        "Best validation loss:",
        round(checkpoint["val_loss"], 4)
    )

    return model, tokenizer, config


# ==========================================
# 4. DISPLAY MAIN MENU
# ==========================================

def display_menu():

    print("\n" + "=" * 50)

    print("          ADVENTURE STORY WEAVER")

    print("=" * 50)

    print("\nYour own GPT-powered adventure story generator!\n")

    print("Choose your generation style:\n")

    print("1. Conservative")
    print("2. Balanced")
    print("3. Creative")
    print("4. Exit")

    print("-" * 50)


# ==========================================
# 5. SELECT GENERATION STYLE
# ==========================================

def choose_style():

    while True:

        display_menu()

        choice = input("\nEnter your choice: ").strip()

        if choice == "4":
            return None

        if choice in GENERATION_STYLES:

            style = GENERATION_STYLES[choice]

            print(
                f"\nSelected style: {style['name']}"
            )

            return style

        print("\nInvalid choice. Please select 1, 2, 3 or 4.")


# ==========================================
# 6. GENERATE ADVENTURE STORY
# ==========================================

def generate_story(model, tokenizer, config, style):

    print("\n" + "=" * 50)

    print("             STORY GENERATION")

    print("=" * 50)

    prompt = input(
        "\nEnter your story prompt:\n> "
    ).strip()

    if not prompt:

        print("\nPrompt cannot be empty.")

        return

    print("\nGenerating your adventure story...")

    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded_prompt],
        dtype=torch.long,
        device=DEVICE
    )

    # Generate new tokens
    with torch.no_grad():

        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            block_size=config["block_size"],
            temperature=style["temperature"],
            top_k=style["top_k"],
            repetition_penalty=style["repetition_penalty"]
        )

    # Extract only newly generated tokens
    new_token_ids = output_ids[
        0,
        len(encoded_prompt):
    ].tolist()

    generated_text = tokenizer.decode(
        new_token_ids
    )

    # Display generated story
    print("\n" + "=" * 50)

    print("              YOUR ADVENTURE")

    print("=" * 50)

    print(prompt + generated_text)

    print("\n" + "=" * 50)

    print("Generation completed!")

    print("=" * 50)


# ==========================================
# 7. AFTER GENERATION MENU
# ==========================================

def after_generation():

    while True:

        print("\nWhat would you like to do next?\n")

        print("1. Generate another story")
        print("2. Change generation style")
        print("3. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice in ["1", "2", "3"]:

            return choice

        print("\nInvalid choice. Please try again.")


# ==========================================
# 8. MAIN APPLICATION
# ==========================================

def main():

    print("\nStarting Adventure Story Weaver...")

    model, tokenizer, config = load_model()

    style = choose_style()

    if style is None:

        print("\nThank you for using Adventure Story Weaver!")

        return

    while True:

        generate_story(
            model,
            tokenizer,
            config,
            style
        )

        choice = after_generation()

        if choice == "1":
            continue

        elif choice == "2":

            style = choose_style()

            if style is None:
                break

        elif choice == "3":
            break

    print("\nThank you for using Adventure Story Weaver!")

    print("Happy storytelling!")


# ==========================================
# 9. APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")

    except Exception as error:
        print("\nApplication error:", error)
        raise
