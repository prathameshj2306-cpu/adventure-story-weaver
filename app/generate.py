
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    block_size,
    temperature=1.0,
    top_k=None,
    repetition_penalty=1.0
):
    # ==========================================
    # 1. INPUT VALIDATION
    # ==========================================

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens cannot be negative"
        )

    if block_size < 1:
        raise ValueError(
            "block_size must be positive"
        )

    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch, sequence]"
        )

    if input_ids.size(1) == 0:
        raise ValueError(
            "Input prompt cannot be empty"
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero"
        )

    if top_k is not None and top_k < 1:
        raise ValueError(
            "top_k must be at least 1"
        )

    if repetition_penalty <= 0:
        raise ValueError(
            "Repetition penalty must be positive"
        )

    if max_new_tokens == 0:
        return input_ids

    model.eval()

    # ==========================================
    # 2. AUTOREGRESSIVE GENERATION LOOP
    # ==========================================

    for _ in range(max_new_tokens):

        # Keep only the most recent context
        input_context = input_ids[:, -block_size:]

        # Predict next-token logits
        logits = model(input_context)

        # Select logits from the final position
        logits = logits[:, -1, :]

        # ==========================================
        # 3. REPETITION PENALTY
        # ==========================================

        if repetition_penalty != 1.0:

            for batch_index in range(input_ids.size(0)):

                previous_tokens = input_ids[
                    batch_index
                ].unique()

                previous_logits = logits[
                    batch_index,
                    previous_tokens
                ]

                penalized_logits = torch.where(
                    previous_logits < 0,
                    previous_logits * repetition_penalty,
                    previous_logits / repetition_penalty
                )

                logits[
                    batch_index,
                    previous_tokens
                ] = penalized_logits

        # ==========================================
        # 4. TEMPERATURE SCALING
        # ==========================================

        logits = logits / temperature

        # ==========================================
        # 5. TOP-K FILTERING
        # ==========================================

        if top_k is not None:

            k = min(top_k, logits.size(-1))

            values, _ = torch.topk(
                logits,
                k
            )

            threshold = values[:, -1].unsqueeze(-1)

            logits = logits.masked_fill(
                logits < threshold,
                float("-inf")
            )

        # ==========================================
        # 6. SOFTMAX PROBABILITIES
        # ==========================================

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        # ==========================================
        # 7. SAMPLE NEXT TOKEN
        # ==========================================

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # ==========================================
        # 8. APPEND TOKEN TO SEQUENCE
        # ==========================================

        input_ids = torch.cat(
            (input_ids, next_token),
            dim=1
        )

    return input_ids
