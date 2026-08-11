import torch

def generate_stream(model,device, tokenizer, text, max_tokens):
    model.eval()

    token_ids = tokenizer.encode(text)
    tokens = torch.tensor(
        token_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    for _ in range(max_tokens):

        with torch.no_grad():
            logits = model(tokens)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )

        # Decode only the newly generated token
        new_text = tokenizer.decode(
            [next_token.item()]
        )

        yield new_text