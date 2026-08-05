import torch


def generate_sample_text(model,device, tokenizer, text, max_new_tokens):
    model.eval()

    tokens = torch.tensor([tokenizer.encode(text)]).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):

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
    return tokenizer.decode(tokens[0].tolist())