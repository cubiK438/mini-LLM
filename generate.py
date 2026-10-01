import torch
from model.transformer_block import Transformer, causal_mask
from tokenizer.tokenizer import tokenize, decode

vocab_size = 35000
d_model = 768
num_layers = 5
heads = 12
intermediate_dim = 3072
max_seq_len = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    heads=heads,
    intermediate_dim=intermediate_dim
).to(device)

checkpoint = torch.load("checkpoints/latest.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    tokens = tokenize(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -max_seq_len:]
            mask = causal_mask(context.size(1)).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(context, mask)

            logits = logits[:, -1, :] / temperature

            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            next_token = indices.gather(-1, torch.multinomial(probs, 1))

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return decode(input_ids[0].tolist())

prompt = "The future of artificial intelligence"
print(generate(prompt))