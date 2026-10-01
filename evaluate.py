import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from model.transformer_block import Transformer, causal_mask

vocab_size = 35000
d_model = 768
num_layers = 5
heads = 12
intermediate_dim = 3072
seq_len = 1024
batch_size = 16
num_batches = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(vocab_size=vocab_size,d_model=d_model,num_layers=num_layers,heads=heads,intermediate_dim=intermediate_dim).to(device)

checkpoint = torch.load("checkpoints/latest.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")

loss_fn = CrossEntropyLoss()
losses = []

with torch.no_grad():
    for i in range(num_batches):
        start = i * batch_size * seq_len
        end = start + batch_size * seq_len

        tokens = torch.from_numpy(val_data[start:end].copy()).long()
        tokens = tokens.reshape(batch_size, seq_len).to(device)

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        mask = causal_mask(inputs.size(1)).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs, mask)
            loss = loss_fn(logits.transpose(1, 2), targets)

        losses.append(loss.item())

mean_loss = np.mean(losses)
perplexity = np.exp(mean_loss)

print(f"Checkpoint step: {checkpoint['step']}")
print(f"Validation loss: {mean_loss}")
print(f"Perplexity: {perplexity}")
print(f"Batches evaluated: {num_batches}")
print(f"Tokens evaluated: {num_batches * batch_size * (seq_len - 1)}")