import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer_block import Transformer

VOCAB_SIZE_OLD = 35000
VOCAB_SIZE_NEW = 35001

CHECKPOINT_PATH = Path("mini-llm-latest.pt")
OUTPUT_PATH = Path("mini-llm-latest.pt")

device = torch.device("cpu")

model = Transformer(
    vocab_size=VOCAB_SIZE_OLD,
    d_model=768,
    num_layers=5,
    heads=12,
    intermediate_dim=3072,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state_dict"])

old_embedding = model.embedding
old_lm_head = model.lm_head

new_embedding = torch.nn.Embedding(VOCAB_SIZE_NEW, 768)
new_lm_head = torch.nn.Linear(768, VOCAB_SIZE_NEW, bias=False)

with torch.no_grad():
    new_embedding.weight[:VOCAB_SIZE_OLD].copy_(old_embedding.weight)
    new_lm_head.weight[:VOCAB_SIZE_OLD].copy_(old_lm_head.weight)

    torch.nn.init.normal_(new_embedding.weight[VOCAB_SIZE_OLD:], mean=0.0, std=0.02)
    torch.nn.init.normal_(new_lm_head.weight[VOCAB_SIZE_OLD:], mean=0.0, std=0.02)

model.embedding = new_embedding
model.lm_head = new_lm_head

#Verify all existing weights are unchanged
assert model.embedding.weight.shape == (35001, 768)
assert model.lm_head.weight.shape == (35001, 768)

#Save a separate checkpoint
checkpoint["model_state_dict"] = model.state_dict()
checkpoint["vocab_size"] = VOCAB_SIZE_NEW
checkpoint["eos_token"] = "<|eos|>"
checkpoint["eos_token_id"] = 35000

torch.save(checkpoint, OUTPUT_PATH)

print(f"Saved expanded checkpoint to {OUTPUT_PATH}")
print("Embedding shape:", tuple(model.embedding.weight.shape))
print("LM head shape:", tuple(model.lm_head.weight.shape))