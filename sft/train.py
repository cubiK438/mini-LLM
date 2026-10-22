import sys
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset,DataLoader,Subset
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer_block import Transformer


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

train_path = f"{ROOT}/sft/data/train.jsonl"
val_path = f"{ROOT}/sft/data/val.jsonl"
checkpoint_path = f"{ROOT}/mini-llm-latest.pt"
resume_path = ROOT / "sft" / "sft-checkpoint.pt"

train = pd.read_json(train_path,lines=True)
val = pd.read_json(val_path,lines=True)

class sft_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {"input_ids":row["input_ids"],
                "prompt_length":row["prompt_length"]}

    def __len__(self):
        return len(self.data)

def collator(batches):

    batch_input_ids = []
    batch_attention_mask = []
    batch_prompt_lengths = []
    max_len = max(len(dic["input_ids"]) for dic in batches)

    for dic in batches:
        original_len = len(dic["input_ids"])
        padding_len = max_len-len(dic["input_ids"])
        attention_mask = original_len * [1] + padding_len * [0]
        padded_ids = dic["input_ids"] + padding_len * [35000]

        batch_input_ids.append(padded_ids)
        batch_attention_mask.append(attention_mask)
        batch_prompt_lengths.append(dic["prompt_length"])
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "prompt_length": torch.tensor(batch_prompt_lengths, dtype=torch.long)
    }



def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lengths = batch["prompt_length"]

            labels = input_ids.clone()

            for i, prompt_len in enumerate(prompt_lengths):
                labels[i, :prompt_len] = -100

            labels[attention_mask == 0] = -100

            inputs = input_ids[:, :-1]
            targets = labels[:, 1:]
            input_attention_mask = attention_mask[:, :-1]

            logits = model(inputs, mask=input_attention_mask)

            # Sum token losses instead of averaging each batch separately
            loss_sum = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum"
            )

            total_loss += loss_sum.item()
            total_tokens += (targets != -100).sum().item()

    model.train()

    return total_loss / total_tokens


train_dataset = sft_dataset(train)
val_dataset = sft_dataset(val)
val_subset = Subset(
    val_dataset,
    range(min(100, len(val_dataset)))
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collator
)
val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collator
)

val_subset_loader = DataLoader(
    val_subset,
    batch_size=2,
    shuffle=False,
    collate_fn=collator
)

print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))
print("Steps per epoch:", len(train_loader))

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

vocab_size = 35001
d_model = 768
num_layers = 5
heads = 12
intermediate_dim = 3072
model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    heads=heads,
    intermediate_dim=intermediate_dim
).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

criterion= torch.nn.CrossEntropyLoss(ignore_index=-100)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.01)

start_epoch = 0
global_step = 0

if resume_path.exists():
    checkpoint = torch.load(
        resume_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]

    print("Resuming from epoch:", start_epoch)
    print("Resuming from step:", global_step)

num_epochs = 1

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lengths = batch["prompt_length"]

        labels = input_ids.clone()

        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100

        labels[attention_mask == 0] = -100
        inputs = input_ids[:, :-1]
        targets = labels[:, 1:]
        input_attention_mask = attention_mask[:, :-1]

        optimizer.zero_grad()

        logits = model(inputs, mask=input_attention_mask)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        global_step+=1
        if global_step % 50 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Step {global_step} | "
                f"Train loss: {loss.item():.4f}"
            )

        if global_step % 250 == 0:
            val_loss = evaluate(
                model, val_subset_loader, device, criterion
            )
            print(f"Validation loss (subset): {val_loss:.4f}")
    val_loss = evaluate(model, val_loader, device, criterion)
    print(f"Epoch {epoch + 1} complete | Validation loss: {val_loss:.4f}")

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        resume_path
    )

    print(f"Checkpoint saved to: {resume_path}")