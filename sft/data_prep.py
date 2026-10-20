import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer.tokenizer import tokenize_batch,decode
from pre_processing import preprocess

def build_prompt(row):
    instruction = row["instruction"]
    context = row["context"]

    prompt = f"Instruction:\n{instruction}\n\n"

    if pd.notna(context) and context.strip():
        prompt += f"Context:\n{context}\n\n"

    prompt += "Response:\n"
    return prompt

df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
df["prompt"] = df.apply(build_prompt, axis=1)

df["prompt_ids"] = tokenize_batch(df["prompt"].to_list())
df["prompt_length"] = df["prompt_ids"].apply(len)

df["response_ids"] = tokenize_batch(df["response"].to_list())

df["input_ids"] = [
    prompt_ids + response_ids + [35000] for prompt_ids, response_ids in zip(df["prompt_ids"], df["response_ids"])]
df["input_length"] = df["input_ids"].apply(len)

df = df[df["input_length"] <= 1024].copy() #model can only take in 1024 input tokens at a time

train, val = train_test_split(df,test_size=0.1,random_state=42)

print(train.shape,val.shape)

train[["input_ids","input_length","prompt_length"]].to_json(f"{ROOT}/sft/data/train.jsonl",orient = "records",lines = True)
val[["input_ids","input_length","prompt_length"]].to_json(f"{ROOT}/sft/data/val.jsonl",orient = "records",lines = True)
