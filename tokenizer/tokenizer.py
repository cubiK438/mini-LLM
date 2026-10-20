from tokenizers import ByteLevelBPETokenizer
from tokenizers import AddedToken
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

vocab_path = os.path.join(base_dir, "trained_tokenizer", "vocab.json")
merges_path = os.path.join(base_dir, "trained_tokenizer", "merges.txt")

tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

def tokenize(text):
    encoded = tokenizer.encode(text)
    return encoded.ids

def tokenize_batch(texts):
    encoded = tokenizer.encode_batch(texts)
    return [item.ids for item in encoded]

def decode(tokens):
    return tokenizer.decode(tokens)

EOS_TOKEN = "<|eos|>"
tokenizer.add_special_tokens([AddedToken(EOS_TOKEN, special=True)])

print("EOS ID:", tokenizer.token_to_id(EOS_TOKEN))
print("Vocabulary size:", tokenizer.get_vocab_size())
save_dir = os.path.join(base_dir, "trained_tokenizer_sft")
os.makedirs(save_dir, exist_ok=True)

tokenizer.save_model(save_dir)