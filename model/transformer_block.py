from .attention import MultiHeadedAttention
from .rms import RMSNorm
from .SwigLUFFN import SwiGLUFFN
import torch.nn as nn
import torch
from .attention import causal_mask

class TransformerBlock(nn.Module):
    def __init__(self,attention,rms1,rms2,swiglu):
        super(TransformerBlock, self).__init__()
        
        self.attention = attention
        self.norm1 = rms1
        self.norm2 = rms2
        self.swiglu = swiglu

    def forward(self,x,mask = None):
        x_norm = self.norm1(x)
        attn = self.attention(x_norm,x_norm,x_norm,mask = mask)
        res = attn + x

        res_norm = self.norm2(res)
        out = self.swiglu(res_norm)
        res_2 = res+out
        return res_2

class Transformer(nn.Module):
    def __init__(self, heads,d_model,num_layers,intermediate_dim,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size,bias=False)

        blocks = []

        for _ in range(num_layers):
            attention = MultiHeadedAttention(h=heads,dim_model=d_model,dropout=0.0)

            rms1 = RMSNorm(d_model)
            rms2 = RMSNorm(d_model)

            swiglu = SwiGLUFFN(
                hidden_dim=d_model,
                intermediate_dim=intermediate_dim
            )

            block = TransformerBlock(attention=attention,rms1=rms1,rms2=rms2,swiglu=swiglu)

            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        causal = causal_mask(seq_len).to(x.device)  # (1, L, L)

        if mask is not None:
            padding = mask[:, None, None, :].bool()  # (B, 1, 1, L)
            causal = causal[:, None, :, :] & padding  # (B, 1, L, L)

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, causal)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits

    
vocab_size = 35000
d_model = 512
num_layers = 4
heads = 8
intermediate_dim = 2048
seq_len = 1024
batch_size = 2

model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    heads=heads,
    intermediate_dim=intermediate_dim
)

batch_size = 2
seq_len = 1024

x = torch.randint(
    0,
    vocab_size,
    (batch_size, seq_len)
)

mask = causal_mask(seq_len)

logits = model(x, mask)

print(logits.shape)


vocab_size = 35000
d_model = 512
heads = 8
num_layers = 4
intermediate_dim = 2048

model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    heads=heads,
    num_layers=num_layers,
    intermediate_dim=intermediate_dim
)

batch_size = 2
seq_len = 16

x = torch.randint(
    0,
    vocab_size,
    (batch_size, seq_len)
)

mask = causal_mask(seq_len)

logits = model(x, mask)

print("Input:", x.shape)
print("Output:", logits.shape)

loss = logits.mean()
loss.backward()

print(model.embedding.weight.grad is not None)
print(model.lm_head.weight.grad is not None)