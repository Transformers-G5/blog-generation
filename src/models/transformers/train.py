import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)





with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

#here are all the unique characters that occures in the dataset
chars = sorted(list(set(text)))
print("chars", chars)


vocab_size = len(chars)
block_size = 256
batch_size = 64
max_itrs = 5000
eval_interval = 300
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4
n_embd = 384
n_layer = 6
n_head = 6
drop_out = 0.2



#create a mapping from characters to integers
#. lookup tables
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s : [ stoi[c] for c in s ]
decode = lambda l : "".join( [itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# split the dataset into train and test
n = int(0.9 * len(data))
train_data = data[ :n]
val_data = data[n: ]


def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint( len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i : i + block_size] for i in ix])
  y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')




class Head(nn.Module):
  def __init__(self, head_size) -> None:
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(drop_out)


  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)   #(B, T, C)
    q = self.query(x) #(B, T, C)

    # compute attention scores
    wei = q @ k.transpose(-2, -1) * C **(-0.5)  #(B, T, C) $ (B, C, T) ---> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  #(B, T, T)
    wei = F.softmax(wei, dim=-1)   #(B, T, T)
    wei = self.dropout(wei)
    #perform the weighted aggrigation of values
    v = self.value(x)
    out = wei @ v    # (B, T, T) @ (B, T, C) ---> (B, T, C)

    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
    self.projection = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(drop_out)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)  #concatination over channel direction
    return self.dropout(self.projection(out))
  

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),    # 4  is from paper
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),  #projection
      nn.Dropout(drop_out)
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  # Transformer block: communication followed by computation
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))     # "x = x + " is residual connection
    x = x + self.ffwd(self.ln2(x))
    return x


class BiagramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    #each token directly reads of the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)   #each token of the block size will get it's own positional embeding vector
    # self.sa_heads = MultiHeadAttention(4, n_embd // 4)  # 4 heads of 8-dimentional self attention
    self.blocks = nn.Sequential(
      *[ Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)]
    )
    self.ln_f =  nn.LayerNorm(n_embd)
    # self.ffwd = FeedForward(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    #idx and targets are both (B, T) tensors of integers
    tok_emb = self.token_embedding_table(idx) # (B, T, C) = Batch size, Time = block_size, Channel=Vocab_Size
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # create positional embedding for token
    x = tok_emb + pos_emb   #add the postional embedding with the token embedding
    x = self.blocks(x)
    # x = self.ffwd(x)   #(B, T, C)
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)

      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_token):
    # idx is (B, T) matrix of indices of current context
    for _ in range(max_new_token):
      #crop the idx to last block size tokens
      idx_cond = idx[:, -block_size:]
      #get the predictions
      logits, loss = self(idx_cond)
      # focus only on the last timesetp
      logits = logits[:, -1, :] #becomes  (B, C)
      #apply softmax to get probablities
      probs = F.softmax(logits, dim=-1) #(B, C)
      #sample from distribution
      idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)

      idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)

    return idx


model = BiagramLanguageModel()
m = model.to(device)

logits, loss = m(xb, yb)

#create optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

#Training
for step in range(max_itrs):
  #sample a batch of data
  xb, yb = get_batch('train')
  #evaluate the loss
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  if step % 100 == 0:
    print(loss.item())

start_tokens = torch.zeros((1, 1), dtype=torch.long)

print(decode(m.generate(start_tokens, 500)[0].tolist()))
