import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
import os


class Main:
  def __init__(self, vocab_size=None, block_size=256, device=None, n_embd=384, n_layer=6, n_head=6, drop_out=0.2) -> None:

    self.block_size = block_size
    self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
    self.n_embd = n_embd
    self.n_layer = n_layer
    self.n_head = n_head
    self.drop_out = drop_out


  def config(self, encoder_dict_path, decoder_dict_path):
    import pickle
    with open(encoder_dict_path, "rb") as ef:
      self.stoi = pickle.load(ef)
    with open(decoder_dict_path, "rb") as df:
      self.itos = pickle.load(df)

    # print(self.stoi)
    # print(self.itos)

    self.vocab_size = len(self.stoi)
    self.encode = lambda s : [ self.stoi[c] for c in s ]
    self.decode = lambda l : "".join( [self.itos[i] for i in l])
    # print("vocab size", self.vocab_size)


    

  def build(self, model_path):
    #masked attention
    parent_self = self
    class Head(nn.Module):
      def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(parent_self.n_embd, head_size, bias=False)
        self.query = nn.Linear(parent_self.n_embd, head_size, bias=False)
        self.value = nn.Linear(parent_self.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(parent_self.block_size, parent_self.block_size)))

        self.dropout = nn.Dropout(parent_self.drop_out)


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
        self.projection = nn.Linear(parent_self.n_embd, parent_self.n_embd)
        self.dropout = nn.Dropout(parent_self.drop_out)

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
          nn.Dropout(parent_self.drop_out)
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
        x = x + self.ln1(self.sa(x))     # "x = x + " is residual connection
        x = x + self.ln2(self.ffwd(x))
        return x


    class BiagramLanguageModel(nn.Module):
      def __init__(self):
        super().__init__()
        #each token directly reads of the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(parent_self.vocab_size, parent_self.n_embd)
        self.position_embedding_table = nn.Embedding(parent_self.block_size, parent_self.n_embd)   #each token of the block size will get it's own positional embeding vector
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4)  # 4 heads of 8-dimentional self attention
        self.blocks = nn.Sequential(
          *[ Block(n_embd=parent_self.n_embd, n_head=parent_self.n_head) for _ in range(parent_self.n_layer)]
        )
        self.ln_f =  nn.LayerNorm(parent_self.n_embd)
        # self.ffwd = FeedForward(self.n_embd)
        self.lm_head = nn.Linear(parent_self.n_embd, parent_self.vocab_size)

      def forward(self, idx, targets=None):
        B, T = idx.shape
        #idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) = Batch size, Time = self.block_size, Channel=Vocab_Size
        pos_emb = self.position_embedding_table(torch.arange(T, device=parent_self.device))  # create positional embedding for token
        x = tok_emb + pos_emb   #add the postional embedding with the token embedding
        x = self.blocks(x)
        # x = self.ffwd(x)   #(B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, self.vocab_size)

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
          idx_cond = idx[:, -parent_self.block_size:]
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


  

    class TextGenerator:
      def __init__(self, model_path):
        self.model = BiagramLanguageModel()
        self.model.load_state_dict(torch.load(model_path, map_location=parent_self.device))

        self.start_tokens = torch.zeros((1, 1), dtype=torch.long)

      def generate(self, inputs, max_len=500, verbose=False):
        prompt = inputs['prompt']
        prompt = torch.tensor([parent_self.encode(prompt)], dtype=torch.long, device=parent_self.device)
        if verbose:
          print("generating..")
        result = parent_self.decode(self.model.generate(prompt, max_new_token=max_len)[0].tolist())
        return result
      
    return TextGenerator(model_path=model_path)
    
  


