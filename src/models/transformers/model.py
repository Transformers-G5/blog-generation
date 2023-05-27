# import TextProcessor
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

from models.transformers.textprocessor import TextProcessor


class Main:
    def __init__(self,   vocab_size=None, block_size=64, device=None, n_embd=384, n_layer=6, n_head=6, drop_out=0.2) -> None:
        self.textProcessor = TextProcessor()
        self.block_size = block_size
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.drop_out = drop_out
        self.vocab_size = vocab_size

    def build(self,  model_path, vocab_path):
        # masked attention
        parent_self = self
        self.textProcessor.build(path=vocab_path)
        self.vocab_size = self.textProcessor.get_vocab_size()

        class Head(nn.Module):
            def __init__(self, head_size, k_hid_dim=100, q_hid_dim=100, v_hid_dim=100) -> None:
                super().__init__()
                self.key = nn.Linear(parent_self.n_embd, k_hid_dim, bias=False)
                self.key_f = nn.Linear(k_hid_dim, head_size, bias=False)

                self.query = nn.Linear(
                    parent_self.n_embd, q_hid_dim, bias=False)
                self.query_f = nn.Linear(q_hid_dim, head_size, bias=False)

                self.value = nn.Linear(
                    parent_self.n_embd, v_hid_dim, bias=False)
                self.value_f = nn.Linear(v_hid_dim, head_size, bias=False)

                self.register_buffer('tril', torch.tril(torch.ones(
                    parent_self.block_size, parent_self.block_size)))

                self.dropout = nn.Dropout(parent_self.drop_out)

            def forward(self, x):
                B, T, C = x.shape
                k = self.key_f(self.key(x))  # (B, T, C)
                q = self.query_f(self.query(x))  # (B, T, C)

                # compute attention scores
                # (B, T, C) $ (B, C, T) ---> (B, T, T)
                wei = q @ k.transpose(-2, -1) * C ** (-0.5)
                wei = wei.masked_fill(
                    self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
                wei = F.softmax(wei, dim=-1)  # (B, T, T)
                wei = self.dropout(wei)
                # perform the weighted aggrigation of values
                v = self.value_f(self.value(x))
                out = wei @ v    # (B, T, T) @ (B, T, C) ---> (B, T, C)

                return out

        class MultiHeadAttention(nn.Module):
            def __init__(self, num_heads, head_size):
                super().__init__()
                self.heads = nn.ModuleList(
                    [Head(head_size=head_size) for _ in range(num_heads)])
                self.projection = nn.Linear(
                    parent_self.n_embd, parent_self.n_embd)
                self.dropout = nn.Dropout(parent_self.drop_out)

            def forward(self, x):
                # concatination over channel direction
                out = torch.cat([h(x) for h in self.heads], dim=-1)
                return self.dropout(self.projection(out))

        class FeedForward(nn.Module):
            def __init__(self, n_embd):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),    # 4  is from paper
                    nn.ReLU(),
                    nn.Linear(4 * n_embd, n_embd),  # projection
                    nn.Dropout(parent_self.drop_out)
                )

            def forward(self, x):
                return self.net(x)

        class DecoderBlock(nn.Module):
            # Transformer block: communication followed by computation
            def __init__(self, n_embd, n_head):
                super().__init__()
                head_size = n_embd // n_head
                self.sa = MultiHeadAttention(n_head, head_size)
                self.ffwd = FeedForward(n_embd)
                self.ln1 = nn.LayerNorm(n_embd)
                self.ln2 = nn.LayerNorm(n_embd)

            def forward(self, x):
                # "x = x + " is residual connection
                x = x + self.ln1(self.sa(x))
                x = x + self.ln2(self.ffwd(x))
                return x

        class LanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                # each token directly reads of the logits for the next token from a lookup table
                self.token_embedding_table = nn.Embedding(
                    parent_self.vocab_size, parent_self.n_embd)
                # each token of the block size will get it's own positional embeding vector
                self.position_embedding_table = nn.Embedding(
                    parent_self.block_size, parent_self.n_embd)
                # self.sa_heads = MultiHeadAttention(4, n_embd // 4)  # 4 heads of 8-dimentional self attention
                self.blocks = nn.Sequential(
                    *[DecoderBlock(n_embd=parent_self.n_embd, n_head=parent_self.n_head) for _ in range(parent_self.n_layer)]
                )
                self.ln_f = nn.LayerNorm(parent_self.n_embd)
                # self.ffwd = FeedForward(self.n_embd)
                self.lm_head = nn.Linear(
                    parent_self.n_embd, parent_self.vocab_size)

            def forward(self, idx, targets=None):
                B, T = idx.shape
                # idx and targets are both (B, T) tensors of integers
                # (B, T, C) = Batch size, Time = self.block_size, Channel=Vocab_Size
                tok_emb = self.token_embedding_table(idx)
                pos_emb = self.position_embedding_table(torch.arange(
                    T, device=parent_self.device))  # create positional embedding for token
                x = tok_emb + pos_emb  # add the postional embedding with the token embedding
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

            def generate(self, idx, max_new_token, stop_at=None):

                if stop_at is not None:
                    stop_at = parent_self.textProcessor.encode([stop_at])[0]
                # idx is (B, T) matrix of indices of current context
                inital_len = idx.shape[1]

                for _ in range(max_new_token):
                    # crop the idx to last block size tokens
                    idx_cond = idx[:, -parent_self.block_size:]
                    # get the predictions
                    logits, loss = self(idx_cond)
                    # focus only on the last timesetp
                    logits = logits[:, -1, :]  # becomes  (B, C)
                    # apply softmax to get probablities
                    probs = F.softmax(logits, dim=-1)  # (B, C)
                    # sample from distribution
                    idx_next = torch.multinomial(
                        probs, num_samples=1)  # (B, 1)

                    if stop_at is not None and idx_next.item() == stop_at:
                        return idx[:, inital_len:]

                    idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

                return idx[:, inital_len:]

            def generate_realtime(self, idx, max_new_token):
                # idx is (B, T) matrix of indices of current context
                for _ in range(max_new_token):
                    # crop the idx to last block size tokens
                    idx_cond = idx[:, -parent_self.block_size:]
                    # get the predictions
                    logits, _ = self(idx_cond)
                    # focus only on the last timesetp
                    logits = logits[:, -1, :]  # becomes  (B, C)
                    # apply softmax to get probablities
                    probs = F.softmax(logits, dim=-1)  # (B, C)
                    # sample from distribution
                    idx_next = torch.multinomial(
                        probs, num_samples=1)  # (B, 1)

                    idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
                    yield idx_next

        class TextGenerator:
            def __init__(self, model_path):
                # self.model = torch.load(model_path, map_location=torch.device(parent_self.device) )
                self.model = LanguageModel()
                self.model.load_state_dict(torch.load(
                    model_path, map_location=parent_self.device))

                self.start_tokens = torch.zeros((1, 1), dtype=torch.long)
                self.encode = parent_self.textProcessor.encode
                self.decode = parent_self.textProcessor.decode
                self.tokenize = parent_self.textProcessor.tokenizer

            def generate(self, inputs, max_len=500, verbose=False, stop_at=None):

                prompt = inputs['prompt']
                prompt = torch.tensor([parent_self.textProcessor.encode(
                    parent_self.textProcessor.tokenizer(prompt))], dtype=torch.long, device=parent_self.device)
                if verbose:
                    print("generating..")
                result = parent_self.textProcessor.decode(self.model.generate(
                    prompt, max_new_token=max_len, stop_at=stop_at)[0].tolist())
                return result

            def generate_realtime(self, inputs, max_len=500, callback=lambda x: print(x, end=' '), line_break=20, delay_seconds=None):

                prompt = inputs['prompt']
                prompt = torch.tensor([parent_self.textProcessor.encode(
                    parent_self.textProcessor.tokenizer(prompt))], dtype=torch.long, device=parent_self.device)
                if delay_seconds is not None:
                    import time
                for i, token in enumerate(self.model.generate_realtime(prompt, max_len)):
                    token = parent_self.textProcessor.decode(token[0].tolist())
                    callback(token)
                    if i % line_break == 0:
                        # print(' ')
                        callback('\n')
                    if delay_seconds is not None:
                        time.sleep(delay_seconds)

        return TextGenerator(model_path=model_path)


if __name__ == '__main__':
    main = Main()
    # main.config(encoder_dict_path='./models/inspirational_encoder.pkl', decoder_dict_path='./models/inspirational_decoder.pkl')
    tg = main.build(model_path='production/blog/blogs_v8.1_loss_0-2.06_model',
                    vocab_path='production/blog/blogs_vocab_v8.1_0.06')

    # result = tg.generate(inputs={"prompt":"Life can be simple again"}, max_len=100, verbose=True)
    # print(result)
    prompt = 'life can be simple again'
    print(prompt)
    tg.generate_realtime(inputs={"prompt": prompt},
                         max_len=100, delay_seconds=0.1)
