import shutil
import sys
from tokenizers.implementations import ByteLevelBPETokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F

tokenizer = None
batch_size = None
max_itrs = None
eval_interval = None
eval_iters = None
device = None
learning_rate = None
n_embd = None
n_layer = None
n_head = None
drop_out = None
block_size = None
vocab_size = None
specials = ["[PAD]", "[start]", "[end]", "[UNK]",
            "[MASK]", '[PROMPT_START]', '[PROMPT_END]']


pad_token_id = None


def realtimePrint(line_text):

    columns, _ = shutil.get_terminal_size((80, 20))
    sys.stdout.write("\r" + line_text + " " * (columns - len(line_text)))
    sys.stdout.flush()


def setup(vocab_path, merges_path, block_size, num_embd, num_layer, num_head):
    global batch_size, max_itrs, device, learning_rate, n_embd, n_layer, n_head, drop_out, vocab_size, tokenizer, pad_token_id
    batch_size = 32
    max_itrs = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 3e-4
    n_embd = num_embd
    n_layer = num_layer
    n_head = num_head
    drop_out = 0.2
    block_size = block_size
    tokenizer = ByteLevelBPETokenizer(vocab=vocab_path, merges=merges_path)
    tokenizer.add_special_tokens(specials)
    tokenizer.enable_padding()
    tokenizer.add_tokens(['[UNK]'])
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    print("setup done")

    # masked attention head

    class Head(nn.Module):
        def __init__(self, head_size) -> None:
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)

            self.register_buffer('tril', torch.tril(
                torch.ones(block_size, block_size)))

            self.dropout = nn.Dropout(drop_out)

        def forward(self, x, padding_mask=None):
            B, T, C = x.shape
            k = self.key(x)  # (B, T, C)
            q = self.query(x)  # (B, T, C)

            # compute attention scores
            # (B, T, C) $ (B, C, T) ---> (B, T, T)
            wei = q @ k.transpose(-2, -1) * C ** (-0.5)

            if padding_mask is not None:
                wei = wei.masked_fill(padding_mask.unsqueeze(
                    1)[:T, :T] == 0, float('-inf'))  # (B, T, T)

            wei = wei.masked_fill(
                self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggrigation of values
            v = self.value(x)

            out = wei @ v    # (B, T, T) @ (B, T, C) ---> (B, T, C)

            return out

    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList(
                [Head(head_size=head_size) for _ in range(num_heads)])
            self.projection = nn.Linear(n_embd, n_embd)
            self.dropout = nn.Dropout(drop_out)

        def forward(self, x, padding_mask=None):
            # concatination over channel direction
            out = torch.cat([h(x, padding_mask=padding_mask)
                            for h in self.heads], dim=-1)
            return self.dropout(self.projection(out))

    class FeedForward(nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),    # 4  is from paper
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(drop_out)
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

        def forward(self, x, padding_mask=None):
            # "x = x + " is residual connection
            x = x + self.ln1(self.sa(x, padding_mask=padding_mask))
            x = x + self.ln2(self.ffwd(x))
            return x

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            # each token directly reads of the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            # each token of the block size will get it's own positional embeding vector
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            # self.sa_heads = MultiHeadAttention(4, n_embd // 4)  # 4 heads of 8-dimentional self attention
            self.blocks = nn.Sequential(
                *[DecoderBlock(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)]
            )
            self.ln_f = nn.LayerNorm(n_embd)
            # self.ffwd = FeedForward(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            if targets is not None:
                padding_mask = (~torch.eq(idx, pad_token_id)
                                ).type(torch.float32)
                padding_mask_target = (
                    ~torch.eq(targets, pad_token_id)).type(torch.float32)

            else:
                padding_mask = None

            # idx and targets are both (B, T) tensors of integers
            # (B, T, C) = Batch size, Time = block_size, Channel=Vocab_Size
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(
                T, device=device))  # create positional embedding for token
            x = tok_emb + pos_emb  # add the postional embedding with the token embedding
            # x = self.blocks(x, padding_mask)
            # propagate padding_mask to each DecoderBlock
            x = self.blocks[0](x, padding_mask=padding_mask)
            for block in self.blocks[1:]:
                x = block(x, padding_mask=padding_mask)

            x = self.ln_f(x)
            logits = self.lm_head(x)  # (B, T, vocab_size)

            if targets == None:
                loss = None
            else:
                B, T, C = logits.shape
                # padding_mask =  padding_mask.unsqueeze(-1).expand(-1, -1, C)

                padding_mask = padding_mask.view(B * T, 1)
                padding_mask_target = padding_mask_target.view(B * T, 1)

                logits = logits.view(B * T, C)
                targets = targets.view(B * T)

                loss = F.cross_entropy(logits, targets, reduction='none')

                # loss = loss.masked_fill(padding_mask == 0, torch.tensor(0.0))
                loss = loss.masked_fill(
                    padding_mask_target == 0, torch.tensor(0.0))

                # print(torch.allclose(loss, loss_masked), "loss == loss_masked")
                loss = loss.mean()

            return logits, loss

        def generate(self, idx, max_new_token, stop_token_id=None, verbose=False):
            # idx is (B, T) matrix of indices of current context
            # verbose = False
            idxtopk = idx.clone()
            for _ in range(max_new_token):
                # crop the idx to last block size tokens
                idx_cond = idx[:, -block_size:]
                idx_condtopk = idxtopk[:, -block_size:]
                # get the predictions
                logits, _ = self(idx_cond)
                logits_topk, _ = self(idx_condtopk)
                # focus only on the last timesetp
                logits = logits[:, -1, :]  # becomes  (B, C)
                logits_topk = logits_topk[:, -1, :]

                # apply softmax to get probablities
                probs = F.softmax(logits, dim=-1)  # (B, C)
                probs_topk = F.softmax(logits_topk, dim=-1)
                # sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                _, idx_next_topk = torch.topk(probs_topk, k=1)

                if stop_token_id is not None and idx_next.item() in stop_token_id:
                    break
                if idx_next.item() == tokenizer.token_to_id('[PAD]'):
                    print("pad skipped")
                    continue

                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
                idxtopk = torch.cat((idxtopk, idx_next_topk), dim=1)

                if verbose:
                    realtimePrint(
                        f"num tokens: {len(idx[0].tolist())} | {(tokenizer.decode(idx_next[0].tolist()))}")

            return idx, idxtopk

        def generateWithTemp(self, idx, max_new_token, stop_token_id=None, verbose=False, temperature=1.0):
            # idx is (B, T) matrix of indices of current context
            # verbose = False
            idxtopk = idx.clone()
            for _ in range(max_new_token):
                # crop the idx to last block size tokens
                idx_cond = idx[:, -block_size:]
                idx_condtopk = idxtopk[:, -block_size:]
                # get the predictions
                logits, _ = self(idx_cond)
                logits_topk, _ = self(idx_condtopk)
                # focus only on the last timestep
                logits = logits[:, -1, :]  # becomes (B, C)
                logits_topk = logits_topk[:, -1, :]

                # apply softmax to get probabilities
                probs = F.softmax(logits / temperature, dim=-1)  # (B, C)
                probs_topk = F.softmax(logits_topk / temperature, dim=-1)
                # sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                _, idx_next_topk = torch.topk(probs_topk, k=1)

                if stop_token_id is not None and idx_next.item() in stop_token_id:
                    break
                if idx_next.item() == tokenizer.token_to_id('[PAD]'):
                    print("pad skipped")
                    continue

                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
                idxtopk = torch.cat((idxtopk, idx_next_topk), dim=1)

                if verbose:
                    realtimePrint(
                        f"num tokens: {len(idx[0].tolist())} | {(tokenizer.decode(idx_next[0].tolist()))}")

            return idx, idxtopk

        def generate_realtime(self, idx, max_new_token):
            # idx is (B, T) matrix of indices of current context
            for _ in range(max_new_token):
                # crop the idx to last block size tokens
                idx_cond = idx[:, -block_size:]
                # get the predictions
                logits, _ = self(idx_cond)
                # focus only on the last timesetp
                logits = logits[:, -1, :]  # becomes  (B, C)
                # apply softmax to get probablities
                probs = F.softmax(logits, dim=-1)  # (B, C)
                # sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
                yield idx_next

    class TextGenerator:
        def __init__(self, model_path):
            # self.model = torch.load(model_path, map_location=torch.device(parent_self.device) )
            self.model_path = model_path
            self.model = LanguageModel()
            print("model initialized")
            self.model.load_state_dict(torch.load(
                model_path, map_location=device))
            self.model.eval()
            self.tokenizer = tokenizer
            self.end_token_id = self.tokenizer.token_to_id('[end]')

        def _gen(self,  context_string=" ", start_token="[start]", end_token="[end]", max_len=100, stop_token_id=None, verbose=False, temperature=1.0):
            context_string = start_token + " " + context_string
            context_string = torch.tensor(
                [tokenizer.encode(context_string).ids], dtype=torch.long, device=device)

            gen, gen_topk = self.model.generateWithTemp(
                context_string, max_len, stop_token_id=stop_token_id, verbose=verbose, temperature=temperature)
            gen = gen[0].tolist()
            gen_topk = gen_topk[0].tolist()

            gen_text = tokenizer.decode(gen)
            gen_text_topk = tokenizer.decode(gen_topk)
            return gen_text, gen_text_topk

        def generate(self, inputs, max_len=500, verbose=False, stop_token_id=None, temperature=1.0):

            prompt = inputs['prompt']

            if verbose:
                print("generating..")
            result, result_topk = self._gen(prompt, max_len=max_len,
                                            stop_token_id=stop_token_id, verbose=verbose, temperature=temperature)
            return result, result_topk

        def model_info(self):
            total_params = sum(p.numel() for p in self.model.parameters())
            total_trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            print("==========")
            print(self.model_path)
            print("-"*len(self.model_path))
            print("Total params:", total_params)
            print("Total trainable params:", total_trainable_params)
            print("==========")
    return TextGenerator


if __name__ == '__main__':

    # tg = TextGenerator()
    pass
