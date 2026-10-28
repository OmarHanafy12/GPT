# imports
import torch
from torch import nn
from torch.nn import functional as f

#Hyperparamters
chunck_size =  8
batch_size = 32
max_iters = 10000
eval_interval = 2000
eval_iters = 200
learning_rate = 0.001


with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# list of unique characters in data set
char = sorted(set(text))
vocab_size = len(char)


# map char to int and vice versa
stoi = {c:i for i,c in enumerate(char)}
itos = {i:c for i,c in enumerate(char)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#encoding the entire data set and store it into a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# train test split
n = int(0.85 * len(data))
train = data[:n]
test = data[n:]

#generates a batch of data
def get_batch(split):
  data = train if split == 'train' else test
  ix = torch.randint(len(data) - chunck_size, (batch_size,))
  x = torch.stack([data[i:i+chunck_size] for i in ix])
  y = torch.stack([data[i+1:i+chunck_size+1] for i in ix])
  return x,y

xb, yb = get_batch('train')

# averages the loss over eval_iters batches for both splits (less noisy than a single batch)
@torch.no_grad()
def estimate_loss(model):
   out = {}
   model.eval()
   for split in ['train', 'test']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
         X, Y = get_batch(split)
         _, loss = model(X, Y)
         losses[k] = loss.item()
      out[split] = losses.mean()
   model.train()
   return out

#define the bigram language model
class BigramLangaugeModel(nn.Module):
  def __init__(self, vocab_size):
     super().__init__()
     self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)

    if targets is None:
      loss = None
    else:
      b,t,c = logits.shape
      logits = logits.view(b*t, c)
      targets = targets.view(b*t)
      loss = f.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)
        logits = logits[:, -1, :] # becomes (B, C)
        probs = f.softmax(logits, dim=-1) # (B, C)
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx


m = BigramLangaugeModel(vocab_size)
logits, loss = m(xb, yb)

# pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for steps in range(max_iters):
  if steps % eval_interval == 0:
    losses = estimate_loss(m)
    print(f"step {steps}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
  xb, yb = get_batch('train')
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(loss.item())
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))