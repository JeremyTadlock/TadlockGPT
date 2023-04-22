import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import wget

dataset_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
filename = wget.download(dataset_url)


with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Character length of the dataset
print("dataset character length: ", len(text))

# Lists first 1000 characters
print(text[:100])

# Take all characters, create a list to store them in, then sort them.
char_list = sorted(list(set(text)))

# list and length of all unique characters found in dataset
unique_character_count = len(char_list)
print(''.join(char_list))
print(unique_character_count)

# Map characters to integers(Tokenize input.txt)
# Every character is mappped to an integer, and vice versa.
# Currently using a character Tokenizer

# SELF-NOTE: Google uses a different sub-word Tokenizer called "sentencepiece". Maybe look into later

# SELF-NOTE:  OpenAI uses TikToken. This is a byte-pair encoder tokenizer.
stoi = {ch:i for i, ch in enumerate(char_list)}
itos = {i:ch for i, ch in enumerate(char_list)}
encode = lambda s: [stoi[c] for c in s] # Encoder: take string input, give int output
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: take int input, give print output

print(encode("Hello World!"))
print(decode(encode("Hello World!")))

# Using PyTorch to encode dataset into a Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# Splits dataset into a training and testing model. using 90%/10%
cutoff = int(0.9 * len(data))
train_data = data[:cutoff]
test_data = data[cutoff:]

# Split dataset into chunks. The transformer will be trained in chunks
# This way, as the transformer trains it's much less computationally expensive
block_size = 8
train_data[:block_size+ 1]

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"input tensor: {context} | Target: {target}")

torch.manual_seed(1337)
batch_size = 4 # How many sequences will be proccesed in parallel using gpus
block_size = 8 # Max context length for predictions

def get_batch(split):
    # get random offset batches between 0->len(data)-block_size
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack 1d tensors in rows
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #stack 1d tensor in columns
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size): # batch dimension
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target is: {target}")

print("------")

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, unique_character_count):
        super().__init__()

        self.token_embedding_table = nn.Embedding(unique_character_count, unique_character_count)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:

            B, T, C = logits.shape

            # reshape to fit PyTorch's cross_entropy function (from 3d to 2d & 2d to 1d
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            #evaluate log fuction using pytorch cross entropy function
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


m = BigramLanguageModel(unique_character_count)
logits, loss = m(xb, yb)
print(logits.shape)

print("Loss:", loss)
print("Target Loss", -(math.log(1/logits.shape[1])))

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# using PyTorch's AdamW optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):

    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()))