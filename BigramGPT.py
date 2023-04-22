import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from email.parser import Parser
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

batch_size = 64  # Number of sequences processed in parallel
block_size = 256  # Max content length for predictions
max_iters = 5000  # was 3000 with lr=1e-2
eval_interval = 500  # was 300 to match max_iters
learning_rate = 3e-4  # was 1e-2 then 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # try to use pytorch's CUDA for GPU parallel processing
eval_iters = 200
num_embeddings = 384  # this number was chosen because 384/6 = 64 (standard)
num_heads = 6
num_layers = 6

# dropout is a way to prevent overfitting in large neural networks. it works by having every forward-backward pass
# randomly shut off a subset of neurons(set to 0). It basically splits trianing into multiple sub-networks then
# re-merges them at testing time.
dropout = 0.2  # link here: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

torch.manual_seed(1337)

# AI's database

directory_names = []
file_names = []
root_dir = "C:\\Users\\Admin\\PycharmProjects\\MyGPT\\wow\\maildir"
counter = 0
for i in os.listdir(root_dir):
    current_dir = root_dir + "\\" + i
    for j in os.listdir(current_dir):
        if j == "all_documents":
            directory_names.append(current_dir + "\\" + j)
            # print(current_dir+"\\"+j)
            break

data_text = ""
print(len(directory_names))
for x, dir_name in enumerate(directory_names):
    # print("NAME:", dir_name)
    for path in os.listdir(dir_name):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_name, path)):
            file_name = dir_name + "\\" + path
            with open(file_name, "r") as f:
                data = f.read()
            email = Parser().parsestr(data)
            body_text = email.get_payload()
            data_text = data_text + "\n" + body_text
            break

data_text = data_text.strip()  # strip new lines/white space
# words = word_tokenize(data_text)
# english_text_list = [word for word in words if word not in stopwords.words('English')]
# english_data_text = " ".join(english_text_list)
# AI's database END
# ----------------------------------

chars = sorted(list(set(data_text)))
vocab_size = len(chars)

#----------------------------
# Byte-Pair Encoder START
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")




# Byte-Pair Encoder END
#----------------------------
#t
# Encoder/Decoder | Character tokenizer
# Map unique character dictionary to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # String to int
decode = lambda l: ''.join([itos[i] for i in l])  # Int to string

# Split input data into train/test data - uses a 90%/10% split
data = torch.tensor(encode(data_text), dtype=torch.long)
bpe_data = torch.tensor(encoding.encode(data_text), dtype=torch.long)
print("BPEBPEBPE")
print(bpe_data)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# load data
def load_batch(split):
    # Generate small batch of data using inputs 'x' and targets 'y'
    bpe_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(bpe_data) - block_size, (batch_size,))
    x = torch.stack([bpe_data[i:i + block_size] for i in ix])
    y = torch.stack([bpe_data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# gets rid of noise when getting loss. Averages the splits instead of randomly sampling(which creates noise)
@torch.no_grad()  # tell's pytorch we AREN'T using backpropagation, saving memory
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = load_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Self-Attention model
# The reason this is self-attention is because the keys/queries/values all come from the same source
# (source = x) <- see forward function

# the line where we apply masking to the weights is what makes this self-attention model a decoder.
# not allowing it to communicate with future nodes.

# if we wanted to perform something like sentiment analysis, where nodes need to all talk to each
# other(including future nodes) then we can simply delete the line where we use masking w/ tril
class Head(nn.Module):
    # ONE head of self-attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # T
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores using openAI's described "Scaled Dot-Product attention" formula
        weights = q @ k.transpose(-2, -1) * C ** -0.5  # (B,T,C) @ (B,C,T) = (B,T,T)

        # we only want current nodes using themselves and past nodes to make guesses for future nodes.
        # we DONT want current nodes to talk to future nodes for info. we use pytorch's tril to achieve this.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)

        weights = F.softmax(weights, dim=-1)  # (B,T,T)

        weights = self.dropout(weights)

        # perform weighted aggregation of values
        v = self.value(x)  # (B,T,C)
        out = weights @ v  # (B,T,T) @ (B,T,C) = (B,T,C)
        return out


# Run multiple heads in parallel
class MultipleHeads(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_embeddings, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # concatenating over the channel dimension(C)
        out = self.dropout(self.projection(out)) # apply projection
        return out


# feedforward consisting of a linear layer, follow by a ReLu nonlinear function
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # the default activation when developing our multilayer Perceptron
            nn.Linear(4 * n_embd, n_embd), # projection layer going back into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x )


# Transformer block: communication followed by computation
class Block(nn.Module):

    def __init__(self, n_embd, num_head):
        super().__init__()
        head_size = n_embd // num_head
        self.sa = MultipleHeads(num_head, head_size)
        self.ffwd = FeedForward(n_embd)

        #Pytorch's pre-norm formulation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # each token reads off the logits for the next token using lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.blocks = nn.Sequential(*[Block(num_embeddings, num_head=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(num_embeddings)
        self.lm_head = nn.Linear(num_embeddings, vocab_size)

    # forward feeding
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are (B,T) tensors of integers
        token_embeddings = self.token_embedding_table(idx)  # (B,T,embeddings)
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_embeddings + positional_embeddings  # encode info w/ tok & pos embeddings(B,T,C)
        x = self.blocks(x)  # apply multiple heads of self-attention(feed x into head). (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #  idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # Transforms from (B, T) to (B, C)

            # applying softmax to get probablilities
            probs = F.softmax(logits, dim=-1)  # Also (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Add sample to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)  # CUDA!!1!1

# Model's parameter count
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# using Pytorch's adamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(iter)
    # Every evaluation interval, evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = load_batch('train')

    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
print("GENERATING SAMPLE TEXT")
for _ in range(10):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(encoding.decode(m.generate(context, max_new_tokens=2500)[0].tolist()))
    print("------------------------------------------------------")
