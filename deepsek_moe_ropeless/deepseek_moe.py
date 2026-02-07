import torch
import torch.nn as nn
import torch.nn.functional as F

# expert NN
class ExpertNN(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.net= nn.Sequential(
        nn.Linear(input_dim, 4*input_dim),
        nn.GELU(),
        nn.Linear(4*input_dim, input_dim),
        nn.Dropout(dropout)
    )

  def forward(self,x):
    return self.net(x)
  
#Implementing the Router (input_from_MLH*Routing Matrix= Expert_selector_matrix)  
num_experts= 3
top_k= 2
n_embed= 8

mh_output = torch.rand(1, 4, n_embed)
topkgate_linear = nn. Linear(n_embed, num_experts) 
expert_selector_matrix = topkgate_linear(mh_output)
print(expert_selector_matrix)


#Implementing top_k load balancing
top_k_logits, top_k_indices = expert_selector_matrix.topk(top_k, dim=-1) # Get top-k experts
top_k_logits, top_k_indices

# So now we will all these things inside a class as Top_k_routing

class TopkRouter(nn.Module):
  def __init__(self, n_embed, num_experts, top_k):
    super(TopkRouter, self).__init__()
    self.top_k = top_k
    self.linear = nn.Linear(n_embed, num_experts)

  def forward (self, mh_output):
    # mh_output is the output tensor from multihead self attention block
    expert_selector_matrix = self.linear(mh_output)
    top_k_logits, indices = expert_selector_matrix.topk(self.top_k, dim=-1)
    zeros = torch.full_like(expert_selector_matrix, float('-inf'))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_output = F.softmax(sparse_logits, dim=-1)
    return router_output, indices
  

#Creating the Sparse MOE (After acquiring the expert selector weight matrix, the top k values are selectively 
# multiplied with the outputs from the corresponding top-k experts for a given token.)  

class SparseMoE (nn.Module) :
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self. router = TopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([ExpertNN(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
    def forward(self, x):
        gating_output, indices = self. router(x)
        final_output = torch. zeros_like (x)
        # Reshape inputs for batch processing
        flat_x = x. view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i). any (dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert (expert_input)
                #Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i]. unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output [expert_mask] += weighted_output. squeeze(1)

        return final_output
    


num_experts = 3
top_k = 2
n_embd = 8
dropout=0.1

mh_output = torch.randn(1, 4, n_embd) 
sparse_moe = SparseMoE (n_embd, num_experts, top_k)
final_output = sparse_moe(mh_output)
print ("Shape of the final output:", final_output. shape)
print (final_output)    




# Transformer Block
# multi Head Attention
class Head(nn.Module):  # this is the single head 
  "' one head of self-attention "''
  def __init__(self, head_size):
    super().__init__()
    self.key = nn. Linear(n_embed, head_size, bias=False)
    self.query = nn. Linear(n_embed, head_size, bias=False)
    self. value = nn. Linear(n_embed, head_size, bias=False)
    self. register_buffer('tril', torch.tril(torch.ones (block_size, block_size)) )
    self.dropout = nn. Dropout (dropout)

  def forward (self, x):
    B, T,C = x. shape
    k = self.key(x)
    # (B,T,C)
    q = self.query (x) # (B,T,C)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) → (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax (wei, dim=-1) # (B, T, T)
    wei = self. dropout (wei)
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,C)
    out = wei @ v # (B, T, T) @ (B, T, C) → (B, T, C)
    return out
  
# multi head 
class MultiHeadAttention(nn.Module):
  ''''" multiple heads of self-attention in parallel "'''
  def __init__(self, num_heads, head_size):
    super() .__init__()
    self.heads = nn.ModuleList ([Head(head_size) for _ in range(num_heads) ])
    self.proj= nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout (self.proj(out))
    return out
  


# Assembling all the transformer layer 
# creating a self attention + mixture of experts block, that may be repeated several number of times
class Block(nn.Module):
  ''''' Mixture of Experts Transformer block: communication followed by computation (multi-head self attention)'''''
  def __init__(self, n_embed, n_head, num_experts, top_k):
    #n_embed: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.smoe = SparseMoE(n_embed, num_experts, top_k)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward (self, x):
    x = x + self.sa(self.ln1(x))
    x= x + self.smoe(self.ln2(x))
    return x
  

# Entire Language Model Inference

class SparseMoELanguageModeL(nn.Module):
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding (vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding (block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, num_experts=num_experts, top_k=top_k) for _ in range (n_layer)])
    self.ln_f = nn.LayerNorm(n_embed) # final layer norm
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward (self, idx, targets=None):
    B, T = idx.shape
    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
    x= tok_emb + pos_emb # (B, T,C)
    x = self.blocks(x) # (B, T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head (x) # (B,T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits. shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate (self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range (max_new_tokens) :
      # crop id to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, loss = self (idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :]# becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
      idx_next = torch. multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx





# we have used first the shakespeare dataset here which karpathy also used in his gpt architecture 
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

torch.manual_seed (1337)

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
chars = sorted (list (set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode= lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode= lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch. tensor (encode (text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split) :
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch. stack([data[i:i+block_size] for i in ix])
  y = torch. stack( [data [i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y




# Defining the LLM loss
@torch.no_grad
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch. zeros (eval_iters)
    for k in range(eval_iters) :
      X, Y = get_batch(split)
      logits, loss = model (X, Y)
      losses [k] = loss.item()
    out [split] = losses. mean ()
  model.train()
  return out


# training hyperparamters 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 20
eval_interval = 100
learning_rate = 1e-3
eval_iters = 400
head_size = 16
n_embed = 128
n_head = 8
n_layer = 8
dropout = 0.1
num_experts = 8
top_k = 2



# Model Initialization with Kaiming Initialzation
#Kaiming initialization (or He initialization) is a weight initialization technique for deep neural networks, specifically 
# designed to work with ReLU activation functions to prevent vanishing/exploding gradients by adjusting weight variance 
# based on the number of input connections (fan-in), ensuring stable signal flow and faster convergence, unlike Xavier 
# initialization which suits symmetric activations like tanh. It effectively compensates for ReLU's tendency to zero out half
#  the neurons, maintaining consistent activation variance across layers for deeper, more stable training.

def kaiming_init_weights(m):
  if isinstance (m, (nn.Linear)) :
    init.kaiming_normal_(m.weight)

model= SparseMoELanguageModeL()
model.apply(kaiming_init_weights)



# pretraining loop of the model here 
m = model.to(device)
print(sum(p.numel() for p in m.parameters ())/1e6,'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss ()
    print (f"step fiter): train loss {losses ['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model (xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward ()
    optimizer.step()




