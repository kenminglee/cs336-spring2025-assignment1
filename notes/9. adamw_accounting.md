# Computing memory usage of a forward pass of AdamW

For each component in a forward pass of a transformer, here are the main usage of memory:
- parameters of each component
- activations at each stage (i.e., output after each stage, which are stored for gradient calculations)
- gradients of each parameter (usually same dimension as param)
- optimizer state (in the case of AdamW, this is 2 times the dimension of parameter -- `m` and `v`, which keeps track of 1st and 2nd order momentum of each parameter).

We now list the memory usage of each component:

## Transformer block

Within each (pre-norm) transformer block, below are the component breakdowns, alongside their memory usage.

### RMSNorm (x2)
- Activation: batch_size x context_length x d_model
- Parameters: d_model
- Gradients: d_model
- Optimizer state: 2 x d_model

Total memory usage: batch_size x context_length x d_model + 4 x d_model

### QKV projections
- Activation: batch_size x num_head x context_length x d_k. Since d_k = d_model/num_head, we can simplify this to batch_size x context_length x d_model
- Parameters: 3 x d_model^2
- Gradients: 3 x d_model^2
- Optimizer state: 6 x d_model^2

Total memory usage: batch_size x context_length x d_model + 12 x d_model^2

### $Q^T K$ matmul
- Activation: batch_size x num_head x context_length^2
- No parameters, gradients, or optimizer state

### Softmax operation
- Activation: batch_size x num_head x context_length^2
- No parameters, gradients, or optimizer state

### Weighted sum of $V$
- Activation: batch_size x num_head x context_length x d_v.
Since d_v==d_k and d_k = d_model/num_head, we simplify this to batch_size x context_length x d_model
- No parameters, gradients, or optimizer state

### Output Projection
- Activation: batch_size x context_length x d_model
- Parameters: d_model^2
- Gradients: d_model^2
- Optimizer state: 2 x d_model^2

Total memory usage: batch_size x context_length x d_model + 4 x d_model^2

### Feed-forward Network: $W_1$ and $W_3$ matmul
- Activation: batch_size x context_length x d_ff
- Parameters: d_model x d_ff
- Gradients: d_model x d_ff
- Optimizer state: 2 x d_model x d_ff

Note that d_ff = 4 x d_model.

Total memory usage: 4 x batch_size x context_length x d_model + 16 d_model^2

### Feed-forward Network: SiLU
- Activation: batch_size x context_length x d_ff = 4 x batch_size x context_length x d_model
- No parameters, gradients, or optimizer state

### Feed-forward Network: $W_2$ matmul
- Activation: batch_size x context_length x d_model
- Parameters: d_model x d_ff
- Gradients: d_model x d_ff
- Optimizer state: 2 x d_model x d_ff

Note that d_ff = 4 x d_model.

Total memory usage: batch_size x context_length x d_model + 16 d_model^2

### Transformer block memory usage

- 2xRMSNorm: 2 x batch_size x context_length x d_model + 8 x d_model
- QKV Projection: batch_size x context_length x d_model + 12 x d_model^2
- QK^T + softmax: 2 x  batch_size x num_head x context_length^2
- weighted sum of V: batch_size x context_length x d_model
- output projection: batch_size x context_length x d_model + 4 x d_model^2
- FFN: 8 x batch_size x context_length x d_model + 32 d_model^2 + 4 x batch_size x context_length x d_model + batch_size x context_length x d_model + 16 d_model^2

In total, we use:

18 batch_size x context_length x d_model + 8 d_model + 64 d_model^2 + 2 batch_size x num_head x context_length^2

per transformer block

## Final RMSNorm

Similar to above, we get a total of : 
batch_size x context_length x d_model + 4 x d_model

## Output Embedding
- Activation: batch_size x context_length x vocab_size
- Parameters: vocab_size x d_model
- Gradients: vocab_size x d_model
- Optimizer state: 2 x vocab_size x d_model

Total: batch_size x context_length x vocab_size + 4 x vocab_size x d_model

## Cross-entropy on logits
- Negligible? (single scalar value)

## Total memory usage

num_layers x (18 batch_size x context_length x d_model + 8 d_model + 64 d_model^2 + 2 batch_size x num_head x context_length^2) + batch_size x context_length x d_model + 4 d_model + batch_size x context_length x vocab_size + 4 vocab_size x d_model

# Computing memory usage of GPT-2 XL model:

Recall GPT-2 XL parameters:
- `vocab_size`: 50257
- `context_length`: 1024
- `num_layers`: 48
- `d_model`: 1600
- `num_heads`: 25
- `d_ff`: 6400

Performing these calculations, we get $3.98526\times10^9\cdot\text{batch\_size} + 8.1866\times10^9$ float32 numbers.

Each number takes 4 bytes, hence this takes:
$3.27463424\times10^{10}+1.59410463\times10^{10}\cdot\text{batch\_size}$ bytes.

For a device with 80GB of memory, the largest batch_size that can fit is 2.964 or $\approx 2$.

# Computing FLOPs of one step of AdamW

AdamW does not use any matmuls.
Therefore, our calculation of FLOPs solely based on matrix multiplications remains unchanged.
This turns out to :

 2 $\cdot$ `batch_size` $\cdot$ `context_length` $\cdot$ `d_model` (4 $\cdot$ `d_model` + 2 $\cdot$ `context_length` + 3 $\cdot$ `d_ff`) FLOPs.

# Computing the amount of time it takes to train a GPT-2 XL model

Let's assume:
- We are trying to train a GPT-2 XL model for 400K steps on a batch size of 1024 on a single A100.
- A100 GPU has a theoretical peak FLOP throughput of 19.5 teraFLOP/s for float32 operations. 
We assume that we are able to get 50% Model FLOPs utilization (MFU), so this means 9.75 teraFLOP/s for float32.

Using our FLOP equation from previously, and plugging in GPT-2 XL specifications, we obtain: $9.05969664\times10^{10}\text{batch\_size}$ FLOPs.

With a batch size of 1024, this sums up to $9.27712936\times10^{13}$ for a single forward pass.

Assuming backward pass takes 2 times the amount of FLOPs, a single forward + backward pass takes $2.78313881\times10^{14}$ FLOPs.

So for 400K steps, this would take a total of $1.11325552\times10^{20}$ FLOPs, which would require 11418005.37 seconds, or approximately 132 days.