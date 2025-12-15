# FLOPs and number of training parameters of a transformer
We assume the FLOPS in a Transformer are dominated by matrix multiplications, so our calculation of FLOPs are solely based on matrix multiplications incurred during a Transformer's forward pass.

Additionally, for a matrix `A` with dimension $m\times n$ and `B` with dimension $n \times p$, we assume that a matmul operation `A` $\times$ `B` incurs $2mnp$ FLOPs.
The intuition behind this can be explained with the following:
- Suppose we simplify `A` to a $1\times n$ vector, and `B` to a $n \times 1$ vector, then when computing `A`$\times$`B`, we perform $n$ multiplications and $n$ summations, expensing $2n$ FLOPs in total.
- Now suppose `A` is now a $m \times n$ matrix, and `B` remains unchanged as a $n \times 1$ vector, then this time  `A` $\times$ `B` incurs $m$ times more operations (multiplications and additions) than previously, therefore expensing $2mn$ FLOPs in total.
- Lastly, suppose `A` remains as a $m \times n$ matrix, while `B` is now a $n\times p$ matrix, then we incur $p$ times more multiplications and additions compared to when `B` was a $n \times 1$ vector. Therefore, this expenses $2mnp$ FLOPs in total.

Also note that the dimensions of all of our matrix weights are flipped, such that a matmul is performed as $W^T x$ (i.e., `W` requires a transpose), reflecting actual implementation in our code.

Recall the components of a Transformer, in a forward pass:

## 1. Converting inputs into embedding

This involves converting an input of token IDs (`batch_size`$\times$`seq_len`) into a tensor of embeddings, by performing a lookup for each token ID on our embedding matrix, converting our input into a dimension of (`batch_size` $\times$ `seq_len` $\times$ `d_model`).

Note that `seq_len` can be no greater than our `context_length`. 
For the purposes of this calculation, we assume we are passing in longest sentences possible, so `seq_len`==`context_length`.

This is a lookup procedure and no matmuls are involved.

Number of trainable parameters: `vocab_size * d_model`

## 2. Passing token embeddings into the 1st transformer block

Recall that our input (token embeddings) have a dimension of (`batch_size` $\times$ `seq_len` $\times$ `d_model`). 
We now analyze the FLOPS for each operation within a pre-norm transformer block.

<p style="text-align: center;">
  <img src="https://deeprevision.github.io/posts/001-transformer/pre-post-ln.png" style="display: block; margin: auto;" width="70%"/>
<i>Image from https://deeprevision.github.io/posts/001-transformer/pre-post-ln.png </i>
</p>


### i. Taking LayerNorm of input

In our implementation, RMSNorm was used instead of LayerNorm to normalize our input along the `d_model` axis. 
The dimension of output remains unchanged from the input.

This process involves performing element-wise operations and reductions, without matmuls. 

Number of trainable parameters: `d_model`

### ii. Performing a causal multi-head self-attention operation

**a. Converting input into Queries, Keys and Values**

To convert our input `x`  into `Q`, `K` and `V`, we multiply `x` with 3*`num_heads` separate weights, each of dimension `d_model` $\times$ `d_q`(or `d_k` or `d_v`, respectively).

Importantly, in our implementation, $d_q==d_k==d_v$, and $d_k = \frac{d_\text{model}}{\text{num\_heads}}$, so we can fuse all of the operations into one large matmul of `x` $\times$ `W_qkv`.

Input `x` has a dimension of `batch_size` $\times$ `seq_len` $\times$ `d_model`, while `W_qkv` has a dimension of (`d_k`$\cdot$ `num_heads` $\cdot$ 3) $\times$`d_model` = 3 $\cdot$ `d_model` $\times$ `d_model`.
So this means that `x` $\times$ `W_qkv` incurs `batch_size` $\cdot$ `2` $\cdot$ `seq_len` $\cdot$ `d_model` $\cdot$ `3*d_model` FLOPs, which simplifies to `6`$\cdot$`batch_size` $\cdot$ `seq_len` $\cdot$ `d_model`$^2$ FLOPs.

Number of trainable parameters: `3 * d_model * d_model`

**b. Add RoPE**

Next is to add positional encoding to our `Q` and `K`. 
In our implementation, RoPE is used, and this process involves element-wise multiplication without any matmuls.

Number of trainable parameters: `0`

**c. Compute Attention**

Recall the Attention operation:

$$\text{Attention}(Q,K,V)=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

For $QK^T$, both `Q` and `K` have dimensions `batch_size` $\times$ `num_head` $\times$  `seq_len` $\times$ `d_k`.
Since the main matmul occurs only in the final two axis, the computation incurred for this operation is `batch_size` $\cdot$ `num_head` $\cdot$ 2 $\cdot$  `seq_len_q` $\cdot$ `d_k` $\cdot$ `seq_len_k` FLOPs.
Additionally, since we are performing self-attention, i.e., `seq_len_q`==`seq_len_k`, and using the property that `d_model` = `num_head` * `d_k`, we can simplify this to 2 $\cdot$ `batch_size` $\cdot$ `d_model` $\cdot$ `seq_len`$^2$ FLOPs.

We note that both the division by $\sqrt{d_k}$ and the $\text{softmax}$ operation are element-wise and no matmuls are involved.

$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ results in a tensor of dimension `batch_size` $\times$ `num_head` $\times$  `seq_len` $\times$ `seq_len`, while `V` has a dimension of `seq_len` $\times$ `d_v`.
Using the matmul FLOPs formula (introduced at the start), we observe that this operation (multiplying the results of softmax with `V`) incurs the following cost: 
2 $\cdot$ `batch_size` $\cdot$ `num_head` $\cdot$ `d_v` $\cdot$ `seq_len`$^2$ FLOPs.
We can further simplify this with the property `d_model` = `num_head` * `d_v` to obtain 2 $\cdot$ `batch_size` $\cdot$ `d_model` $\cdot$ `seq_len`$^2$ FLOPs.

Adding the two parts, we observe that the Attention computation incurs 4 $\cdot$ `batch_size` $\cdot$ `d_model` $\cdot$ `seq_len`$^2$ FLOPs.

Number of trainable parameters: `0` (no new parameters introduced here)

**d. Computing output projection**

The last step of computing multi-head self-attention is to concatenate all of our attention heads, and perform a matmul with $W_O$.

After completing the attention computation in part **c**, our tensor has a dimension of `batch_size` $\times$ `num_head` $\times$ `seq_len` $\times$ `d_v`.
The attention heads are merged, obtaining `batch_size` $\times$ `seq_len` $\times$ `d_model`. This is then multiplied with $W_O$ of dimension `d_model` $\times$ `d_model`.

This operation incurs 2 $\cdot$ `batch_size` $\cdot$ `seq_len` $\cdot$ `d_model`$^2$ FLOPs.

Number of trainable parameters: `d_model * d_model`

---

Summing part **a** to **d**, we see that the entire causal multi-head self-attention operation incurs: `8`$\cdot$`batch_size` $\cdot$ `seq_len` $\cdot$ `d_model`$^2$ FLOPs + 4 $\cdot$ `batch_size` $\cdot$ `d_model` $\cdot$ `seq_len`$^2$ FLOPs.

Total number of trainable parameters from **a** to **d**: `4 * d_model * d_model`

### iii. Adding residual connection 

This part involves adding two tensors (element-wise) of dimension `batch_size` $\times$ `seq_len` $\times$ `d_model` together. 
No matmuls are involved.

Number of trainable parameters: `0` (none introduced here)

### iv. LayerNorm in the 2nd sublayer

Similar to above, computing `RMSNorm` does not involve any matrix multiplications.

Number of trainable parameters: `d_model`

### v. Feed-forward networks with SwiGLU 

Recall the SwiGLU operation:
$$
\text{SwiGLU}(x) = W_2(\text{SiLU}(W_1 x) \odot W_3 x)
$$

In this stage, there are 3 main matmul operations:
- $\mathbf{W_1 x}$ : $W_1$ has dimension `d_ff`$\times$`d_model`, while $x$ has dimension `batch_size` $\times$ `seq_len` $\times$ `d_model`. This matmul operation incurs the cost of 2 $\cdot$ `batch_size` $\cdot$ `seq_len` $\cdot$ `d_model` $\cdot$ `d_ff` FLOPs.
- $\mathbf{W_3 x}$ : $W_3$  has the same dimension as $W_1$, so it incurs the same cost as $W_1 x$.
- $\mathbf{W_2 z}$ : $W_2$ has dimension `d_model`$\times$`d_ff`, while $z$ has dimension `batch_size` $\times$ `seq_len` $\times$ `d_ff`. This matmul operation incurs the cost of 2 $\cdot$ `batch_size` $\cdot$ `seq_len` $\cdot$ `d_model` $\cdot$ `d_ff` FLOPs.

In total, this sums up to 6 $\cdot$ `batch_size` $\cdot$ `seq_len` $\cdot$ `d_model` $\cdot$ `d_ff` FLOPs.

Number of trainable parameters: `3 * d_model * d_ff`

### vi. Adding residual connection
This step involves the element-wise addition of two tensors of dimension `batch_size` $\times$ `seq_len` $\times$ `d_model`, no matmuls are involved. 

Number of trainable parameters: `0` (none introduced here)

### Total FLOPs/trainable parameters of a transformer block
Summing up all of the above, we obtain:

8 $\cdot$`batch_size` $\cdot$ `seq_len` $\cdot$ `d_model`$^2$  + 

4 $\cdot$ `batch_size` $\cdot$ `d_model` $\cdot$ `seq_len`$^2$ +
 
6 $\cdot$ `batch_size` $\cdot$ `seq_len` $\cdot$ `d_model` $\cdot$ `d_ff` FLOPs

This simplifies to 2 $\cdot$ `batch_size` $\cdot$ `seq_len` $\cdot$ `d_model` (4 $\cdot$ `d_model` + 2 $\cdot$ `seq_len` + 3 $\cdot$ `d_ff`) FLOPs.

Total number of trainable parameters: `4 * d_model * d_model` + `2 * d_model` + `3 * d_model * d_ff`.

## 3. LayerNorm
Similar to above, computing `RMSNorm` does not involve any matrix multiplications.

Number of trainable parameters: `d_model`

## 4. Final linear output head
In this final stage, we multiply $W_\text{lm\_head}$ with $x$.

$W_\text{lm\_head}$ has a dimension of `vocab_size` $\times$ `d_model`, while $x$ has a dimension of `batch_size` $\times$ `seq_len` $\times$ `d_model`.
This incurs a cost of 2 $\cdot$ `batch_size` $\cdot$ `seq_len` $\cdot$ `d_model` $\cdot$ `vocab_size` FLOPs.

Number of trainable parameters: `vocab_size * d_model`

## 5. Sum
Adding all of the costs, we observe that the cost (in FLOPs) of a forward pass of a transformer (assuming only matmuls are considered) is:

$$ 
2\cdot\text{batch\_size}\cdot \text{seq\_len}\cdot\text{d\_model}\left( \text{vocab\_size}+\text{num\_layers}(4 \cdot \text{d\_model} + 2 \cdot \text{seq\_len} + 3 \cdot \text{d\_ff})\right) 
$$

Total number of trainable parameters:
$$
\text{d\_model}\left( 2\cdot\text{vocab\_size} + 1 + \text{num\_layers}(4\cdot\text{d\_model}+2+3\cdot\text{d\_ff})\right)
$$

Note that $\text{num\_layers}$ is the number of pre-norm transformer blocks in our transformer architecture.

# Analyzing Resource Usage of GPT-2

GPT-2 XL:
- `vocab_size`: 50257
- `context_length`: 1024
- `num_layers`: 48
- `d_model`: 1600
- `num_heads`: 25
- `d_ff`: 6400

## Number of trainable parameters

Using our formula above, we estimate GPT-2 XL to have approximately 2.127 billion trainable parameters.

Assuming each parameter is represented using single-precision floating-point (FP32), which takes up 4 bytes, then 2.127 billion parameter would require approximately 8.508 GB of memory.

## Computing FLOPs required for a forward pass

Assuming our input sequence has `context_length` tokens, and that `batch\_size` is 1, then using our formula above, a forward pass would require:

- QKV + O multiplication: 1.007E12 FLOPs, 22.3%
- Final output layer: 1.646E11 FLOPs, 3.64%
- Attention operation: 3.221E11 FLOPs, 7.14%
- FFN: 3.020E12 FLOPs, 66.9%
- Total: 4.513E12

As we can see, FFN requires the most FLOPs in the forward pass.

We now compare between the FLOPs, and proportion for the different GPT variants:


| GPT-2 variants | small | medium | large | XL |
| --------------- | ------ | ------- | ------- | ------- |
QKV+O Multiplication | 5.798E10, 10.8% | 2.062E11, 14.9% | 4.832E11, 18.4% | 1.007E12, 22.3% |
Final output layer | 7.905E10, 14.7% | 1.054E11, 7.63% | 1.317E11, 5.02% | 1.646E11, 3.64% |
Attention Operation | 3.865E10, 7.18% | 1.031E11, 7.46% | 1.933E11, 7.38% | 3.221E11, 7.14% | 
FFN | 3.624E11, 67.3% | 9.664E11, 70.0% | 1.812E12, 69.2% | 3.020E12, 66.9% | 
Total | 5.381E11 | 1.381E12 | 2.620E12 | 4.513E12 |

## Increasing `context_length` of GPT-2 XL

If we increase GPT-2 XL's context length to 16384, a forward pass now require:

- QKV + O multiplication: 1.611E13 FLOPs, 10.8%
- Final output layer: 2.635E12 FLOPs, 1.76%
- Attention operation: 8.246E13 FLOPs, 55.2%
- FFN: 4.832E13 FLOPs, 32.3%
- Total: 1.495E14

When the context length increases to 16,384, the attention operation, which scales quadratically to context length, becomes the dominant term.