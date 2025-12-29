Recall the Attention operation:

$$\text{Attention}(Q,K,V)=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q$ has dimension (batch-size, $\ldots$, seq-len-q, $d_k$)
- $K$ has dimension (batch-size, $\ldots$, seq-len-k, $d_k$)
- $V$ has dimension (batch-size, $\ldots$, seq-len-k, $d_v$)

In the case of masked attention, we are also given a mask of dimension (batch-size, $\ldots$, seq-len-q, seq-len-k).

**Intuition of** $\mathbf{QK^T}$ 

After the $QK^T$ operation, we obtain a tensor with dimension (batch-size, $\ldots$, seq-len-q, seq-len-k).

Intuitively, assuming a batch-size of 1, $QK^T$ results in a 2D matrix of dimension (seq-len-q, seq-len-k).
In this case, each element $x_{ij}$ represents the amount of attention that the $i$-th token of q should pay to $j$-th token of k.
Note that q and k are not necessarily the same, such as in the case of cross-attention, where q is computed from the decoder while k is computed from the encoder (think of cases like translation).

**Intuition of Masking**

Additionally, in the case of training causal decoders (e.g., GPTs), we want to ensure that the future tokens are masked, ensuring that the current token output is only conditioned on the past tokens.
This is performed by masking, which is to set attention output of $QK^T$ to -$\infty$ for future tokens (i.e., $x_{ij} \rightarrow -\infty, \, \forall j>i$).

Since $QK^T$ results in a tensor of dimension (batch-size, $\ldots$, seq-len-q, seq-len-k), the mask also needs to have the same dimension.

In practice, this mask is a lower triangular matrix (i.e., upper triangle set to False), and can be easily implemented with `torch.tril`.

**Which dimension to take softmax over?**

Recall that $QK^T$ results in a tensor of dimension (batch-size, $\ldots$, seq-len-q, seq-len-k).

Given that we want to know how much token $i$ in $Q$ (i.e., token in position $i$) should attend to each token $j$ in $K$, this means that we should normalize over all tokens in $K$ for each token $i$, hence we take the softmax over the final axis.

**Intuition of multiplying by V**

After computing the softmax, multiplying with $V$ is akin to taking a weighted average of the value of each token (in $K$).

**Implementing multi-head attention**

Multi-head comes from the fact that rather than computing a single Q, K and V, we compute multiple QKVs in parallel for every input.
In other words, we have multiple $W_q$, $W_k$ and $W_v$.

Recall that $W_q$ and $W_k$ each have dimension $d_k \times d_\text{model}$ (that are used to multiply with input $x$ with dimension $\text{batch} \times \text{seq\_len}\times d_\text{model}$), while $W_v$ has dimension $d_v \times d_\text{model}$.

Conveniently, in the multi-head setup, $d_k$ and $d_v$ are usually set to $d_\text{model}/\text{num\_head}$. 
Therefore, instead of doing matmuls with $x$ individually for each head, we can simply set $W_q$, $W_k$ and $W_v$ to $d_\text{model} \times d_\text{model}$.
And after multiplying these weights with our input $x$, we can then evenly split on the final axis to obtain the output of each individual head.
This reduces the number of matmuls from $3 \times \text{num\_head}$ to just 3.


Last but not least, since $W_q$, $W_k$ and $W_v$ all share the same dimension, we can further reduce the number of matmuls required to just 1, by combining $W_q$, $W_k$ and $W_v$ into a large matrix of size $3\cdot d_\text{model} \times d_\text{model}$. Then, after multiplying this matrix with our input $x$, we simply split our final axis by 3 (into $W_q$, $W_k$ and $W_v$), then split each evenly again by the final axis to obtain the output of each individual head.

These can be concisely represented with the following code:

```python
# in def __init__(self):
self.qkv_weights = Linear(d_model, 3*d_model)

#...

# in def forward(self, x):
Q,K,V = einx.rearrange("b... seq_len (o head d_k) -> o b... head seq_len d_k", self.qkv_weights(x), head=self.num_heads, o=3)

# compute mask of dim: batch_size x head x seq_len_q x seq_len_k
mask = torch.ones(Q.shape[:-1]+(K.shape[-2],))
mask = torch.tril(mask).bool() # set upper triangular part to False

# perform attention operation (described above)
attn = scaled_dot_product_attention(Q, K, V, mask=mask)
# concat outputs of all heads
attn = einx.rearrange("batch head seq_len_q dv -> batch seq_len_q (head dv)", attn, head=self.num_heads)
return self.o_weight(attn)
```