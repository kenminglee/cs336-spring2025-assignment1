
In short, Relative Positional Embeddings (RoPE), is one of the ways to inject positional awareness into our query and key embeddings.
It does so by  multiplying a (pairwise) rotation matrix $R^i$ with query and key tokens.

Assuming a batch size of 1, for every token $i$ (i.e., the $i$-th token in a sentence, represented as $z^i$ with dimension $1\times d_\text{model}$), multiply it with $W_q$ or $W_k$ of dimension $d_k \times d_\text{model}$ (converting it to a query or key, respectively), and then rotate it by multiplying it with the rotation matrix $R^i$ of dimension $d_k \times d_k$.
This is written as $R^i W_q z^i$ in short.

# Rotation Matrix $R^i$

Let $d=d_k$, then the full rotation matrix $R^i$ is given by:

$$
R^i = 
\begin{bmatrix}
R_1^i & 0 & 0 & \ldots & 0\\
0 & R_2^i & 0 & \ldots & 0\\
0 & 0 & R_3^i & \ldots & 0\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
0 & 0 & 0 & \ldots & R_{d/2}^i
\end{bmatrix}
$$

where each $0$ is a zero $2\times2$ matrix and $ R_k^i = 
\begin{pmatrix} 
\cos\theta_{i,k} & -\sin\theta_{i,k}\\
\sin\theta_{i,k} & \cos\theta_{i,k}
\end{pmatrix}$.

Expanding this out, we obtain:

$$
R^i = 
\begin{bmatrix}
\cos\theta_{i,1}  & -\sin\theta_{i,1} & 0 & 0 & \ldots & 0 & 0\\
\sin\theta_{i,1} & \cos\theta_{i,1} & 0 & 0& \ldots & 0 & 0\\
0 & 0 & \cos\theta_{i,2}  & -\sin\theta_{i,2} & \ldots & 0 & 0\\
0 & 0 & \sin\theta_{i,2} & \cos\theta_{i,2} & \ldots & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
0 & 0 & 0 & \ldots  & 0 & \cos \theta_{i, d/2} & -\sin \theta_{i, d/2}\\
0 & 0 & 0 & \ldots  & 0 & \sin \theta_{i, d/2} & \cos \theta_{i, d/2} & \\
\end{bmatrix}
$$

Note that $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$ for $k\in \{1,\ldots, d/2\}$ and $\Theta$ is some constant (e.g., 10000). 
As mentioned previously, the dimension of $R^i$ is $d\times d$.

# Multiplication Operation with Full Matrix

Suppose we want to perform RoPE on our query tokens, then suppose our query token has the following structure (assume batch size of 1):

$$
Q = 
\begin{bmatrix}
x^0_0 & x^0_1 & x^0_2 & \ldots & x^0_{d-1}\\
x^1_0 & x^1_1 & x^1_2 & \ldots & x^1_{d-1}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
x^{N-1}_0 & x^{N-1}_1 & x^{N-1}_2 & \ldots & x^{N-1}_{d-1}\\
\end{bmatrix}
$$

where $N$ is our maximum sequence length, and $d$ is the size of our embedding (i.e., $d_k$).

Then, doing $R^i(Q^i)^T$ (rotation for query token at position $i$; transpose required for dimensional compatibility) equates to:

$$
R^i (Q^i)^T = 
\begin{bmatrix}
x_0^i \cos \theta_{i,1} - x_1^i \sin \theta_{i,1}\\
x_0^i \sin \theta_{i,1} + x_1^i \cos \theta_{i,1}\\

x_2^i \cos \theta_{i,2} - x_3^i \sin \theta_{i,2}\\
x_2^i \sin \theta_{i,2} + x_3^i \cos \theta_{i,2}\\

\vdots\\


x_{d-2}^i \cos \theta_{i,d/2} - x_{d-1}^i \sin \theta_{i,d/2}\\
x_{d-2}^i \sin \theta_{i,d/2} + x_{d-1}^i \cos \theta_{i,d/2}\\
\end{bmatrix}
$$

# Implementing RoPE

Note that for a certain constant $\Theta$ value, $R^i$ only needs to be computed once, and can be reused across batches or attention layers.

However, while it is possible to compute once and store $R^i$ for all $i\in \{0,1,\ldots, \text{max-seq-len}\}$, this is unnecessarily large -- we need to store a tensor of dimension $\text{max-seq-len} \times d \times d$, making it less ideal.

## High-level approach: precomputing sin and cos
Alternatively, suppose we have a list of cosines and sines, $\text{cos\_buf} = [\cos\theta_{i,1}, \cos\theta_{i,2}, \ldots, \cos\theta_{i,d/2}]$ and $\text{sin\_buf} = [\sin\theta_{i,1}, \sin\theta_{i,2}, \ldots, \sin\theta_{i,d/2}]$, then we can perform element-wise multiplication with matrix $Q$ in an _interleaved_ manner, to similarly obtain $R^iQ^i$.

For instance, in the column vector $R^i (Q^i)^T $ above, notice that:
- every even row of $R^i (Q^i)^T$ (assume 0-indexing) can be obtained by even columns of $x^i$ in $Q$ (assume 0-indexing as well) times $\text{cos\_buf}$, minus odd columns of $x^i$ times $\text{sin\_buf}$.
- vice versa, every odd row of $R^i (Q^i)^T$ can be obtained by even columns of $x^i$ times $\text{sin\_buf}$ plus odd columns of $x^i$ times $\text{cos\_buf}$.

Hence, for all $i$ values (representing all token positions), we just need to generalize the case above, such that we precompute $\text{cos\_buf}$ and $\text{sin\_buf}$, both of which have a dimension of $\text{max-seq-len} \times d/2$, such that each row holds the rotational values for a token at a particular position.

With this, we now only need to cache our precomputed $\text{cos\_buf}$ and $\text{sin\_buf}$, reducing our memory requirements to only $\text{max-seq-len} \times d/2 \times 2 = \text{max-seq-len} \times d$

## Computing theta matrix

To precompute $\text{sin\_buf}$ and $\text{cos\_buf}$, we first need to compute a matrix of $\theta_{i,k}$, before taking the $\cos$ and $\sin$ of it:

$$
\begin{bmatrix}
\theta_{0,0} & \theta_{0,1} & \theta_{0,2} & \ldots & \theta_{0, d/2}\\
\theta_{1,0} & \theta_{1,1} & \theta_{1,2} & \ldots & \theta_{1, d/2}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
\theta_{N-1, 0} & \theta_{N-1, 1} & \theta_{N-1, 2} & \ldots & \theta_{N-1, d/2}
\end{bmatrix}
$$

where $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$.

Instead of creating and filling up an empty matrix of dimension $N \times d/2$ manually, we observe that the part that is dependent on $i$ (numerator) can be separated by the part dependent on $k$ (denominator).

This allows us to do broadcasting from multiplying two vectors, one containing all $i$ values, while the other containing all of $\frac{1}{\Theta^{(2k-2)/d}}$.

On a small note, also note that the denominator, $\frac{1}{\Theta^{(2k-2)/d}}$, where $k$ ranges from $1$ to $d/2$, can be simplified to $\Theta^{-2k/d}$ where $k$ ranges from $0$ to $d/2 - 1$.

This yields us the final implementation,

```python
multiplier = torch.arange(max_seq_len).float()
angle = theta ** (-2*torch.arange(d_k//2).float()/d_k)

# broadast: (max_seq_len x 1) x (1 x d_k_half) = (max_seq_len x d_k_half)
final_angle = einx.multiply('max_seq_len, d_k_half -> max_seq_len d_k_half', multiplier, angle).to(device=device)
```

## Computing sin and cos matrix

Given $\theta$ matrix from the previous section, $\text{sin\_buf}$ and $\text{cos\_buf}$ can be computed by simply taking the $\sin$ and $\cos$ of $\theta$ matrix, respectively.

## Implementing rotation with precomputed sin and cos matrices

We first investigate the behaviour of reshaping in Numpy/PyTorch. 
Suppose we have an input $x$ of shape $(\ldots, \text{seq\_len}, \text{d\_k}) $, where $\text{seq\_len}=1$ and $d_k=4$:
```python
>>> x = torch.tensor([[[1,2,3,4]]]) # x.shape=(1,1,4)
>>> b = torch.reshape(x, (1,1,2,2))
>>> b[..., 0]
tensor([[[1, 3]]])
>>> b[..., 1]
tensor([[[2, 4]]])
```

We notice that if we were to reshape $x$ by splitting the last axis into two halves $(\ldots,\, \text{seq\_len},\, \text{d\_k/2},\, 2)$, and index based on the last axis, we obtain the interleaving properties required, where the first half contains all even  indices of $x$ across the $d_k$ dimension, while the second half contains the odd counterpart. 

Therefore, this allows us to proceed with the method described above, by first splitting the last axis of $x$ by two halves, performing our operations, then merging the two halves back into one.

This yields us the following implementation:
```python
x = einx.rearrange("b... seq_len (d_k_half a) -> b... seq_len d_k_half a", x, a=2)
        
x_even = x[..., 0]
x_odd = x[..., 1]

x_even_new = (cos_buf * x_even) - (sin_buf * x_odd)
x_odd_new = (cos_buf * x_odd) + (sin_buf * x_even)

x[..., 0] = x_even_new
x[..., 1] = x_odd_new

x = einx.rearrange("b... seq_len d_k_half a -> b... seq_len (d_k_half a)", x, a=2)
```

Some additional notes:

1. Importantly, we need to assign temporary variables during the intermediate steps, rather than directly writing the results back to $x$ (e.g., $\text{x\_even\_new}$ and $\text{x\_odd\_new}$).
This ensures that we do not accidentally perform our remaining operations based on the new values.

2. In the code implementation, we also need to take care of cases where the $\text{seq\_len}$ dimension of the input $x$ does not equate to $\text{max\_seq\_len}$, which will result in an error when performing element-wise multiplication with our cos and sin buffers, both of which are $\text{max\_seq\_len}\times d/2$ in dimension.
Hence, we need to first slice our cos and sin buffer based on the token position of input $x$ along the $\text{seq\_len}$ axis.