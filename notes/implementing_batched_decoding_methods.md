Main idea: now that we have a model, we want to be able to generate text from it.

Recall the decoding process: 
- tokenize string prompt into a sequence of tokenIDs (whose length is less than context_length)
- pass it into transformer, which outputs a tensor of dimension (1 x seq_len x vocab_size). 
Omitting the 1st dimension (which represents the batch_size), we have a matrix of seq_len rows and vocab_size columns.
    - each row represents the unormalized distribution modelling the next word.
    - in this case, we are only interested in the prediction for the next (unseen) word, so we retrieve only the last row.
- left with a vector of vocab_size, we can normalize it (using softmax), and sample from the softmaxed distribution to get the tokenID of the predicted next word.

# Decoding tricks
Assuming that we are left with a vector of vocab_size probabilities predicting the next word, what is the optimal way to sample from this distribution?

## Temperature scaling

One way to potentially improve our text quality is to control the certainty of our prediction, by introducing a temperature parameter $\tau$ into our softmax function:

$$
\text{softmax}(v, \tau)_i = \frac{\exp (v_i/ \tau)}{\sum_{j=1}^{|\text{vocab\_size}|} \exp (v_j / \tau)}
$$

Studying the behaviour of this function, we note that when $\tau$ is greater than 0,
- Increasing $\tau$ makes our softmax distribution flat, increasing stochasticity (i.e., each word has almost the same probability of being chosen)
- Setting $\tau \rightarrow 0$ exaggerates the differences between $v_i$ , making us increasingly closer to a hard-max (i.e., dirac-delta distribution on the max value)

On the other hand, when $\tau$ is set to a negative value,
- Decreasing $\tau$ (i.e., $\tau \rightarrow -\infty$ in the extreme case) gives us the same result as massively increasing $\tau$ (i.e., $\tau \rightarrow \infty$), where the distribution becomes increasingly flat
- Seting $\tau \rightarrow 0^-$ exaggerates the differences between $v_i$ , but this time making our softmax closer to a hard-min (i.e., dirac-delta distribution on the min value).

Hence ensuring $\tau > 0$  would probably make more sense in our case.

## Top-k Sampling
The idea is that instead of sampling from all possible words from the vocabulary, we should instead only sample from the top-k most probable words (tokens) in the vocabulary.

## Top-p/Nucleus Sampling
The idea is that instead of sampling from all possible words from the vocabulary, we want to sample from a bag (i.e., smallest possible subset) of words whose cumulative probability hits a set threshold (P).

In order to obtain the smallest possible subset of words, priority is given to words whose probability is high.

Consider the following toy example
- Probabilities: mat (40%), chair (30%), sofa (25%), floor (5%),
- Suppose we set our P value to 0.9, then our smallest subset of words whose cumulative probability meets the threshold is: {mat, chair, sofa}
    - mat + chair + sofa = 0.4 + 0.3 + 0.25 = 0.95, which exceeds 0.9.
    - floor is not included because it is not necessary to hit the threshold.

But how to implement this as a batch (dim = batch_size x vocab_size)?
1. The idea is to first sort each row by their probabilities. 
Importantly, we sort in *ascending* order (we will have an off-by-one error if sort in descending order! -- see example below to see why).
2. Calculate the cumulative sum for each row.
For each row, the point from which the cumulative sum is larger than 1-p, these are indices that we want to keep.

For example, suppose we have a probability distribution over next words as such:

```json
{
    "mat": 0.02,
    "door": 0.05,
    "carpet": 0.03,
    "letter": 0.4,
    "cat": 0.3,
    "dog": 0.2
}
```

Then, suppose:
- we have a $p$ value of 0.9, then our nucleus should have {"letter", "cat", "dog"}
    - sorted probs: [0.02, 0.03, 0.05, 0.2, 0.3, 0.4]
    - corresponding sorted list: ["mat", "carpet", "door", "dog", "cat", "letter"]
    - cumulative sum: [0.02, 0.05, 0.1, 0.3, 0.6, 1.0]
    - cum_sum <= 0.1: [T, T, T, F, F, F]. Note that this mask is able to correctly separate between words we want to include(F)/exclude(T).
- we have a $p$ value of 0.93, then our nucleus should have {"letter", "cat", "dog", "door"}
    - sorted probs: [0.02, 0.03, 0.05, 0.2, 0.3, 0.4]
    - corresponding sorted list: ["mat", "carpet", "door", "dog", "cat", "letter"]
    - cumulative sum: [0.02, 0.05, 0.1, 0.3, 0.6, 1.0]
    - cum_sum <= 0.07: [T, T, F, F, F, F]. Note that this mask is able to correctly separate between words we want to include(F)/exclude(T).
- Just to prove why we have to compute cumulative sum on an ascending order, suppose we do it in descending order on a $p$ value of 0.93.
Similar to above, our nucleus should have {"letter", "cat", "dog", "door"}
    - sorted probs: [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
    - corresponding sorted list: ["letter", "cat", "dog", "door", "carpet", "mat"]
    - cumulative sum: [0.4, 0.7, 0.9, 0.95, 0.98, 1.0]
    - cum_sum smaller or equals to 0.93: [T, T, T, F, F, F]. Note that this mask is **unable** to correctly separate between words we want to include(T)/exclude(F).


3. Theoretically, with this mask, we can correctly separate out words that we want to keep in the nucleus, and can therefore set probabilities of words we want to exclude to 0.

4. However, note that this would NOT WORK since this mask only applies to a matrix with *sorted* rows!
Therefore, we have to reapply this mask to our original *unsorted* matrix!<br>
The key is to, notice that our masking operation (setting cum_sum <= 1-p ) doesn't care about the order of each row.
So, we just need to reapply the cumulative sum to the position in the original matrix (using `torch.scatter`), before computing the mask!

This boils down to the following implementation:
```python
# each row of sorted_idx shows index of ascending order probs for that row
sorted_idx = torch.argsort(probs, dim=-1)
# sort each row of probs by ascending order
sorted_probs = torch.take_along_dim(probs, dim=-1, indices=sorted_idx)
# take cumulative sum of each row
sorted_probs_cumsum = torch.cumsum(sorted_probs, dim=-1)
# assign cumulative sum back to the original unsorted probs
probs_cumsum = torch.scatter(probs, dim=-1, index=sorted_idx, src=sorted_probs_cumsum)
# only keep those whose cumsum > 1-p
# set to 0 so that we don't have negative values, so no need to do a full softmax; a simple division will do
probs[probs_cumsum <= (1-p)] = 0
```
