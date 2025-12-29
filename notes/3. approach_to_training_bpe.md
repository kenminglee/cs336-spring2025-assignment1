# Speeding Up BPE Training Process
Two main parts of the pipeline were optimized: the pre-tokenization stage, and the merging stage.

## Pre-Tokenization
To speed up pretokenization, we performed the following optimizations:
1. Split the dataset into smaller chunks, and perform pretokenization on each chunk in parallel, before re-combining the results.

Importantly, instead of naively splitting the dataset into equal chunks, we want to find proper boundaries to split on (i.e., right before a special token), so that we don't risk splitting in between a special token.

2. Have each worker process return a Counter object that keeps track of the number of occurrence of each pretoken in that chunk of text, rather than a list of pretoken strings. 

The Counter object is much more concise, making the reduction (i.e., merging results from all worker processes) much faster.

The disadvantage of this is we lose the order of the original pretokens (necessary for when we are trying to tokenize a string of text). 
Hence, we add an option to return a list.

## Merging
In the naive implementation of BPE training, we perform the following:
1. Find the most common byte pair among all pretokens
2. Perform the merge by:
    - Iterating across all pretokens to find pretokens affected by the merge
    - For each pretoken affected by the merge, do the merge, then update all bytepairs (with the new counts) in the pretoken.

To speed things up, we start with two dataclasses:
```python
@dataclass
class PreToken:
    # unique id so that class is hashable for quick retrieval
    id: int  
    # pretoken represented as a sequence of bytes according to current vocab
    bytestring: tuple[bytes]
    # number of occurrence of this pretoken in the corpus
    num_occurrence: int
    # mapping of bp -> num occurrence of bp in this pretoken
    bp_count: dict[tuple[bytes, bytes], int] = field(init=False)

    def __post_init__(self):
        self._update_bp_count()

    def _update_bp_count(self):
        self.bp_count = Counter(zip(self.bytestring[:-1], self.bytestring[1:]))

    # updates bp_count and bytestring and returns the old bp_count. Note: bp is short for byte-pair
    def merge_and_update(self, bp:tuple[bytes, bytes]) -> dict[tuple[bytes, bytes],int]:
        new_bytestring = []
        i = 0
        while i<len(self.bytestring):
            if self.bytestring[i:i+2]==bp:
                new_bytestring.append(bp[0]+bp[1])
                i += 2
            else:
                new_bytestring.append(self.bytestring[i])
                i += 1
        self.bytestring = tuple(new_bytestring)
        old_bp_count = self.bp_count
        self._update_bp_count()
        return old_bp_count
    
    def __hash__(self) -> int:
        return self.id
    
    def __eq__(self, other: object, /) -> bool:
        return isinstance(other, PreToken) and other.id==self.id
            

@dataclass
class BytePair:
    bp: tuple[bytes, bytes]
    # list of pretokens that contain this bp
    parents: set[PreToken] = field(default_factory=lambda:set(), init=False)
    # count of bp in the entire corpus
    count: int = field(default=0, init=False)

    def add_parent(self, parent:PreToken):
        self.parents.add(parent)
        self.count += (parent.bp_count[self.bp]*parent.num_occurrence)
    
    def remove_parent(self, parent: PreToken, old_bp_count: dict[tuple[bytes, bytes],int]):
        self.parents.remove(parent)
        self.count -= (parent.num_occurrence*old_bp_count[self.bp])

    def update_parent_count(self, parent: PreToken, old_bp_count: dict[tuple[bytes, bytes], int]):
        self.count -= (parent.num_occurrence*old_bp_count[self.bp])
        self.count += (parent.num_occurrence*parent.bp_count[self.bp])
```

Every pretoken contain a list of bytepairs, while a bytepair contains a list of parent pretokens (i.e., pretokens that contains this bytepair).

With this setup, instead of iterating across all pretokens after every merge, we could immediately fetch all pretokens that we need to update, given a bytepair.

Additionally, by doing the following set operations, we can quickly find all bytepairs that needed to be updated after performing a merge on a pretoken.

For example, suppose we have a pretoken "a,b,c,b,c" and we want to merge "a,b":

- Before the merge, our bytepair count for this pretoken is: `old_bp_count={(a,b):1, (b,c):2, (c,b):1}`
- After the merge, our new bytepair count for this pretoken becomes `new_bp_count={(ab,c):1, (b,c):1, (c,b):1}`

This leads to the 3 following cases to take care of: 
- By doing `old_bp_count.keys()-new_bp_count.keys()`, we can obtain bytepairs that no longer exist after this merge: `bps_to_remove = {(a, b)}`
- Vice versa, by doing the opposite, `new_bp_count.keys()-old_bp_count.keys()`, we obtain new bytepairs that exist after this merge: `bps_to_add = {(ab, c)}`
- Last but not least, through set operations, we can also find unchanged bytepairs, whose count has changed due to the merge: `unchanged_bps_w_changed_value = {(b,c)}`