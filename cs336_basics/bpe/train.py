import os
from collections import Counter
from dataclasses import dataclass, field
import time

from tqdm import tqdm

from cs336_basics.bpe.serialization import write_vocab_to_file, write_merges_to_file
from cs336_basics.bpe.pretokenization import pretokenize, pretokenize_str, NON_WHITESPACE_PRE_TOKENIZER
from cs336_basics import ROOT_DIR



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

def train(
    pretokens: Counter[str], # obtained after pretokenization step
    vocab_size:int,
    special_tokens:list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size>(min_vocab_size:=256+len(special_tokens)), f"Vocab size must be at least {min_vocab_size}"

    vocab: dict[int, bytes] = {i:tok.encode("utf-8") for i,tok in enumerate(special_tokens)}
    vocab.update({len(special_tokens)+i : bytes([i]) for i in range(256)})
    merges: list[tuple[bytes, bytes]] = []
    bps:dict[tuple[bytes, bytes], BytePair] = {}
    for uid, (pretoken, count) in enumerate(pretokens.items()):
        pretok = PreToken(uid, tuple(bytes([tok]) for tok in pretoken.encode("utf-8")), count)
        for bp in pretok.bp_count.keys():
            if bp not in bps:
                bps[bp] = BytePair(bp)
            bps[bp].add_parent(pretok)
    
    for _ in tqdm(range(vocab_size-len(vocab)), desc="Vocab size"):
    
        # Find bytepair(s) with maximum count
        max_count:int = 0
        maxes:list[tuple[bytes,bytes]] = []
        for bp in bps.values():
            if bp.count > max_count:
                max_count = bp.count
                maxes = [bp.bp]
            elif bp.count == max_count:
                maxes.append(bp.bp)
            
        
        # Find lexicographically greatest pair among tiebreakers
        bp_to_merge = max(maxes)

        # Update all pretokens that are affected by the merge (i.e., contain the merged bytepair)
        for pretok in bps[bp_to_merge].parents:
            # e.g., Pretok: a,b,c,b,c, merge "a,b"
            # old_bp_count: {(a,b):1, (b,c):2, (c,b):1}
            # new_bp_count: {(ab,c):1, (b,c):1, (c,b):1}
            # step 1: bps_to_remove = {(a, b)}
            # step 2: bps_to_add = {(ab, c)}
            # step 3: unchanged_bps_w_changed_value = {(b,c)}
            old_bp_count = pretok.merge_and_update(bp_to_merge)
            new_bp_count = pretok.bp_count

            # 1) find byte pairs that no longer exist due to the merge
            bps_to_remove = old_bp_count.keys() - new_bp_count.keys()

            for bp in bps_to_remove:
                # don't worry about the byte pair to merge this iteration -- the entire BytePair obj will be deleted anyways!
                if bp==bp_to_merge:
                    continue
                # remove PreToken parent from this bytepair
                bps[bp].remove_parent(pretok, old_bp_count)

            # 2) find new byte pairs created due to the merge
            bps_to_add = new_bp_count.keys() - old_bp_count.keys()
            
            for bp in bps_to_add:
                if bp not in bps:
                    bps[bp] = BytePair(bp)
                bps[bp].add_parent(pretok)

            # 3) find existing byte pairs whose count has changed
            #    i) find unchanged bps whose count DID NOT change
            unchanged_bps_w_unchanged_values:set[tuple[bytes, bytes]] = {i[0] for i in (new_bp_count.items() & old_bp_count.items())} 
            #    ii) find all unchanged bps 
            unchanged_bps:set[tuple[bytes,bytes]] = new_bp_count.keys() & old_bp_count.keys()
            #    iii) loop through all unchanged_bps with CHANGED values, and update their counts
            for bp in (unchanged_bps - unchanged_bps_w_unchanged_values):
                bps[bp].update_parent_count(pretok, old_bp_count)

        # no longer needed since the bytepair (i.e., 2 tokens) is now considered a single token              
        del bps[bp_to_merge]

        vocab[len(vocab)] = bp_to_merge[0]+bp_to_merge[1]
        merges.append(bp_to_merge)
    return vocab, merges

def bpe_example(
    text: str,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"]
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    pretokens: Counter[str] = pretokenize_str(text, special_tokens, pretokenize_regex=NON_WHITESPACE_PRE_TOKENIZER, return_counter=True)
    return train(pretokens, vocab_size, special_tokens)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_processes:int = 4
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    pretokens: Counter[str] = pretokenize(input_path, num_processes, special_tokens, return_counter=True)
    return train(pretokens, vocab_size, special_tokens)

def train_bpe_tinystories(
    num_processes:int = 32,
    output_dir:str=os.path.join(ROOT_DIR, "../data")
):
    start_time = time.time()
    fp = os.path.join(ROOT_DIR, "../data/TinyStoriesV2-GPT4-train.txt")
    assert os.path.isfile(fp), f"{fp} does not exist!"
    vocab, merges = train_bpe(fp, 10000, special_tokens=["<|endoftext|>"], num_processes=num_processes)
    elapsed_time = time.time()-start_time
    print(f"Finished training on TinyStories dataset, spent {int(elapsed_time // 60)} minutes and {elapsed_time % 60} seconds, now saving to disk...")
    write_vocab_to_file(vocab, os.path.join(output_dir, "TinyStoriesV2-GPT4-train-vocab.json"))
    write_merges_to_file(merges, os.path.join(output_dir, "TinyStoriesV2-GPT4-train-merges.txt"))

def train_bpe_owt(
    num_processes:int = 32,
    output_dir:str=os.path.join(ROOT_DIR, "../data")
):
    start_time = time.time()
    fp = os.path.join(ROOT_DIR, "../data/owt_train.txt")
    assert os.path.isfile(fp), f"{fp} does not exist!"
    vocab, merges = train_bpe(fp, 32000, special_tokens=["<|endoftext|>"], num_processes=num_processes)
    elapsed_time = time.time()-start_time
    print(f"Finished training on owt dataset, spent {int(elapsed_time // 60)} minutes and {elapsed_time % 60} seconds, now saving to disk...")
    write_vocab_to_file(vocab, os.path.join(output_dir, "owt-train-vocab.json"))
    write_merges_to_file(merges, os.path.join(output_dir, "owt-train-merges.txt"))

if __name__=="__main__":
    print("Running bpe_example test:")
    vocab, merge = bpe_example(
        '''
        low low low low low
        lower lower widest widest widest
        newest newest newest newest newest newest
        ''',
        257+6,
        special_tokens = ["<|endoftext|>"]
    )
    assert len(vocab.items()&{257: b'st', 258: b'est', 259: b'ow', 260: b'low', 261: b'west', 262: b'ne'}.items())==6, "Vocab does not match!"
    assert merge==[(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e')]
    print("bpe_example test passed!")

    print('Please run "uv run pytest tests/test_train_bpe.py" to test train_bpe')

    # print("Training BPE on TinyStories")
    # train_bpe_tinystories()

    # print("Training BPE on owt")
    # train_bpe_owt()
    
