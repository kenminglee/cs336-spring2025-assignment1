from typing import Self
from collections.abc import Iterable, Iterator

from cs336_basics.bpe.serialization import read_merges_from_file, read_vocab_from_file
from cs336_basics.bpe.pretokenization import pretokenize_str
from cs336_basics.bpe.common import PreToken, BytePair

class Tokenizer:
    '''
    Important note: to speed up repeated operations, we perform caching on the existing vocab/merges.
    So if we want to change the vocab/merges, a new Tokenizer instance MUST be created.
    '''
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self._vocab = vocab
        self._bytes_encoder: dict[bytes, int] = {v:k for k,v in vocab.items()}
        self._pretoken_encoder_cache: dict[str, list[int]] = {}
        
        # If special token is already in vocab, directly assign it in pretoken_encoder_cache so that it doesn't get touched during the merging process
        # If special token is not in vocab, expand vocab to include this special token by assigning a new id 
        if special_tokens:
            # reverse sorting to ensure longer/overlapping patterns are captured first (e.g., "<|endoftext|><|endoftext|>" gets priority over "<|endoftext|>")
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            unassigned_id = max(self._vocab.keys())+1
            for special_tok in special_tokens:
                special_tok_bytes = special_tok.encode("utf-8")
                if special_tok_bytes not in self._bytes_encoder:
                    self._bytes_encoder[special_tok_bytes] = unassigned_id
                    self._vocab[unassigned_id] = special_tok_bytes
                    unassigned_id += 1
                self._pretoken_encoder_cache[special_tok] = [self._bytes_encoder[special_tok_bytes]]
            
        self._merges = merges
        self._special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    )->Self:
        vocab = read_vocab_from_file(vocab_filepath)
        merges = read_merges_from_file(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(
        self,
        text: str
    ) -> list[int]:
        pretoken_str_seq: list[str] = pretokenize_str(text, self._special_tokens, consume_special_token=False)
        pretokens: dict[str, PreToken] = {}
        bps: dict[tuple[bytes, bytes], BytePair] = {}
        for pretoken in set(pretoken_str_seq):
            # if we have cached pretoken, or it is a special_token, skip it
            if pretoken in self._pretoken_encoder_cache:
                continue
            pretokens[pretoken] = PreToken(len(pretokens), tuple(bytes([b]) for b in pretoken.encode("utf-8")), 0)
            for bp in pretokens[pretoken].bp_count.keys():
                if bp not in bps:
                    bps[bp] = BytePair(bp)
                bps[bp].add_parent(pretokens[pretoken])
        
        # Go through all merges in order, and re-apply them on all pretokens
        for bp in self._merges:
            if bp not in bps: 
                continue
            pretokens_to_update:set[PreToken] = bps[bp].parents
            for pretoken in pretokens_to_update:
                old_bp_count = pretoken.merge_and_update(bp)
                new_bp_count = pretoken.bp_count

                for bp_to_remove in (old_bp_count.keys()-new_bp_count.keys()):
                    if bp_to_remove==bp: 
                        continue
                    bps[bp_to_remove].remove_parent(pretoken, old_bp_count)
                
                for bp_to_add in (new_bp_count.keys()-old_bp_count.keys()):
                    if bp_to_add not in bps:
                        bps[bp_to_add] = BytePair(bp_to_add)
                    bps[bp_to_add].add_parent(pretoken)
        
        # Now that merging is complete, go through each merged byte for every pretoken, and convert them into integers
        pretoken_int_seq = []
        for pretoken_str in pretoken_str_seq:
            if pretoken_str not in self._pretoken_encoder_cache:
                self._pretoken_encoder_cache[pretoken_str] = [self._bytes_encoder[b] for b in pretokens[pretoken_str].bytestring]
            pretoken_int_seq.extend(self._pretoken_encoder_cache[pretoken_str])
        return pretoken_int_seq


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, ids: list[int]) -> str:
        # Note: b'\xef\xbf\xbd' is the replacement character U+FFFD (aka '�'), obtained by doing `"\uFFFD".encode("utf-8")`
        bytes_list = [self._vocab.get(id, b'\xef\xbf\xbd') for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors='replace')


if __name__=="__main__":
    print("Running bpe_encoding example:")
    test_str = "the cat ate"
    test_vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    test_merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    tokenizer = Tokenizer(test_vocab, test_merges)
    encoded = tokenizer.encode(test_str)
    assert encoded==[9,7,1,5,10,3], "Incorrect encoding!"
    assert tokenizer.decode([9,7,1,5,10,3])==test_str, "Incorrect decoding!"
    print("bpe_encoding example passed!")

    from cs336_basics import ROOT_DIR
    import os
    VOCAB_PATH = os.path.join(ROOT_DIR,"../tests/fixtures/gpt2_vocab.json")
    MERGES_PATH = os.path.join(ROOT_DIR,"../tests/fixtures/gpt2_merges.txt")

    print("Running test_roundtrip_unicode_string_with_special_tokens")
    
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    res = tokenizer.decode(encoded_ids)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count("<|endoftext|>") == 3
    assert res==test_string
    print("test_roundtrip_unicode_string_with_special_tokens should be good")


    print("Running test_overlapping_special_tokens")
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # Test roundtrip
    assert tokenizer.decode(ids) == test_string
    print("test_overlapping_special_tokens should be good")
