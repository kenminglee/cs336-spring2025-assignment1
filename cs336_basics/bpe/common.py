from collections import Counter
from dataclasses import dataclass, field

@dataclass
class PreToken:
    # unique id so that class is hashable for quick retrieval
    id: int  
    # pretoken represented as a sequence of bytes according to current vocab
    bytestring: tuple[bytes, ...]
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