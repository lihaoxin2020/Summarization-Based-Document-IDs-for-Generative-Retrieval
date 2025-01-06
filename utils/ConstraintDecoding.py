from typing import List, Dict
from transformers import StoppingCriteria

import torch


class KeywordTrie:
    def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
        r"""
        A helper class that builds a trie with the words represented in `nested_token_ids`.
        """
        self.max_height = max([len(one) for one in nested_token_ids])

        root = {}
        for token_ids in nested_token_ids:
            level = root
            for tidx, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}

                level = level[token_id]

        self.trie = root

    def next_tokens(self, current_seq) -> Dict:
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = self.trie

        for current_token in current_seq:
            current_token = int(current_token)
            if current_token not in start:
                return {}
            start = start[current_token]

        # next_tokens = list(start.keys())

        return start

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])
    
    def detect_id(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) > 1:
            return False
        if len(next_nodes) == 0:
            return True
        else:
            return self.detect_id(next_nodes[0])


class PrefixAllowedKeywords:
    def __init__(self, keyword_start_ids: List[int], keyword_ids: List[List[int]], eos_token: int):
        self.keyword_start_ids = keyword_start_ids
        self.keyword_ids = keyword_ids
        self.eos_token = eos_token

        self.trie = KeywordTrie(keyword_ids)

    def find_keyword_start(self, sent: List):
        for idx in range(len(sent)):
            if list(sent[idx:idx + len(self.keyword_start_ids)]) == self.keyword_start_ids:
                return idx + len(self.keyword_start_ids)
        assert False, "Keyword start not found"

    def __call__(self, batch_id, sent) -> List:
        keyword_prefix = sent[self.find_keyword_start(sent):]
        next_tokens = self.trie.next_tokens(keyword_prefix)
        # leaves = self.trie.detect_id(next_tokens)
        # or (len(keyword_prefix) >= 10 and self.trie.detect_id(next_tokens))
        if len(next_tokens) == 0:
            return [self.eos_token]

        return list(next_tokens.keys())


class RetrievalStoppingCriteria(StoppingCriteria):
    def __init__(self, id_trie: KeywordTrie, keyword_start_ids, stops):
        super().__init__()
        self.stops = stops
        self.trie = id_trie
        self.keyword_start_ids = keyword_start_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        starting = None
        for idx in range(len(input_ids[0])):
            if list(input_ids[0][idx:idx + len(self.keyword_start_ids)]) == self.keyword_start_ids:
                starting = idx + len(self.keyword_start_ids)
        # starting = ex.index(self.keyword_start_ids) + len(self.keyword_start_ids)
        assert starting is not None, "Keyword start not found"
        for ex in input_ids:
            next_tokens = self.trie.next_tokens(ex[starting:])
            if not self.trie.detect_id(next_tokens):
                return False
        return True

            # if not self.trie.detect_id(next_token)
        
        # return input_ids[-len(self.stops):] == self.stops
