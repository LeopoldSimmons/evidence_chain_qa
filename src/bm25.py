# -*- coding: utf-8 -*-
"""一个无第三方依赖的BM25实现。"""
import math
from collections import Counter
from typing import List, Sequence


class BM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = [list(doc) for doc in corpus_tokens]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in self.corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.term_freqs = [Counter(doc) for doc in self.corpus_tokens]
        self.df = Counter()
        for doc in self.corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        self.n_docs = len(self.corpus_tokens)

    def idf(self, term: str) -> float:
        # Robertson-Sparck Jones idf with smoothing.
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], doc_idx: int) -> float:
        score = 0.0
        tf = self.term_freqs[doc_idx]
        dl = self.doc_len[doc_idx]
        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query_tokens: Sequence[str], top_k: int = 5) -> List[tuple]:
        scores = [(i, self.score(query_tokens, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
