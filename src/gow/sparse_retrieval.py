"""Word-matching retrieval: TF-IDF, BM25 and (later) TW-IDF.

All three follow the same recipe from the paper (equations 5, 6 and 7):

    score of a paper for a claim = sum, over the claim's words, of
        (how important the word is INSIDE this paper)      <- "term weight"
      x (how rare the word is across ALL papers)           <- "IDF"

They differ only in how the term weight is calculated.
"""
import math
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

# default settings, taken from the paper (section 5)
DEFAULT_B = {"tfidf": 0.20, "bm25": 0.75, "twidf": 0.003, "twidf_log": 0.003}
K1 = 1.2


def count_words(tokens):
    """Term frequency: how many times each word appears. {'word': count}"""
    return dict(Counter(tokens))


def _term_weight(raw, length_norm, model):
    """Turn a raw weight (a count, or a graph weight) into the final term weight."""
    if model == "tfidf":   # equation 5: dampen the count, then length-normalise
        return (1.0 + math.log(1.0 + math.log(raw))) / length_norm
    if model == "bm25":    # equation 6: count saturates, K holds the length normalisation
        return ((K1 + 1.0) * raw) / (K1 * length_norm + raw)
    if model == "twidf":   # equation 7: graph weight, length-normalised
        return raw / length_norm
    if model == "twidf_log":   # our variant: TW-IDF with the same log "brake" as TF-IDF
        return (1.0 + math.log(1.0 + math.log(raw))) / length_norm
    raise ValueError(f"Unknown model: {model}")


class SparseRetriever:
    """Index a collection once, then search it many times."""

    def __init__(self, doc_ids, doc_weights, doc_lengths, model="bm25", b=None):
        """
        doc_ids     : list of paper ids
        doc_weights : list of {word: raw weight}, one dict per paper
                      (word counts for tfidf/bm25, graph weights for twidf/twidf_log)
        doc_lengths : list with the number of words in each paper
        model       : "tfidf", "bm25", "twidf" or "twidf_log"
        b           : length-normalisation strength (None = the paper's default)
        """
        self.doc_ids = list(doc_ids)
        self.model = model
        b = DEFAULT_B[model] if b is None else b

        n_docs = len(self.doc_ids)
        average_length = sum(doc_lengths) / n_docs

        # document frequency: in how many papers does each word appear?
        doc_freq = Counter()
        for weights in doc_weights:
            doc_freq.update(weights.keys())

        # give every word a column number, and compute its IDF (rarity)
        self.vocab = {word: column for column, word in enumerate(doc_freq)}
        idf = {word: math.log((n_docs + 1) / df) for word, df in doc_freq.items()}

        # fill a big table: one row per paper, one column per word
        rows, cols, values = [], [], []
        for row, (weights, length) in enumerate(zip(doc_weights, doc_lengths)):
            length_norm = 1.0 - b + b * (length / average_length)
            for word, raw in weights.items():
                if raw <= 0:
                    continue
                rows.append(row)
                cols.append(self.vocab[word])
                values.append(_term_weight(raw, length_norm, model) * idf[word])

        # "csc" format makes picking out word columns fast
        self.matrix = csr_matrix(
            (values, (rows, cols)), shape=(n_docs, len(self.vocab))
        ).tocsc()

    def search(self, query_tokens, top_k=1000):
        """Return the ids of the best-matching papers, best first."""
        columns = [self.vocab[w] for w in set(query_tokens) if w in self.vocab]
        if not columns:
            return []
        scores = np.asarray(self.matrix[:, columns].sum(axis=1)).ravel()
        top_k = min(top_k, len(scores))
        # sort by score (high to low); ties are broken by original order
        best = np.argsort(-scores, kind="stable")[:top_k]
        return [self.doc_ids[i] for i in best if scores[i] > 0]

    def search_all(self, queries, tokenize, top_k=1000):
        """Search every claim. Returns a 'run': {claim_id: ranked list of paper ids}."""
        return {cid: self.search(tokenize(text), top_k) for cid, text in queries.items()}
