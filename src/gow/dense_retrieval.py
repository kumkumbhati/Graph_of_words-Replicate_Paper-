"""Meaning-based retrieval with a sentence-embedding model.

Every text becomes a list of numbers (a vector) that captures its meaning.
Texts with similar meaning get similar vectors, even if they share no words.
Search = find the papers whose vectors point in the most similar direction.
"""
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self, doc_ids, doc_texts, model_name=DEFAULT_MODEL, cache_file=None, batch_size=64):
        """
        doc_ids    : list of paper ids
        doc_texts  : list of paper texts (same order)
        cache_file : optional path to save the paper vectors, so they are
                     computed only once (computing them is the slow part)
        """
        from sentence_transformers import SentenceTransformer

        self.doc_ids = list(doc_ids)
        self.model = SentenceTransformer(model_name)

        if cache_file is not None and cache_file.exists():
            self.doc_vectors = np.load(cache_file)
        else:
            self.doc_vectors = self.model.encode(
                list(doc_texts), batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=True,
            )
            if cache_file is not None:
                np.save(cache_file, self.doc_vectors)

        if len(self.doc_vectors) != len(self.doc_ids):
            raise ValueError("Cached vectors do not match the corpus. Delete the cache file and run again.")

    def search_all(self, queries, top_k=1000):
        """Search every claim. Returns a 'run': {claim_id: ranked list of paper ids}."""
        claim_ids = list(queries)
        claim_vectors = self.model.encode(
            [queries[c] for c in claim_ids], normalize_embeddings=True,
        )
        similarity = claim_vectors @ self.doc_vectors.T   # cosine similarity (vectors are normalised)
        top_k = min(top_k, similarity.shape[1])
        best = np.argsort(-similarity, axis=1, kind="stable")[:, :top_k]
        return {c: [self.doc_ids[i] for i in row] for c, row in zip(claim_ids, best)}
