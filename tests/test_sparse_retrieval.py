import math

import pytest

from gow_rag.sparse_retrieval import SparseRetriever, count_words

DOCS = {"d1": "cat cat dog".split(), "d2": "dog bird".split(), "d3": "fish fish cat bird mouse".split()}


def build(model):
    return SparseRetriever(list(DOCS), [count_words(t) for t in DOCS.values()],
                           [len(t) for t in DOCS.values()], model=model)


def test_bm25_weight_matches_equation_6():
    engine = build("bm25")
    avdl = (3 + 2 + 5) / 3
    K = 1.2 * (1 - 0.75 + 0.75 * 3 / avdl)
    expected = (2.2 * 2) / (K + 2) * math.log(4 / 2)        # 'cat' in d1: tf=2, df=2
    assert engine.matrix[0, engine.vocab["cat"]] == pytest.approx(expected)


@pytest.mark.parametrize("model", ["tfidf", "bm25", "twidf", "twidf_log"])
def test_search_ranks_matching_document_first(model):
    assert build(model).search(["cat", "dog"])[0] == "d1"


def test_unknown_words_return_nothing():
    assert build("bm25").search(["zebra"]) == []
