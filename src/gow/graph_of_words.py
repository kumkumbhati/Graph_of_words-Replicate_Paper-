"""Graph-of-words: the document representation from Rousseau & Vazirgiannis (2013).

Idea: every distinct word is a dot (vertex). We slide a window over the text and
draw an arrow (edge) from each word to the words that follow it inside the window.
Arrows are unweighted: seeing the same pair twice does NOT add a second arrow.

A word's weight is its INDEGREE: the number of different words that point to it,
that is, the number of different contexts it appears in.
"""
from collections import defaultdict

PAPER_WINDOW = 4   # the window size used in the paper


def build_edges(tokens, window=PAPER_WINDOW):
    """Return the set of arrows (word_before, word_after) in the text.

    A window of 4 means: a word is linked to the 3 words that follow it.
    Using a set means each arrow exists only once (unweighted graph).
    A word never points to itself.
    """
    edges = set()
    for i, word in enumerate(tokens):
        for following in tokens[i + 1 : i + window]:
            if following != word:
                edges.add((word, following))
    return edges


def graph_term_weights(tokens, window=PAPER_WINDOW, weight="indegree"):
    """Return {word: weight} for one document.

    weight = "indegree"  : arrows coming IN  (the paper's choice)
             "outdegree" : arrows going OUT
             "degree"    : distinct neighbours, ignoring direction (undirected graph)
    """
    incoming = defaultdict(set)
    outgoing = defaultdict(set)
    for before, after in build_edges(tokens, window):
        outgoing[before].add(after)
        incoming[after].add(before)

    weights = {}
    for word in set(tokens):
        if weight == "indegree":
            weights[word] = len(incoming[word])
        elif weight == "outdegree":
            weights[word] = len(outgoing[word])
        elif weight == "degree":
            weights[word] = len(incoming[word] | outgoing[word])
        else:
            raise ValueError(f"Unknown weight: {weight}")
    return weights
