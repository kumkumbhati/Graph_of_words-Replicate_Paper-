from gow_rag.graph_of_words import build_edges, graph_term_weights


def test_window_links_each_word_to_the_next_three():
    edges = build_edges(["a", "b", "c", "d", "e"], window=4)
    assert ("a", "d") in edges          # inside the window
    assert ("a", "e") not in edges      # outside the window
    assert ("b", "a") not in edges      # edges follow word order


def test_edges_are_unweighted_and_have_no_self_loops():
    edges = build_edges(["x", "y", "x", "y"], window=4)
    assert edges == {("x", "y"), ("y", "x")}


def test_indegree_counts_distinct_contexts_not_repetitions():
    tokens = "protein binds receptor and protein activates receptor while protein degrades".split()
    weights = graph_term_weights(tokens)
    assert weights["protein"] == 5      # appears 3 times, but in 5 distinct contexts
    assert weights["binds"] == 1


def test_first_word_can_have_zero_indegree():
    assert graph_term_weights(["alpha", "beta"])["alpha"] == 0
