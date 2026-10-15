"""Scoring functions: the same 'spoon' used to taste every retrieval method.

Words used here
---------------
ranked   : list of doc_ids a method returned, best guess first
relevant : set of doc_ids that are truly correct for that claim
run      : {claim_id: ranked list}  -> all answers from one method
qrels    : {claim_id: relevant set} -> the answer key
"""
import math

import pandas as pd


def precision_at_k(ranked, relevant, k=10):
    """Of the top k results, what fraction is correct?"""
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(ranked, relevant, k=10):
    """Of all correct papers, what fraction appears in the top k?"""
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked, relevant, k=10):
    """1 / position of the first correct paper (0 if none in the top k)."""
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def average_precision(ranked, relevant):
    """Rewards putting ALL correct papers near the top (the paper's main metric)."""
    hits = 0
    total = 0.0
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


def ndcg_at_k(ranked, relevant, k=10):
    """Like precision, but a correct paper counts more the higher it is ranked."""
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    best_possible = sum(1.0 / math.log2(p + 1) for p in range(1, min(len(relevant), k) + 1))
    return dcg / best_possible


def evaluate_per_query(run, qrels, k=10):
    """Return a table with one row per claim and one column per metric."""
    rows = []
    for claim_id, relevant in qrels.items():
        ranked = run.get(claim_id, [])
        rows.append({
            "claim_id": claim_id,
            "AP": average_precision(ranked, relevant),
            f"P@{k}": precision_at_k(ranked, relevant, k),
            f"Recall@{k}": recall_at_k(ranked, relevant, k),
            f"MRR@{k}": reciprocal_rank(ranked, relevant, k),
            f"nDCG@{k}": ndcg_at_k(ranked, relevant, k),
        })
    return pd.DataFrame(rows).set_index("claim_id")


def evaluate(run, qrels, k=10):
    """Return the average of every metric over all claims (AP averaged = MAP)."""
    scores = evaluate_per_query(run, qrels, k).mean()
    return scores.rename({"AP": "MAP"}).round(4).to_dict()


def paired_t_test(per_query_a, per_query_b, metric="AP"):
    """Is method A really different from method B, or could it be luck?

    Compares the two methods claim by claim (a 'paired' test, as in the paper).
    Returns the average difference (A minus B) and the p-value.
    A p-value below 0.05 means the difference is unlikely to be luck.
    """
    from scipy.stats import ttest_rel

    a = per_query_a[metric]
    b = per_query_b[metric].loc[a.index]
    result = ttest_rel(a, b)
    return {"mean_difference": round(float((a - b).mean()), 4),
            "p_value": float(result.pvalue)}
