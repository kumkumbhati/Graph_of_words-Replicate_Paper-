"""Combine several rankings into one (a 'hybrid' search)."""
from collections import defaultdict


def reciprocal_rank_fusion(runs, k=60, top_k=1000):
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    Each method gives every paper points: 1 / (k + its position).
    Papers ranked high by several methods collect the most points.
    k = 60 is the standard value from the original paper, so nothing is tuned.

    runs : list of runs, each {claim_id: ranked list of paper ids}
    """
    fused = {}
    for claim_id in runs[0]:
        points = defaultdict(float)
        for run in runs:
            for position, doc_id in enumerate(run.get(claim_id, []), start=1):
                points[doc_id] += 1.0 / (k + position)
        ranked = sorted(points, key=lambda d: -points[d])
        fused[claim_id] = ranked[:top_k]
    return fused
