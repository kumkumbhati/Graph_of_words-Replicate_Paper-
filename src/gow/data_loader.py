"""Load the SciFact dataset in the shape every experiment needs.

If a local copy exists in the data folder (saved by notebook 05b), it is used:
faster, and works without the internet. Otherwise the data is downloaded.
"""
import pandas as pd

from .paths import DATA_DIR
from .preprocessing import normalize_text

DATASET_NAME = "allenai/scifact"


def _read(config, split):
    """Read one part of SciFact: from the data folder if saved there, else from the internet."""
    if config == "corpus":
        local_file = DATA_DIR / "scifact_corpus.jsonl"
    else:
        local_file = DATA_DIR / f"scifact_claims_{split}.jsonl"

    if local_file.exists():
        return pd.read_json(local_file, lines=True, dtype=False)

    from datasets import load_dataset
    return load_dataset(DATASET_NAME, config)[split].to_pandas()


def load_corpus():
    """Return a DataFrame with one row per paper: doc_id, title, full_text."""
    df = _read("corpus", "train")
    df["doc_id"] = df["doc_id"].astype(str).str.strip()
    title = df["title"].apply(normalize_text)
    abstract = df["abstract"].apply(lambda sentences: normalize_text(" ".join(sentences)))
    df["full_text"] = title + ". " + abstract
    return df[["doc_id", "title", "full_text"]]


def load_claims(split="validation"):
    """Return (queries, qrels) for one split.

    queries: {claim_id: claim text}
    qrels:   {claim_id: set of doc_ids that are correct answers}

    Claims that have no evidence document are skipped, because
    there is no correct answer to look for.
    """
    df = _read("claims", split)
    df["id"] = df["id"].astype(str)
    df["evidence_doc_id"] = df["evidence_doc_id"].astype(str).str.strip()
    df = df[~df["evidence_doc_id"].isin(["", "nan", "None"])]

    queries = {}
    qrels = {}
    for claim_id, claim, doc_id in zip(df["id"], df["claim"], df["evidence_doc_id"]):
        queries[claim_id] = normalize_text(claim)
        qrels.setdefault(claim_id, set()).add(doc_id)
    return queries, qrels
