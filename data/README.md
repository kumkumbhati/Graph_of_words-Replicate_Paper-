# Data

This project uses **SciFact** (Wadden et al., 2020), loaded from the Hugging Face Hub as [`allenai/scifact`](https://huggingface.co/datasets/allenai/scifact).

| Part | Size | Used as |
|---|---|---|
| Corpus | 5,183 paper abstracts | Documents to search |
| Claims, `validation` split | 188 claims with evidence | Development set (used for the diagnosis in notebook 05) |
| Claims, `train` split | 505 claims with evidence | Held-out confirmation set (no method is trained, so these claims were never used to make a choice) |

Claims without an evidence document are skipped, since there is no correct answer to retrieve.

## Files

- `scientific_stopwords.txt` - project stopword list (NLTK English list minus some function words such as *not*, *than*, *between*). Compared against the full NLTK list in notebook 05.
- `scifact_*.jsonl`, `minilm_corpus_vectors.npy` - local caches created by notebooks 05 and 07. They are not stored in the repository (see `.gitignore`); the code downloads the data automatically when they are missing.
