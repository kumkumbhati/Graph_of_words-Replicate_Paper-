# Graph-of-Words Retrieval on Scientific Claims: A Replication Study

A replication of **Graph-of-word and TW-IDF** (Rousseau & Vazirgiannis, CIKM 2013) on scientific literature, followed by a diagnosis of why it behaves differently there and a comparison with modern dense retrieval.

The original paper showed that weighting a word by the **number of distinct contexts** it appears in (its indegree in a graph-of-words) beats classic term-frequency models such as BM25 on TREC news and web collections. This project asks whether that result carries over to a very different kind of text: **scientific abstracts**, using the [SciFact](https://huggingface.co/datasets/allenai/scifact) claim-to-evidence retrieval task.

## Key findings

1. **TW-IDF as published does not beat BM25 on SciFact.** It is significantly worse (MAP 0.655 vs 0.819 on held-out claims, paired t-test p < 1e-25).
2. **The cause is the absence of a concave damping of the term weight.** Text preparation (stopword list, stemming), length normalisation and claim length were each tested and ruled out as the main cause. Adding the logarithmic damping that TF-IDF uses, `1 + ln(1 + ln(tw))`, which the original paper found unnecessary on TREC, raises TW-IDF to **statistical parity with BM25** on held-out claims (MAP 0.810 vs 0.819, p = 0.12).
3. **In a hybrid with a dense model, graph weighting and term counting are interchangeable partners.** Both hybrids significantly improve on the dense model alone, and they are statistically tied with each other (p = 0.44).

## Research questions

| # | Question | Answer |
|---|---|---|
| RQ1 | Does TW-IDF outperform TF-IDF and BM25 on scientific claim retrieval, as it does on TREC? | No, it is significantly worse |
| RQ2 | If not, why? | Missing concave damping of the indegree; not preprocessing, length normalisation or query length |
| RQ3 | Does graph-based weighting still add value next to a modern dense retriever? | It helps as much as BM25 does, but not more |

## Results

All numbers use the paper's preprocessing (stopword removal and Porter stemming). The **validation** claims were used during the diagnosis; the **train** claims were never used to make any choice, so they act as a held-out confirmation set (no model is trained).

### Replication (notebook 06)

| Claims | Method | MAP | P@10 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| validation (188) | TF-IDF | 0.782 | 0.100 | 0.925 | 0.785 | 0.817 |
| | BM25 | 0.781 | 0.100 | 0.924 | 0.784 | 0.816 |
| | TW-IDF (paper) | 0.604 | 0.088 | 0.834 | 0.607 | 0.658 |
| | TW-IDF-log (ours) | 0.757 | 0.097 | 0.910 | 0.760 | 0.794 |
| **train (505, held out)** | TF-IDF | 0.812 | 0.103 | 0.932 | 0.816 | 0.841 |
| | BM25 | **0.819** | 0.102 | 0.931 | 0.824 | 0.847 |
| | TW-IDF (paper) | 0.655 | 0.095 | 0.872 | 0.659 | 0.707 |
| | TW-IDF-log (ours) | 0.810 | 0.101 | 0.918 | 0.813 | 0.836 |

P@10 is low for every method because most claims have a single correct paper, so the best possible value is about 0.1.

### Significance (paired t-test on per-claim average precision)

| Claims | Comparison | Mean difference in AP | p-value | Significant? |
|---|---|---|---|---|
| validation | TW-IDF vs BM25 | -0.177 | 7.3e-11 | yes |
| validation | TW-IDF-log vs BM25 | -0.024 | 0.017 | yes |
| train | TW-IDF vs BM25 | -0.164 | 1.4e-26 | yes |
| train | TW-IDF-log vs BM25 | -0.009 | 0.12 | **no** |
| train | TW-IDF-log vs TW-IDF | +0.155 | 1.4e-23 | yes |

### Diagnosis (notebook 05, validation claims, MAP)

| Text preparation | TF-IDF | BM25 | TW-IDF |
|---|---|---|---|
| Project stopwords, no stemming | 0.760 | 0.770 | 0.600 |
| Full NLTK stopwords, no stemming | 0.764 | 0.771 | 0.603 |
| Full stopwords + Porter stemming (paper) | 0.782 | 0.781 | 0.604 |

| TW-IDF variant | MAP |
|---|---|
| Paper (b = 0.003) | 0.604 |
| Stronger length normalisation, b = 0.20 | 0.630 |
| Stronger length normalisation, b = 0.75 | 0.644 |
| **Log damping, b = 0.003** | **0.757** |
| Log damping, b = 0.20 | 0.755 |
| BM25, for reference | 0.781 |

Preprocessing barely moves TW-IDF, and length normalisation helps only a little; damping closes most of the gap.

**Claim length** (all 693 claims, MAP): the gap between BM25 and TW-IDF is *largest* for short claims (0.23 for 1-5 words) and shrinks for long ones (0.09 for 12+ words). So the paper's use of very short TREC title queries does not explain the difference.

### Dense and hybrid retrieval (notebook 07)

Dense model: `sentence-transformers/all-MiniLM-L6-v2`. Hybrids use Reciprocal Rank Fusion (k = 60, no tuning).

| Method | MAP (validation) | MAP (train) | nDCG@10 (train) |
|---|---|---|---|
| BM25 | 0.781 | 0.819 | 0.847 |
| TW-IDF-log | 0.757 | 0.810 | 0.836 |
| Dense (MiniLM) | 0.728 | 0.780 | 0.818 |
| Hybrid: Dense + BM25 | 0.799 | 0.829 | 0.860 |
| Hybrid: Dense + TW-IDF-log | **0.800** | **0.834** | **0.863** |

On the held-out claims, both hybrids beat the dense model alone (p < 1e-5), the dense model alone is below BM25 (p = 0.012), and the two hybrids are statistically tied (p = 0.44). Whether the hybrids significantly beat BM25 alone has not yet been tested.

## Method

| Component | Implementation |
|---|---|
| Graph-of-words | Directed, unweighted, sliding window of 4 over the whole document, no self-loops (`src/gow_rag/graph_of_words.py`) |
| Term weight | Indegree of each word's vertex |
| Scoring | TF-IDF, BM25 and TW-IDF exactly as equations 5-7 of the paper, with its default parameters (`src/gow_rag/sparse_retrieval.py`) |
| Our variant | TW-IDF-log: TW-IDF with `1 + ln(1 + ln(tw))` damping; nothing else changed |
| Dense retrieval | MiniLM sentence embeddings, cosine similarity (`src/gow_rag/dense_retrieval.py`) |
| Fusion | Reciprocal Rank Fusion, k = 60 (`src/gow_rag/fusion.py`) |
| Evaluation | MAP (top 1000), P@10, Recall@10, MRR@10, nDCG@10; paired Student's t-test as in the paper (`src/gow_rag/evaluation.py`) |

## Repository structure

```
scientific-literature-GOW-RAG/
├── src/gow_rag/                 shared, tested code used by every notebook
│   ├── paths.py                 project folder locations
│   ├── data_loader.py           SciFact corpus, claims and relevance judgements
│   ├── preprocessing.py         tokenisation, stopwords, stemming
│   ├── graph_of_words.py        graph construction and degree-based weights
│   ├── sparse_retrieval.py      TF-IDF, BM25, TW-IDF, TW-IDF-log
│   ├── dense_retrieval.py       sentence-embedding retrieval
│   ├── fusion.py                reciprocal rank fusion
│   └── evaluation.py            metrics and significance test
├── notebooks/
│   ├── 00_setup_check.ipynb
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_classic_baselines.ipynb
│   ├── 04_twidf_replication.ipynb
│   ├── 05_twidf_diagnosis.ipynb
│   ├── 06_confirmation_and_significance.ipynb
│   └── 07_dense_and_hybrid.ipynb
├── results/tables/              every number reported above, as CSV
├── tests/                       unit tests (pytest)
├── data/                        stopword list and data notes
└── paper/                       reference to the original paper
```

Notebooks 03 and 04 are the first pass with the project stopword list and no stemming; the final numbers come from notebooks 05-07.

## How to reproduce

```bash
git clone https://github.com/kumkumbhati/scientific-literature-GOW-RAG.git
cd scientific-literature-GOW-RAG
pip install -r requirements.txt
pytest                      # run the unit tests
jupyter notebook            # then run the notebooks in order, 00 to 07
```

The dataset downloads automatically the first time. Notebook 05 saves a local copy in `data/`; later notebooks use it. On a laptop CPU, everything except the dense encoding in notebook 07 runs in a few minutes.

## Limitations

- **Collection size.** SciFact has 5,183 abstracts and 693 claims with evidence, far smaller than the TREC collections in the paper (0.5 to 25 million documents).
- **Stopword list.** The paper used Terrier's stopword list; this project uses NLTK's English list as the closest available equivalent.
- **Dense model.** MiniLM is a general-purpose model; a model trained on scientific text might change the hybrid conclusions.
- **Multiple comparisons.** p-values are reported without correction for the number of tests.
- **Diagnosis on validation.** The damping fix was found on the validation claims; it is confirmed on the held-out train claims, but not on a second collection.

## Future work

- **Graph neural networks.** Learn term or document representations directly from the graph-of-words (for example a trained GNN over word graphs), instead of a fixed indegree weight.
- **Retrieval-augmented generation.** Use the best hybrid retriever to supply evidence to a language model that verifies or answers scientific claims, and measure whether better retrieval leads to better answers.
- **More collections.** Test the damping finding on other scientific retrieval sets (for example other BEIR datasets) to see whether it generalises.

## References

- F. Rousseau and M. Vazirgiannis. *Graph-of-word and TW-IDF: New Approach to Ad Hoc IR.* CIKM 2013. [doi:10.1145/2505515.2505671](https://doi.org/10.1145/2505515.2505671)
- D. Wadden et al. *Fact or Fiction: Verifying Scientific Claims.* EMNLP 2020 (SciFact).
- G. V. Cormack, C. L. A. Clarke and S. Buettcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009.

## Author

**Kumkum Bhati** - [github.com/kumkumbhati](https://github.com/kumkumbhati)

Licensed under the MIT License.
