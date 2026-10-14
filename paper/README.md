# Reference paper

François Rousseau and Michalis Vazirgiannis. 2013.
**Graph-of-word and TW-IDF: New Approach to Ad Hoc IR.**
In *Proceedings of the 22nd ACM International Conference on Information and Knowledge Management (CIKM '13)*, San Francisco, USA.
DOI: [10.1145/2505515.2505671](https://doi.org/10.1145/2505515.2505671)

The PDF is not included in this repository for copyright reasons; it is available from the DOI link above.

## Settings taken from the paper

| Setting | Value |
|---|---|
| Graph | directed, unweighted, sliding window of 4 |
| Term weight (TW) | indegree of the term's vertex |
| TW-IDF | `tw / (1 - b + b * |d| / avdl) * log((N + 1) / df)`, b = 0.003 |
| TF-IDF (pivoted) | `(1 + ln(1 + ln(tf))) / (1 - b + b * |d| / avdl) * log((N + 1) / df)`, b = 0.20 |
| BM25 | `((k1 + 1) * tf) / (K + tf) * log((N + 1) / df)`, K = k1 * (1 - b + b * |d| / avdl), k1 = 1.2, b = 0.75 |
| Preprocessing | stopword removal and Porter stemming |
| Evaluation | MAP and P@10, paired Student's t-test |
