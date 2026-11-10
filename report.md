# Text Generator Lab — one-page report

**Author:** Ayoub El Houani

**Problem and solution.** A small model can generate fluent-looking text without understanding it. We compare a plain POS-HMM, an HMM whose states encode sentence roles, and a trigram model. All are trained on 960 synthetic sentences and tested on 240 held-out sentences split by document. The Streamlit bench samples batches, displays hidden states or estimated tags, reports four metrics, exports CSV and tags user input with log-space Viterbi. The results describe this tiny grammar, not real English quality.

**Generation at T = 1.0.** Each model generates 500 sentences with seed 42. Valid means the coarse tag pattern occurs in training; plausible means the first noun–verb pair occurs in training; novel means the exact sentence is absent from training; distinct-2 is the fraction of unique word bigrams.

| Model | Valid pattern | Plausible pair | Novel | Distinct-2 |
|---|---:|---:|---:|---:|
| Plain HMM | 16.2% | 14.8% | 99.8% | 0.198 |
| Role-split HMM | 100.0% | 60.6% | 99.4% | 0.148 |
| Trigram | 57.0% | 76.0% | 91.2% | 0.105 |

The plain first-order HMM reuses the same POS state in different sentence positions, so it can take transitions that combine incompatible positions. Role-split states remember subject, object or prepositional roles, preventing those transitions in this corpus. Its 100% is for this sample and corpus-defined pattern metric, not a general guarantee of grammatical English. Trigram tags are estimated by the role-split HMM.

**Temperature.** For a writing toy, choose T = 1.5 for more variation; for autocomplete, choose T = 0.5 to favour frequent continuations and reduce surprises. These are heuristic choices, not a measured autocomplete benchmark.

**Examples.** Role-split HMM, seed 42, T = 1: “the children eat the bread at the park .” is a plausible novel example. “the farmers eat the river by the school .” has a valid tag pattern but fails in meaning: tag transitions and independent word emissions encode little semantic compatibility. Its noun–verb pair is seen, illustrating the metric's limits.

**Real data.** The same plain HMM and log-space Viterbi score 100% token accuracy on the 240 held-out synthetic sentences. On Universal Dependencies English EWT (12,544 train / 2,077 test, CC BY-SA 4.0), the evaluation run produced:

| Model | Accuracy | Known words | Unknown words | Whole sentence |
|---|---:|---:|---:|---:|
| Most-frequent tag | 86.2% | 90.8% | 28.6% | 31.9% |
| HMM + Viterbi | 88.0% | 92.9% | 28.2% | 40.4% |

Unknown words are 7.5% of test tokens. Real web text has 17 tags, many more patterns, ambiguity and unseen words; smoothed emissions without lexical features struggle with unknowns. The synthetic set has six tags and a tightly controlled grammar. State splitting improves positional structure but does not teach meaning or guarantee good real-world tagging. Results can vary with treebank revision.
