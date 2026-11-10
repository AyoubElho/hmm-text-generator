# Text Generator Lab

Compare a plain part-of-speech HMM, a role-split HMM and a trigram generator trained on a small synthetic corpus. Generate and download batches, examine their tag patterns and corpus-relative metrics, and tag sentences with Viterbi decoding.

The corpus is synthetic and the grammar is tiny. The metrics are defined relative to this corpus and do **not** measure the quality of real English.

## Run locally

From this directory, with Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose a model, temperature, seed and batch size in the sidebar, then select **Generate**. Generated HMM tags are sampled hidden states; the trigram has no hidden states, so its tags are estimated by the role-split HMM. The tagger warns about unknown words and uses smoothed probabilities to estimate their tags.

**Metrics:** a valid pattern has a coarse tag sequence seen in training; a plausible subject–verb has a first noun/first verb pair seen in training; novelty means an exact sentence was not in training; distinct-2 is the fraction of unique adjacent word pairs among all generated pairs. These are simple proxies, not semantic or grammatical judges.

## Evaluation

```bash
python -m src.evaluate       # 500 synthetic samples per model and temperature
python -m src.evaluate_real  # downloads UD English EWT on first run
```

Outputs are in `results/`; the one-page report is `report.pdf`, generated from the editable LaTeX source `report.tex` with `pdflatex report.tex` (run twice to update links). The real-data test uses the [Universal Dependencies English EWT treebank](https://github.com/UniversalDependencies/UD_English-EWT), CC BY-SA 4.0. It downloads the training and test files into the ignored `data/ewt/` directory. Dataset revisions can change results.

## Deploy on Streamlit Community Cloud

1. Put this directory in a GitHub repository. Include `app.py`, `src/`, `requirements.txt`, and **`data/corpus.csv`**; omit `.venv/` and `data/ewt/`.
2. On [Streamlit Community Cloud](https://share.streamlit.io/), create an app from that repository and select `app.py` as the main file (or the relative path if the project is in a subdirectory).
3. Open the resulting public URL in a private browser window, generate a batch with each model, try the tagger with an unknown word, and download a CSV.

`results/` contains reproducible evaluation outputs; the web app itself does not download the real-data corpus.
