"""Interactive test bench for the synthetic POS-HMM and trigram generators."""
import random
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import HMM, NGramGenerator, coarse, generation_metrics, load_tagged, split_by_document


st.set_page_config(page_title="Text Generator Lab", page_icon="✦", layout="wide")
DATA = Path(__file__).resolve().parent / "data" / "corpus.csv"


@st.cache_resource
def get_models():
    train, _ = split_by_document(load_tagged(str(DATA)))
    return train, HMM(train), HMM(train, split_states=True), NGramGenerator([words for words, _ in train])


def generate_batch(model_name, count, temperature, seed, train, plain, split, trigram):
    rng = random.Random(seed)
    patterns = {tuple(tags) for _, tags in train}
    samples, rows = [], []
    for number in range(1, count + 1):
        if model_name == "Trigram":
            words = trigram.sample_sentence(rng, temperature)
            states = split.viterbi(words)
        else:
            model = plain if model_name == "Plain HMM" else split
            words, states = model.sample_with_states(rng, temperature)
        tags = coarse(states)
        samples.append((words, tags))
        rows.append({"#": number, "Sentence": " ".join(words), "Tags / states": " ".join(states),
                     "Valid pattern": "Yes" if tuple(tags) in patterns else "No"})
    return pd.DataFrame(rows), generation_metrics(samples, train, split)


train, plain, split, trigram = get_models()
st.title("Text Generator Lab")
st.caption("Explore how hidden states, temperature and context shape generated sentences.")
st.info("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined relative to this corpus and do not measure the quality of real English.")

with st.sidebar:
    st.header("Control panel")
    model_name = st.selectbox("Model", ["Plain HMM", "Role-split HMM", "Trigram"])
    temperature = st.slider("Temperature", 0.3, 3.0, 1.0, 0.1,
                            help="Lower values favour frequent choices; higher values increase variation.")
    seed = st.number_input("Random seed", min_value=0, max_value=1000000, value=42, step=1)
    count = st.number_input("Number of sentences", min_value=10, max_value=500, value=100, step=10)
    generate = st.button("Generate", type="primary", use_container_width=True)
    st.caption("The same model, seed and settings produce the same batch.")

if generate:
    frame, metrics = generate_batch(model_name, int(count), temperature, int(seed), train, plain, split, trigram)
    st.session_state["batch"] = (frame, metrics, model_name, temperature, int(seed))

st.header("Generated batch")
if "batch" not in st.session_state:
    st.write("Choose a model and settings, then select **Generate**.")
else:
    frame, metrics, batch_model, batch_temperature, batch_seed = st.session_state["batch"]
    st.caption(f"{batch_model} · {len(frame)} sentences · temperature {batch_temperature:.1f} · seed {batch_seed}")
    if batch_model == "Trigram":
        st.caption("A trigram has no hidden states: the displayed tags are estimated with the role-split HMM's Viterbi tagger.")
    else:
        st.caption("These are the HMM's sampled hidden states. Role-split suffixes indicate subject (S), object (O) or prepositional phrase (P).")
    st.subheader("Batch metrics")
    columns = st.columns(4)
    columns[0].metric("Valid tag pattern", f"{metrics['valid_tag_pattern']:.1%}")
    columns[1].metric("Plausible subject–verb", f"{metrics['plausible_subject_verb']:.1%}")
    columns[2].metric("Novel sentences", f"{metrics['novel_sentences']:.1%}")
    columns[3].metric("Distinct-2", f"{metrics['distinct_2']:.1%}")
    st.caption("Valid pattern and subject–verb mean seen in training; novelty means not copied from training. Distinct-2 is the fraction of unique word bigrams in the batch.")
    st.dataframe(frame, hide_index=True, use_container_width=True, height=360)
    st.download_button("Download CSV", frame.to_csv(index=False).encode("utf-8"),
                       file_name="generated_sentences.csv", mime="text/csv")

st.divider()
st.header("Viterbi tagger")
st.caption("Type a sentence to see the most probable part-of-speech tag for each word. Punctuation is a separate token.")
sentence = st.text_input("Sentence to tag", value="the cats watch the birds .")
if sentence.strip():
    words = re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", sentence.lower(), flags=re.UNICODE)
    unknown = sorted({word for word in words if word not in split.vocab})
    if unknown:
        st.warning("Unknown word(s): " + ", ".join(unknown) + ". Tags are estimates using smoothing.")
    tags = split.viterbi(words)
    if words:
        st.dataframe(pd.DataFrame([words, tags], index=["Word", "Viterbi tag"]), use_container_width=True)
