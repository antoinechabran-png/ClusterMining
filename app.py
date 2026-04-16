import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import networkx as nx
from pyvis.network import Network
from collections import Counter
import itertools
import re
import tempfile

# --- Settings ---
st.set_page_config(page_title="English Semantic Explorer", layout="wide")

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Standard noise words to exclude
DEFAULT_EXCLUSIONS = [
    "product", "smell", "feel", "really", "just", "like", "little",
    "think", "lot", "make", "also", "bit", "quite", "something",
    "seem", "evoke", "find", "remind"
]

STOP_WORDS = set(stopwords.words('english'))

# --- Functions ---

@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()

def preprocess_english(text, lemmatizer, custom_stops):
    if not isinstance(text, str) or text.strip() == "":
        return []

    text = text.lower()

    # Negation Handling: "don't like" -> "not_like"
    text = re.sub(r"\b(not|no|don't|can't|won't|never)\s+(\w+)", r"not_\2", text)

    # Simple whitespace/punctuation tokenizer (no C deps)
    tokens = re.findall(r'\b[a-z][a-z]+\b', text)

    result = []
    for token in tokens:
        if token not in STOP_WORDS and token not in custom_stops and len(token) > 2:
            lemma = lemmatizer.lemmatize(token)
            result.append(lemma)
    return result

# --- UI ---

st.sidebar.title("Map Controls")

# Exclusion list
user_extra_stops = st.sidebar.text_area("Additional exclusion words (comma separated):", "")
all_stops = set(DEFAULT_EXCLUSIONS + [w.strip().lower() for w in user_extra_stops.split(",") if w.strip()])

# Sensitivity
min_freq = st.sidebar.slider("Min word occurrence", 1, 50, 5)
min_edge = st.sidebar.slider("Min connection strength (co-occurrence)", 1, 20, 3)

st.title("🌐 English Semantic Relationship Map")

uploaded_file = st.file_uploader("Upload your Excel corpus", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    col = st.selectbox("Select the column containing text data:", df.columns)

    if st.button("Generate Map"):
        lemmatizer = load_lemmatizer()

        with st.spinner("Processing text..."):
            # Clean tokens
            df['tokens'] = df[col].apply(lambda x: preprocess_english(x, lemmatizer, all_stops))

            # Frequencies & Pairs
            word_freq = Counter(itertools.chain.from_iterable(df['tokens']))
            pair_counts = Counter()
            for tokens in df['tokens']:
                unique_tokens = sorted(set(tokens))
                for pair in itertools.combinations(unique_tokens, 2):
                    pair_counts[pair] += 1

            # Build Network
            G = nx.Graph()
            for (u, v), weight in pair_counts.items():
                if weight >= min_edge:
                    if word_freq[u] >= min_freq and word_freq[v] >= min_freq:
                        G.add_node(u, size=word_freq[u], title=f"Occurrences: {word_freq[u]}")
                        G.add_node(v, size=word_freq[v], title=f"Occurrences: {word_freq[v]}")
                        G.add_edge(u, v, value=weight)

            if len(G.nodes) == 0:
                st.warning("The map is empty. Try lowering the 'Min occurrence' or 'Min connection' sliders.")
            else:
                # Setup Pyvis
                net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333333")
                net.from_nx(G)

                # Optimized Physics for clean visual
                net.toggle_physics(True)
                net.set_options("""
                var options = {
                  "physics": {
                    "barnesHut": { "gravitationalConstant": -2000, "centralGravity": 0.3, "springLength": 95 },
                    "minVelocity": 0.75
                  }
                }
                """)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        st.components.v1.html(f.read(), height=800)
