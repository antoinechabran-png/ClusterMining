import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
from collections import Counter
import itertools
import re
import tempfile

# --- Settings ---
st.set_page_config(page_title="Semantic Map Explorer", layout="wide")

LANG_MODELS = {
    "English": "en_core_web_sm",
    "French": "fr_core_news_sm",
    "German": "de_core_news_sm",
    "Italian": "it_core_news_sm",
    "Spanish": "es_core_news_sm"
}

DEFAULT_EXCLUSIONS = [
    "product", "smell", "feel", "really", "just", "like", "little", 
    "think", "lot", "make", "also", "bit", "quite", "something", 
    "seem", "evoke", "find", "remind"
]

# --- Optimized Core Functions ---

@st.cache_resource
def load_nlp(language):
    """Loads SpaCy model once and caches it in memory."""
    model_name = LANG_MODELS[language]
    # We disable unnecessary components to save RAM and time
    nlp = spacy.load(model_name, disable=["ner", "parser", "attribute_ruler"])
    return nlp

def clean_text(text, nlp, custom_stops):
    if not isinstance(text, str) or text.strip() == "":
        return []
    
    # 1. Negation Handling: Join 'not' with the next word (e.g., 'not good' -> 'not_good')
    # Works for English (not/don't), French (ne/pas), Spanish/Italian (no), German (nicht)
    text = re.sub(r'\b(not|no|n\'t|don\'t|pas|ne|nicht|no)\s+(\w+)', r'\1_\2', text.lower())
    
    # 2. Process with SpaCy (Pipe optimized for speed)
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        # Filter: Stopwords, Punctuation, Custom exclusions, and short noise
        if (not token.is_stop and not token.is_punct and 
            lemma not in custom_stops and len(lemma) > 2):
            tokens.append(lemma)
    return tokens

# --- Streamlit UI ---

st.sidebar.title("Configuration")

# 1. Language Selection
lang_choice = st.sidebar.selectbox("1. Select Language", list(LANG_MODELS.keys()))

# 2. Exclusion Words
st.sidebar.subheader("2. Filter Words")
user_extra_stops = st.sidebar.text_area("Add more words to exclude (comma separated):", "")
all_stops = set(DEFAULT_EXCLUSIONS + [w.strip().lower() for w in user_extra_stops.split(",") if w.strip()])

# 3. Network Sensitivity
st.sidebar.subheader("3. Map Sensitivity")
min_freq = st.sidebar.slider("Min word frequency", 1, 20, 2)
min_edge = st.sidebar.slider("Min co-occurrence (connection strength)", 1, 10, 2)

st.title("🌐 Interactive Semantic Map")
st.info("Upload an Excel file to visualize word relationships.")

uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    col = st.selectbox("Which column contains the text?", df.columns)
    
    if st.button("Generate Map"):
        # Load the model only when the button is clicked
        nlp = load_nlp(lang_choice)
        
        with st.spinner(f"Analyzing {len(df)} rows..."):
            # Process text
            df['clean_tokens'] = df[col].apply(lambda x: clean_text(x, nlp, all_stops))
            
            # Count word frequency for node size
            word_freq = Counter(itertools.chain.from_iterable(df['clean_tokens']))
            
            # Count co-occurrences for edges
            pair_counts = Counter()
            for tokens in df['clean_tokens']:
                # Filter out unique tokens and sort to avoid duplicate (A,B) and (B,A)
                unique_tokens = sorted(set(tokens))
                for pair in itertools.combinations(unique_tokens, 2):
                    pair_counts[pair] += 1

            # Build NetworkX graph
            G = nx.Graph()
            
            # Add edges if they meet threshold
            for (u, v), weight in pair_counts.items():
                if weight >= min_edge:
                    # Only add words that appear often enough
                    if word_freq[u] >= min_freq and word_freq[v] >= min_freq:
                        G.add_node(u, size=word_freq[u]*2, title=f"Frequency: {word_freq[u]}")
                        G.add_node(v, size=word_freq[v]*2, title=f"Frequency: {word_freq[v]}")
                        G.add_edge(u, v, value=weight, title=f"Co-occurrences: {weight}")

            if len(G.nodes) == 0:
                st.warning("No relationships found. Try lowering the 'Min co-occurrence' threshold.")
            else:
                # Create Pyvis visualization
                net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#111111", notebook=False)
                net.from_nx(G)
                
                # Improve layout physics
                net.toggle_physics(True)
                net.set_options("""
                var options = {
                  "physics": {
                    "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100 },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "stabilization": { "iterations": 150 }
                  }
                }
                """)
                
                # Save to a temporary file to display in Streamlit
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800)
