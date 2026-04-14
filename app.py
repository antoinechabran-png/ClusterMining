import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from pyvis.network import Network
from collections import Counter
import itertools
import re

# --- Configuration & Styling ---
st.set_page_config(page_title="Semantic Explorer", layout="wide")

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

# --- Helper Functions ---

@st.cache_resource
def load_nlp(language):
    model_name = LANG_MODELS[language]
    try:
        return spacy.load(model_name)
    except OSError:
        st.error(f"Model {model_name} not found. Please install it.")
        return None

def preprocess_text(text, nlp, custom_stops):
    if not isinstance(text, str):
        return []
    
    # Simple Negation Handling: replace "not good" with "not_good"
    text = re.sub(r'\b(not|no|n\'t|don\'t|pas|ne|nicht|no)\s+(\w+)', r'\1_\2', text.lower())
    
    doc = nlp(text)
    tokens = []
    for token in doc:
        # Lemmatization + Suffix handling (built into SpaCy)
        lemma = token.lemma_.lower()
        if (not token.is_stop and not token.is_punct and 
            lemma not in custom_stops and len(lemma) > 1):
            tokens.append(lemma)
    return tokens

# --- UI Layout ---

st.title("🌐 Interactive Semantic Mapper")

tabs = st.tabs(["📁 Data Upload", "⚙️ Preprocessing", "📊 Visualization"])

with tabs[0]:
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    lang_choice = st.selectbox("Select Corpus Language", list(LANG_MODELS.keys()))
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        column_to_analyze = st.selectbox("Select text column", df.columns)

with tabs[1]:
    st.subheader("Text Cleaning Settings")
    extra_stops = st.text_area("Additional exclusion words (comma separated):", "")
    
    # Combine exclusions
    all_stops = set(DEFAULT_EXCLUSIONS + [w.strip().lower() for w in extra_stops.split(",") if w.strip()])
    
    st.write(f"**Current Exclusions:** {', '.join(list(all_stops)[:15])}...")

with tabs[2]:
    if uploaded_file and st.button("Generate Semantic Map"):
        nlp = load_nlp(lang_choice)
        
        with st.spinner("Processing text and building relationships..."):
            # 1. Process Text
            df['tokens'] = df[column_to_analyze].apply(lambda x: preprocess_text(x, nlp, all_stops))
            
            # 2. Build Co-occurrence
            # Using simple window-based co-occurrence within sentences/rows
            pair_counts = Counter()
            for tokens in df['tokens']:
                # Generate unique pairs within each row
                for pair in itertools.combinations(sorted(set(tokens)), 2):
                    pair_counts[pair] += 1

            # 3. Create Network
            G = nx.Graph()
            # Filter for visibility: only pairs that occur > threshold
            threshold = 2 
            for (u, v), weight in pair_counts.items():
                if weight >= threshold:
                    G.add_edge(u, v, weight=weight)

            # 4. Pyvis Visualization
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            net.from_nx(G)
            
            # Repulsion physics for better spacing
            net.toggle_physics(True)
            
            # Save and display
            path = "semantic_map.html"
            net.save_graph(path)
            with open(path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            
            st.components.v1.html(html_data, height=650)
