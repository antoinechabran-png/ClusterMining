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
import community as community_louvain  # python-louvain

# --- Settings ---
st.set_page_config(page_title="English Semantic Explorer", layout="wide")

# Download required NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Standard noise words
DEFAULT_EXCLUSIONS = [
    "product", "smell", "feel", "really", "just", "like", "little",
    "think", "lot", "make", "also", "bit", "quite", "something",
    "seem", "evoke", "find", "remind"
]
STOP_WORDS = set(stopwords.words("english"))

# Professional Palette
CLUSTER_COLORS = ["#0085AF", "#E07B39", "#6AAB6A", "#C62F4B", "#8B6BB1", "#E8C63C", "#4BA8B0", "#B85C8A", "#7B9E3E", "#D4724A"]
BORDER_COLORS = ["#013848", "#7A3A10", "#2A6A2A", "#6B0020", "#4A2A7A", "#8A6A00", "#1A5A60", "#6A1A4A", "#3A5A00", "#7A2A00"]

@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()

def preprocess_english(text, lemmatizer, custom_stops):
    if not isinstance(text, str) or text.strip() == "": return []
    text = text.lower()
    text = re.sub(r"\b(not|no|don't|can't|won't|never)\s+(\w+)", r"not_\2", text)
    tokens = re.findall(r"\b[a-z][a-z]+\b", text)
    return [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and t not in custom_stops and len(t) > 2]

def build_network_html(G, partition):
    cluster_ids = sorted(set(partition.values()))
    color_map  = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(cluster_ids)}
    border_map = {c: BORDER_COLORS[i % len(BORDER_COLORS)] for i, c in enumerate(cluster_ids)}

    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="#333333")
    
    for node in G.nodes():
        cluster = partition[node]
        freq = G.nodes[node].get("size", 10)
        net.add_node(node, label=node, title=f"Occurrences: {freq}",
            color={"background": color_map[cluster], "border": border_map[cluster], "highlight": {"background": "#FF8000", "border": "#CC5500"}},
            size=max(10, min(40, 10 + freq * 1.2)), shape="dot", group=str(cluster),
            x=G.nodes[node].get("x", 0), y=G.nodes[node].get("y", 0), physics=False,
            font={"size": 14, "color": "#333333", "face": "Arial", "strokeWidth": 2, "strokeColor": "#ffffff"},
            borderWidth=2)

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=data.get("weight", 1), color={"color": "#e0e0e0", "highlight": "#FF8000"}, smooth=False)

    net.set_options('{"physics": {"enabled": false}, "interaction": {"hover": true, "zoomSpeed": 0.5}}')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f: html = f.read()

    # Layout mimicking your reference: Clean Toolbar
    buttons_html = '<button onclick="showAll()" style="margin:3px;padding:6px 12px;border-radius:20px;border:1px solid #ddd;cursor:pointer;background:#fff;font-size:12px;font-weight:bold;">🔄 Reset View</button>'
    for i, c in enumerate(cluster_ids):
        col = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        buttons_html += f'<button onclick="filterCluster(\'{c}\')" style="margin:3px;padding:6px 12px;border-radius:20px;border:none;cursor:pointer;background:{col};color:#fff;font-size:12px;">Cluster {i+1}</button>'

    inject = f"""
<div id="cluster-toolbar" style="position:absolute;top:15px;left:50%;transform:translateX(-50%);z-index:1000;background:rgba(255,255,255,0.9);padding:8px 20px;border-radius:50px;box-shadow:0 2px 10px rgba(0,0,0,0.1);display:flex;gap:5px;align-items:center;border:1px solid #eee;">
  <span style="font-family:Arial; font-size:12px; font-weight:bold; color:#666; margin-right:10px;">ISOLATE:</span>
  {buttons_html}
</div>
<script>
  var _originalColors = {{}};
  var FADE_BG = 'rgba(240,240,240,0.3)', FADE_BORDER = 'rgba(220,220,220,0.3)', FADE_FONT = 'rgba(200,200,200,0.3)', FADE_EDGE = 'rgba(240,240,240,0.1)';
  function _saveOriginal() {{
    if (typeof network === 'undefined') return false;
    if (Object.keys(_originalColors).length > 0) return true;
    network.body.data.nodes.get().forEach(function(n) {{ _originalColors[n.id] = {{ color: JSON.parse(JSON.stringify(n.color || {{}})), font: JSON.parse(JSON.stringify(n.font || {{}})) }}; }});
    return true;
  }}
  function showAll() {{
    if(!_saveOriginal()) return;
    network.body.data.nodes.update(network.body.data.nodes.get().map(n => ({{ id: n.id, color: _originalColors[n.id].color, font: {{ color: "#333" }} }})));
    network.body.data.edges.update(network.body.data.edges.get().map(e => ({{ id: e.id, color: {{ color: '#e0e0e0' }} }})));
  }}
  function filterCluster(cid) {{
    if(!_saveOriginal()) return;
    network.body.data.nodes.update(network.body.data.nodes.get().map(n => (
      String(n.group) === String(cid) ? {{ id: n.id, color: _originalColors[n.id].color, font: {{ color: "#333", strokeWidth: 2, strokeColor: "#fff" }} }}
      : {{ id: n.id, color: {{ background: FADE_BG, border: FADE_BORDER }}, font: {{ color: FADE_FONT, strokeWidth: 0 }} }}
    )));
    network.body.data.edges.update(network.body.data.edges.get().map(e => ({{
      id: e.id, color: {{ color: (String(network.body.data.nodes.get(e.from).group) === String(cid) && String(network.body.data.nodes.get(e.to).group) === String(cid)) ? '#888' : FADE_EDGE }}
    }})));
  }}
</script>
"""
    return html.replace("</body>", inject + "\n</body>")

# --- Streamlit Layout ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80) # Placeholder for your logo
    st.title("Settings")
    
    uploaded_file = st.file_uploader("📂 Upload Excel corpus", type=["xlsx"])
    
    st.markdown("---")
    min_freq = st.slider("Min occurrences", 1, 50, 5)
    min_edge = st.slider("Min connections", 1, 20, 3)
    n_clusters = st.slider("Target Clusters", 2, 10, 5)
    
    user_extra_stops = st.text_area("Exclusions (comma separated):", "")
    all_stops = set(DEFAULT_EXCLUSIONS + [w.strip().lower() for w in user_extra_stops.split(",") if w.strip()])

st.title("🌐 English Semantic Relationship Map")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    col = st.selectbox("Text Column", df.columns)

    if st.button("🚀 Generate Interactive Analysis", use_container_width=True):
        lemmatizer = load_lemmatizer()
        with st.spinner("Processing..."):
            df["tokens"] = df[col].apply(lambda x: preprocess_english(x, lemmatizer, all_stops))
            word_freq = Counter(itertools.chain.from_iterable(df["tokens"]))
            pair_counts = Counter()
            for tokens in df["tokens"]:
                unique_tokens = sorted(set(tokens))
                for pair in itertools.combinations(unique_tokens, 2): pair_counts[pair] += 1

            G = nx.Graph()
            for (u, v), w in pair_counts.items():
                if w >= min_edge and word_freq[u] >= min_freq and word_freq[v] >= min_freq:
                    G.add_node(u, size=word_freq[u]); G.add_node(v, size=word_freq[v])
                    G.add_edge(u, v, weight=w)

            if len(G.nodes) == 0:
                st.warning("No connections found.")
            else:
                p = community_louvain.best_partition(G)
                pos = nx.spring_layout(G, seed=42, k=3.5 / max(1, len(G.nodes)**0.5))
                for n, (x, y) in pos.items():
                    G.nodes[n]["x"], G.nodes[n]["y"] = float(x)*1000, float(y)*1000

                # Top Metric Row
                st.markdown("### 📊 Cluster Insights")
                cluster_ids = sorted(set(p.values()))
                cols = st.columns(len(cluster_ids))
                for i, cid in enumerate(cluster_ids):
                    members = sorted([w for w, c in p.items() if c == cid], key=lambda x: -word_freq[x])
                    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                    cols[i].markdown(f"""
                        <div style="background:{color}; color:white; padding:12px; border-radius:10px; border-left: 5px solid rgba(0,0,0,0.2);">
                            <div style="font-size:0.8em; opacity:0.8;">CLUSTER {i+1}</div>
                            <div style="font-weight:bold; font-size:1.1em; margin-bottom:5px;">{members[0].upper()}</div>
                            <div style="font-size:0.75em; line-height:1.2;">{", ".join(members[1:4])}</div>
                        </div>
                    """, unsafe_allow_html=True)

                html_map = build_network_html(G, p)
                
                # Sidebar Download
                st.sidebar.markdown("---")
                st.sidebar.download_button("💾 Download HTML Map", data=html_map, file_name="semantic_map.html", mime="text/html", use_container_width=True)
                
                st.components.v1.html(html_map, height=800)
