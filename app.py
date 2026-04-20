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

# Standard noise words to exclude
DEFAULT_EXCLUSIONS = [
    "product", "smell", "feel", "really", "just", "like", "little",
    "think", "lot", "make", "also", "bit", "quite", "something",
    "seem", "evoke", "find", "remind"
]

STOP_WORDS = set(stopwords.words("english"))

# Cluster colour palette
CLUSTER_COLORS = [
    "#0085AF", "#E07B39", "#6AAB6A", "#C62F4B", "#8B6BB1",
    "#E8C63C", "#4BA8B0", "#B85C8A", "#7B9E3E", "#D4724A"
]
BORDER_COLORS = [
    "#013848", "#7A3A10", "#2A6A2A", "#6B0020", "#4A2A7A",
    "#8A6A00", "#1A5A60", "#6A1A4A", "#3A5A00", "#7A2A00"
]

# --- NLP helpers ---
@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()

def preprocess_english(text, lemmatizer, custom_stops):
    if not isinstance(text, str) or text.strip() == "":
        return []
    text = text.lower()
    text = re.sub(r"\b(not|no|don't|can't|won't|never)\s+(\w+)", r"not_\2", text)
    tokens = re.findall(r"\b[a-z][a-z]+\b", text)
    result = []
    for token in tokens:
        if token not in STOP_WORDS and token not in custom_stops and len(token) > 2:
            lemma = lemmatizer.lemmatize(token)
            result.append(lemma)
    return result

# --- Pyvis HTML builder ---
def build_network_html(G, partition):
    cluster_ids = sorted(set(partition.values()))
    color_map  = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]  for i, c in enumerate(cluster_ids)}
    border_map = {c: BORDER_COLORS[i % len(BORDER_COLORS)]    for i, c in enumerate(cluster_ids)}

    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333333")

    for node in G.nodes():
        cluster = partition[node]
        freq = G.nodes[node].get("size", 10)
        node_size = max(10, min(40, 10 + freq * 1.2))
        x = G.nodes[node].get("x", 0)
        y = G.nodes[node].get("y", 0)
        
        net.add_node(
            node,
            label=node,
            title=f"Occurrences: {freq} | Cluster: {cluster + 1}",
            color={
                "background": color_map[cluster],
                "border": border_map[cluster],
                "highlight": {"background": "#FF8000", "border": "#CC5500"},
            },
            size=node_size,
            shape="dot", # Changed to dot for cleaner look
            group=str(cluster),
            x=x, y=y,
            physics=False,
            font={"size": 14, "color": "#333333", "face": "Arial", "strokeWidth": 2, "strokeColor": "#ffffff"},
            borderWidth=2,
        )

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(u, v,
                     value=weight,
                     color={"color": "#e0e0e0", "highlight": "#FF8000"},
                     smooth=False)

    net.set_options("""
    var options = {
      "physics": { "enabled": false },
      "interaction": { "hover": true, "zoomSpeed": 0.5, "dragNodes": true },
      "edges": { "smooth": false }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()

    buttons_html = (
        '<button onclick="showAll()" '
        'style="margin:3px;padding:5px 14px;border-radius:4px;border:1px solid #aaa;'
        'cursor:pointer;background:#f0f0f0;font-size:13px;font-weight:bold;">Show All</button>\n'
    )
    for i, c in enumerate(cluster_ids):
        col = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        buttons_html += (
            f'<button onclick="filterCluster(\'{c}\')" '
            f'style="margin:3px;padding:5px 14px;border-radius:4px;border:none;'
            f'cursor:pointer;background:{col};color:#fff;font-size:13px;">'
            f'Cluster {i + 1}</button>\n'
        )

    inject = f"""
<div id="cluster-toolbar" style="
    position:absolute;top:10px;left:50%;transform:translateX(-50%);
    z-index:1000;background:rgba(255,255,255,0.95);
    padding:10px;border-radius:8px;
    box-shadow:0 4px 15px rgba(0,0,0,0.15);
    display:flex;flex-wrap:wrap;justify-content:center;gap:6px;width:90%;">
  <span style="font-size:14px;font-weight:800;margin-right:10px;color:#333;display:flex;align-items:center;">
    🎯 Highlight:
  </span>
  {buttons_html}
</div>

<script>
  var _originalColors = {{}};
  var FADE_BG     = 'rgba(230,230,230,0.15)';
  var FADE_BORDER = 'rgba(200,200,200,0.15)';
  var FADE_FONT   = 'rgba(180,180,180,0.15)';
  var FADE_EDGE   = 'rgba(220,220,220,0.05)';

  function _saveOriginal() {{
    if (typeof network === 'undefined') return false;
    if (Object.keys(_originalColors).length > 0) return true;
    
    network.body.data.nodes.get().forEach(function(n) {{
      _originalColors[n.id] = {{
        color: JSON.parse(JSON.stringify(n.color || {{}})),
        font:  JSON.parse(JSON.stringify(n.font || {{}}))
      }};
    }});
    return true;
  }}

  function showAll() {{
    if(!_saveOriginal()) return;
    var nodeUpdates = network.body.data.nodes.get().map(function(n) {{
      return {{ id: n.id, color: _originalColors[n.id].color, font: _originalColors[n.id].font }};
    }});
    network.body.data.nodes.update(nodeUpdates);
    var edgeUpdates = network.body.data.edges.get().map(function(e) {{
      return {{ id: e.id, color: {{ color: '#e0e0e0' }} }};
    }});
    network.body.data.edges.update(edgeUpdates);
  }}

  function filterCluster(clusterId) {{
    if(!_saveOriginal()) return;
    var clusterStr = String(clusterId);

    var nodeUpdates = network.body.data.nodes.get().map(function(n) {{
      if (String(n.group) === clusterStr) {{
        return {{ id: n.id, color: _originalColors[n.id].color, font: _originalColors[n.id].font }};
      }} else {{
        return {{
          id: n.id,
          color: {{ background: FADE_BG, border: FADE_BORDER }},
          font: {{ color: FADE_FONT }}
        }};
      }}
    }});
    network.body.data.nodes.update(nodeUpdates);

    var edgeUpdates = network.body.data.edges.get().map(function(e) {{
      var fromNode = network.body.data.nodes.get(e.from);
      var toNode = network.body.data.nodes.get(e.to);
      var inCluster = (String(fromNode.group) === clusterStr && String(toNode.group) === clusterStr);
      return {{
        id: e.id,
        color: {{ color: inCluster ? '#888' : FADE_EDGE }}
      }};
    }});
    network.body.data.edges.update(edgeUpdates);
  }}
</script>
"""
    html = html.replace("</body>", inject + "\n</body>")
    return html


# --- Streamlit UI ---
st.sidebar.title("Map controls")

user_extra_stops = st.sidebar.text_area("Additional exclusion words (comma separated):", "")
all_stops = set(DEFAULT_EXCLUSIONS + [
    w.strip().lower() for w in user_extra_stops.split(",") if w.strip()
])

min_freq = st.sidebar.slider("Min word occurrence", 1, 50, 5)
min_edge = st.sidebar.slider("Min connection strength", 1, 20, 3)
n_clusters = st.sidebar.slider("Number of clusters (approx.)", 2, 10, 5)

st.title("🌐 English Semantic Relationship Map")

uploaded_file = st.file_uploader("Upload your Excel corpus", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    col = st.selectbox("Select text column:", df.columns)

    if st.button("Generate Map"):
        lemmatizer = load_lemmatizer()

        with st.spinner("Analyzing semantic structure..."):
            df["tokens"] = df[col].apply(lambda x: preprocess_english(x, lemmatizer, all_stops))

            word_freq = Counter(itertools.chain.from_iterable(df["tokens"]))
            pair_counts = Counter()
            for tokens in df["tokens"]:
                unique_tokens = sorted(set(tokens))
                for pair in itertools.combinations(unique_tokens, 2):
                    pair_counts[pair] += 1

            G = nx.Graph()
            for (u, v), weight in pair_counts.items():
                if weight >= min_edge:
                    if word_freq[u] >= min_freq and word_freq[v] >= min_freq:
                        G.add_node(u, size=word_freq[u])
                        G.add_node(v, size=word_freq[v])
                        G.add_edge(u, v, weight=weight)

            if len(G.nodes) == 0:
                st.warning("No connections found with current settings.")
            else:
                best_partition, best_diff = None, 999
                for seed in range(20):
                    p = community_louvain.best_partition(G, random_state=seed)
                    diff = abs(len(set(p.values())) - n_clusters)
                    if diff < best_diff:
                        best_diff, best_partition = diff, p

                # INCREASED K for better spacing/visibility
                pos = nx.spring_layout(G, seed=42, k=3.5 / max(1, len(G.nodes) ** 0.5))
                scale = 1000
                for node, (x, y) in pos.items():
                    G.nodes[node]["x"] = float(x) * scale
                    G.nodes[node]["y"] = float(y) * scale

                # --- Cluster Legend ---
                cluster_ids = sorted(set(best_partition.values()))
                st.markdown("### Cluster Insights")
                cols = st.columns(min(len(cluster_ids), 4))
                for i, c in enumerate(cluster_ids):
                    members = sorted([w for w, cl in best_partition.items() if cl == c], key=lambda w: -word_freq.get(w, 0))
                    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                    with cols[i % len(cols)]:
                        st.markdown(
                            f'<div style="background:{color};color:#fff;padding:10px;border-radius:8px;margin-bottom:10px;min-height:80px">'
                            f'<b>Cluster {i+1}</b><br><small>{", ".join(members[:5])}...</small></div>',
                            unsafe_allow_html=True
                        )

                html_map = build_network_html(G, best_partition)
                st.components.v1.html(html_map, height=800)
