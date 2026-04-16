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

# Cluster colour palette (up to 10 clusters)
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
    # Negation handling: "don't like" -> "not_like"
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
    """
    Build a static Pyvis network with cluster colouring and
    a per-cluster highlight selector injected as a custom overlay.
    """
    cluster_ids = sorted(set(partition.values()))
    color_map  = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]  for i, c in enumerate(cluster_ids)}
    border_map = {c: BORDER_COLORS[i % len(BORDER_COLORS)]    for i, c in enumerate(cluster_ids)}

    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#ffffff")

    for node in G.nodes():
        cluster  = partition[node]
        freq     = G.nodes[node].get("size", 10)
        node_size = max(14, min(45, 14 + freq * 1.5))
        x = G.nodes[node].get("x", 0)
        y = G.nodes[node].get("y", 0)
        net.add_node(
            node,
            label=node,
            title=f"Occurrences: {freq} | Cluster: {cluster + 1}",
            color={
                "background": color_map[cluster],
                "border":     border_map[cluster],
                "highlight":  {"background": "#FF8000", "border": "#CC5500"},
            },
            size=node_size,
            shape="box",
            group=str(cluster),
            x=x, y=y,
            physics=False,
            shadow={"enabled": True, "size": 8},
            font={"size": 13, "color": "#ffffff"},
            borderWidth=2,
        )

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(u, v,
                     value=weight,
                     color={"color": "#cccccc", "highlight": "#FF8000"},
                     smooth=False)

    # Completely disable physics → static, no dancing
    net.set_options("""
    var options = {
      "physics": { "enabled": false },
      "interaction": { "hover": true, "zoomSpeed": 1 },
      "edges": { "smooth": false }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()

    # --- Inject cluster highlight toolbar ---
    buttons_html = (
        '<button onclick="showAll()" '
        'style="margin:3px;padding:5px 14px;border-radius:4px;border:1px solid #aaa;'
        'cursor:pointer;background:#f0f0f0;font-size:13px;">All</button>\n'
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
    position:fixed;top:12px;left:50%;transform:translateX(-50%);
    z-index:9999;background:rgba(255,255,255,0.95);
    padding:8px 16px;border-radius:8px;
    box-shadow:0 2px 12px rgba(0,0,0,0.18);
    display:flex;flex-wrap:wrap;align-items:center;gap:4px;">
  <span style="font-size:13px;font-weight:700;margin-right:6px;color:#333;">
    Highlight cluster:
  </span>
  {buttons_html}
</div>

<script>
  var _originalColors = {{}};
  var FADE_BG     = 'rgba(210,210,210,0.20)';
  var FADE_BORDER = 'rgba(180,180,180,0.20)';
  var FADE_FONT   = 'rgba(190,190,190,0.35)';
  var FADE_EDGE   = 'rgba(210,210,210,0.10)';

  function _saveOriginal() {{
    if (Object.keys(_originalColors).length > 0) return;
    network.body.data.nodes.get().forEach(function(n) {{
      _originalColors[n.id] = {{
        color: JSON.parse(JSON.stringify(n.color)),
        font:  JSON.parse(JSON.stringify(n.font || {{}}))
      }};
    }});
  }}

  function showAll() {{
    _saveOriginal();
    var nodeUpdates = network.body.data.nodes.get().map(function(n) {{
      return {{ id: n.id, color: _originalColors[n.id].color, font: _originalColors[n.id].font }};
    }});
    network.body.data.nodes.update(nodeUpdates);
    var edgeUpdates = network.body.data.edges.get().map(function(e) {{
      return {{ id: e.id, color: {{ color: '#cccccc', highlight: '#FF8000' }} }};
    }});
    network.body.data.edges.update(edgeUpdates);
  }}

  function filterCluster(clusterId) {{
    _saveOriginal();
    var clusterStr = String(clusterId);

    var nodeUpdates = network.body.data.nodes.get().map(function(n) {{
      if (String(n.group) === clusterStr) {{
        return {{ id: n.id, color: _originalColors[n.id].color, font: _originalColors[n.id].font }};
      }} else {{
        return {{
          id: n.id,
          color: {{
            background: FADE_BG,
            border: FADE_BORDER,
            highlight: {{ background: FADE_BG, border: FADE_BORDER }}
          }},
          font: {{ color: FADE_FONT }}
        }};
      }}
    }});
    network.body.data.nodes.update(nodeUpdates);

    var clusterSet = new Set(
      network.body.data.nodes.get()
        .filter(function(n) {{ return String(n.group) === clusterStr; }})
        .map(function(n) {{ return n.id; }})
    );
    var edgeUpdates = network.body.data.edges.get().map(function(e) {{
      var inCluster = clusterSet.has(e.from) && clusterSet.has(e.to);
      return {{
        id: e.id,
        color: {{ color: inCluster ? '#666' : FADE_EDGE, highlight: '#FF8000' }}
      }};
    }});
    network.body.data.edges.update(edgeUpdates);
  }}
</script>
"""
    html = html.replace("</body>", inject + "\n</body>")
    return html


# --- Streamlit UI ---

st.sidebar.title("Map Controls")

user_extra_stops = st.sidebar.text_area("Additional exclusion words (comma separated):", "")
all_stops = set(DEFAULT_EXCLUSIONS + [
    w.strip().lower() for w in user_extra_stops.split(",") if w.strip()
])

min_freq  = st.sidebar.slider("Min word occurrence",                    1, 50, 5)
min_edge  = st.sidebar.slider("Min connection strength (co-occurrence)", 1, 20, 3)
n_clusters = st.sidebar.slider("Number of clusters (approx.)",           2, 10, 5)

st.title("🌐 English Semantic Relationship Map")

uploaded_file = st.file_uploader("Upload your Excel corpus", type=["xlsx"])

if uploaded_file:
    df  = pd.read_excel(uploaded_file)
    col = st.selectbox("Select the column containing text data:", df.columns)

    if st.button("Generate Map"):
        lemmatizer = load_lemmatizer()

        with st.spinner("Processing text and building graph…"):

            # Tokenise
            df["tokens"] = df[col].apply(
                lambda x: preprocess_english(x, lemmatizer, all_stops)
            )

            # Frequencies & co-occurrence pairs
            word_freq   = Counter(itertools.chain.from_iterable(df["tokens"]))
            pair_counts = Counter()
            for tokens in df["tokens"]:
                unique_tokens = sorted(set(tokens))
                for pair in itertools.combinations(unique_tokens, 2):
                    pair_counts[pair] += 1

            # Build NetworkX graph
            G = nx.Graph()
            for (u, v), weight in pair_counts.items():
                if weight >= min_edge:
                    if word_freq[u] >= min_freq and word_freq[v] >= min_freq:
                        G.add_node(u, size=word_freq[u])
                        G.add_node(v, size=word_freq[v])
                        G.add_edge(u, v, weight=weight)

            if len(G.nodes) == 0:
                st.warning("The map is empty. Try lowering the sliders.")
            else:
                # --- Cluster detection: Louvain, pick run closest to n_clusters ---
                best_partition, best_diff = None, 999
                for seed in range(30):
                    p = community_louvain.best_partition(G, random_state=seed)
                    diff = abs(len(set(p.values())) - n_clusters)
                    if diff < best_diff:
                        best_diff, best_partition = diff, p

                # --- Spring layout → fixed pixel positions (static) ---
                pos = nx.spring_layout(G, seed=42, k=2.2 / max(1, len(G.nodes) ** 0.5))
                scale = 900
                for node, (x, y) in pos.items():
                    G.nodes[node]["x"] = float(x) * scale
                    G.nodes[node]["y"] = float(y) * scale

                # --- Cluster legend above the map ---
                cluster_ids = sorted(set(best_partition.values()))
                st.markdown("#### Cluster Legend")
                cols = st.columns(min(len(cluster_ids), 5))
                for i, c in enumerate(cluster_ids):
                    members = sorted(
                        [w for w, cl in best_partition.items() if cl == c],
                        key=lambda w: -word_freq.get(w, 0)
                    )
                    color     = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                    top_words = ", ".join(members[:8])
                    suffix    = "…" if len(members) > 8 else ""
                    with cols[i % len(cols)]:
                        st.markdown(
                            f'<div style="background:{color};color:#fff;padding:8px 10px;'
                            f'border-radius:6px;margin-bottom:6px;">'
                            f'<b>Cluster {i+1}</b> ({len(members)} words)<br>'
                            f'<span style="font-size:12px">{top_words}{suffix}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # --- Render map ---
                html = build_network_html(G, best_partition)
                st.components.v1.html(html, height=820, scrolling=False)
