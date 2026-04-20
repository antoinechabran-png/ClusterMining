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

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="English Semantic Explorer", layout="wide")

# ─── NLTK ───────────────────────────────────────────────────────────────────
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("omw-1.4",  quiet=True)

# ─── Constants ──────────────────────────────────────────────────────────────
DEFAULT_EXCLUSIONS = [
    "product", "smell", "feel", "really", "just", "like", "little",
    "think", "lot", "make", "also", "bit", "quite", "something",
    "seem", "evoke", "find", "remind",
]
STOP_WORDS = set(stopwords.words("english"))

# Same palette as reference (5 clusters → extend if needed)
CLUSTER_COLORS = [
    "#0085AF",  # 1 – teal-blue
    "#E8A838",  # 2 – amber
    "#C62F4B",  # 3 – red
    "#6AAB6A",  # 4 – green
    "#8B6BB1",  # 5 – purple
    "#4BA8B0",  # 6
    "#E07B39",  # 7
    "#B85C8A",  # 8
    "#7B9E3E",  # 9
    "#D4724A",  # 10
]
BORDER_COLORS = [
    "#013848", "#7A5010", "#6B0020", "#2A6A2A", "#4A2A7A",
    "#1A5A60", "#7A3A10", "#6A1A4A", "#3A5A00", "#7A2A00",
]

# ─── NLP ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_lemmatizer():
    return WordNetLemmatizer()

def preprocess(text, lemmatizer, custom_stops):
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    text = re.sub(r"\b(not|no|don't|can't|won't|never)\s+(\w+)", r"not_\2", text)
    tokens = re.findall(r"\b[a-z][a-z]+\b", text)
    return [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and t not in custom_stops and len(t) > 2
    ]

# ─── Network builder ─────────────────────────────────────────────────────────
def build_html(G, partition, word_freq):
    cluster_ids = sorted(set(partition.values()))
    color_map  = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]  for i, c in enumerate(cluster_ids)}
    border_map = {c: BORDER_COLORS[i % len(BORDER_COLORS)]    for i, c in enumerate(cluster_ids)}

    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333333")

    for node in G.nodes():
        cluster = partition[node]
        freq    = G.nodes[node].get("size", 10)
        x       = G.nodes[node].get("x", 0)
        y       = G.nodes[node].get("y", 0)
        net.add_node(
            node,
            label=node,
            title=f"<b>{node}</b><br>Occurrences: {freq}<br>Cluster: {cluster + 1}",
            color={
                "background": color_map[cluster],
                "border":     border_map[cluster],
                "highlight":  {"background": "#FF8000", "border": "#CC5500"},
            },
            size=max(10, min(40, 10 + freq * 1.2)),
            shape="box",
            group=str(cluster),
            x=x, y=y,
            physics=False,
            font={"size": 13, "color": "#ffffff", "face": "Arial",
                  "strokeWidth": 2, "strokeColor": "rgba(0,0,0,0.3)"},
            borderWidth=2,
            shadow={"enabled": True, "color": "rgba(0,0,0,0.15)", "size": 6, "x": 2, "y": 2},
        )

    for u, v, data in G.edges(data=True):
        net.add_edge(
            u, v,
            value=data.get("weight", 1),
            color={"color": "#c8d8e8", "highlight": "#FF8000", "opacity": 0.7},
            smooth=False,
        )

    # Physics off, hover on, same zoom speed as reference
    net.set_options("""{
      "physics": {"enabled": false},
      "interaction": {"hover": true, "zoomSpeed": 1},
      "edges": {"smooth": false}
    }""")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()

    # ── Cluster legend pills ──────────────────────────────────────────────────
    legend_pills = ""
    for i, c in enumerate(cluster_ids):
        members = sorted(
            [w for w, cl in partition.items() if cl == c],
            key=lambda w: -word_freq.get(w, 0),
        )
        top = members[0].upper() if members else f"C{i+1}"
        col = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        legend_pills += (
            f'<div onclick="filterCluster({c})" '
            f'style="background:{col};color:#fff;padding:6px 14px;border-radius:20px;'
            f'cursor:pointer;font-size:12px;font-weight:bold;white-space:nowrap;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.18);user-select:none;" '
            f'title="{", ".join(members[:6])}">'
            f'● C{i+1} – {top}'
            f'</div>\n'
        )

    # ── JS: store originals from node DATA (not live node objects) so
    #        switching clusters always restores from the ground-truth snapshot,
    #        never from an already-faded state.  ──────────────────────────────
    inject = f"""
<!-- ═══ CLUSTER TOOLBAR ═══ -->
<div id="ctoolbar" style="
  position:absolute; top:14px; left:50%; transform:translateX(-50%);
  z-index:9999;
  background:rgba(255,255,255,0.96);
  padding:8px 18px;
  border-radius:40px;
  box-shadow:0 2px 14px rgba(0,0,0,0.13);
  border:1px solid #e8e8e8;
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
  <span style="font-size:11px;font-weight:700;color:#888;letter-spacing:.08em;margin-right:4px;">ISOLATE</span>
  <div onclick="showAll()"
    style="background:#f0f0f0;color:#555;padding:6px 14px;border-radius:20px;
    cursor:pointer;font-size:12px;font-weight:bold;border:1px solid #ddd;
    white-space:nowrap;user-select:none;">↺ All</div>
  {legend_pills}
</div>

<script>
// ── Ground-truth snapshot (filled once, never mutated) ──────────────────────
var _GT = null;   // {{ nodeId: {{ color, font }} }}

function _ensureGT() {{
  if (_GT !== null) return;
  _GT = {{}};
  network.body.data.nodes.get().forEach(function(n) {{
    _GT[n.id] = {{
      color: JSON.parse(JSON.stringify(n.color || {{}})),
      font:  JSON.parse(JSON.stringify(n.font  || {{}})),
    }};
  }});
}}

var FADE_NODE = {{ background:"rgba(220,220,220,0.25)", border:"rgba(200,200,200,0.2)",
                   highlight:{{ background:"rgba(220,220,220,0.25)", border:"rgba(200,200,200,0.2)" }} }};
var FADE_FONT = {{ color:"rgba(180,180,180,0.25)", strokeWidth:0 }};
var DIM_EDGE  = "rgba(200,200,200,0.12)";
var FULL_EDGE = "#c8d8e8";

function showAll() {{
  _ensureGT();
  network.body.data.nodes.update(
    network.body.data.nodes.get().map(function(n) {{
      return {{ id:n.id, color:_GT[n.id].color, font:_GT[n.id].font }};
    }})
  );
  network.body.data.edges.update(
    network.body.data.edges.get().map(function(e) {{
      return {{ id:e.id, color:{{ color:FULL_EDGE, highlight:"#FF8000" }} }};
    }})
  );
}}

function filterCluster(cid) {{
  _ensureGT();
  var cs = String(cid);

  // Build set of node-ids that belong to the target cluster
  var inCluster = {{}};
  network.body.data.nodes.get().forEach(function(n) {{
    if (String(n.group) === cs) inCluster[n.id] = true;
  }});

  // Update nodes — always read from GT, never from current state
  network.body.data.nodes.update(
    network.body.data.nodes.get().map(function(n) {{
      if (inCluster[n.id]) {{
        return {{ id:n.id, color:_GT[n.id].color, font:_GT[n.id].font }};
      }} else {{
        return {{ id:n.id, color:FADE_NODE, font:FADE_FONT }};
      }}
    }})
  );

  // Update edges
  network.body.data.edges.update(
    network.body.data.edges.get().map(function(e) {{
      var keep = inCluster[e.from] && inCluster[e.to];
      return {{ id:e.id, color:{{ color: keep ? "#7ab4c8" : DIM_EDGE, highlight:"#FF8000" }} }};
    }})
  );
}}
</script>
"""
    # Inject just before </body>
    return html.replace("</body>", inject + "\n</body>")


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    uploaded_file = st.file_uploader("📂 Upload Excel corpus", type=["xlsx"])
    st.markdown("---")
    min_freq   = st.slider("Min word occurrences",         1, 50,  5)
    min_edge   = st.slider("Min connection strength",      1, 20,  3)
    n_clusters = st.slider("Target number of clusters",    2, 10,  5)
    st.markdown("---")
    user_extra_stops = st.text_area("Extra exclusion words (comma-sep):", "")

all_stops = set(
    DEFAULT_EXCLUSIONS
    + [w.strip().lower() for w in user_extra_stops.split(",") if w.strip()]
)

# ─── Main ────────────────────────────────────────────────────────────────────
st.title("🌐 English Semantic Relationship Map")

if uploaded_file:
    df  = pd.read_excel(uploaded_file)
    col = st.selectbox("Text column", df.columns)

    if st.button("🚀 Generate map", use_container_width=True):
        lemmatizer = load_lemmatizer()

        with st.spinner("Analysing text and building graph…"):

            # Tokenise
            df["tokens"] = df[col].apply(lambda x: preprocess(x, lemmatizer, all_stops))

            # Frequencies
            word_freq   = Counter(itertools.chain.from_iterable(df["tokens"]))
            pair_counts = Counter()
            for tokens in df["tokens"]:
                ut = sorted(set(tokens))
                for pair in itertools.combinations(ut, 2):
                    pair_counts[pair] += 1

            # Build graph
            G = nx.Graph()
            for (u, v), w in pair_counts.items():
                if w >= min_edge and word_freq[u] >= min_freq and word_freq[v] >= min_freq:
                    G.add_node(u, size=word_freq[u])
                    G.add_node(v, size=word_freq[v])
                    G.add_edge(u, v, weight=w)

            if len(G.nodes) == 0:
                st.warning("No connections found. Try lowering the sliders.")
                st.stop()

            # ── Louvain clustering ──────────────────────────────────────────
            best_p, best_d = None, 999
            for seed in range(30):
                p    = community_louvain.best_partition(G, random_state=seed)
                diff = abs(len(set(p.values())) - n_clusters)
                if diff < best_d:
                    best_d, best_p = diff, p

            # ── Spring layout → fixed pixel coords ─────────────────────────
            pos = nx.spring_layout(G, seed=42, k=3.5 / max(1, len(G.nodes) ** 0.5))
            for node, (x, y) in pos.items():
                G.nodes[node]["x"] = float(x) * 1000
                G.nodes[node]["y"] = float(y) * 1000

            # ── Cluster summary cards ───────────────────────────────────────
            cluster_ids = sorted(set(best_p.values()))
            st.markdown("### Cluster overview")
            card_cols = st.columns(len(cluster_ids))
            for i, cid in enumerate(cluster_ids):
                members = sorted(
                    [w for w, c in best_p.items() if c == cid],
                    key=lambda w: -word_freq[w],
                )
                col_bg = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                card_cols[i].markdown(
                    f"""<div style="background:{col_bg};color:#fff;padding:12px 10px;
                        border-radius:10px;border-left:5px solid rgba(0,0,0,0.2);">
                        <div style="font-size:.75em;opacity:.8;letter-spacing:.06em;">CLUSTER {i+1}</div>
                        <div style="font-weight:bold;font-size:1.05em;margin:4px 0;">
                          {members[0].upper() if members else "—"}
                        </div>
                        <div style="font-size:.72em;line-height:1.4;opacity:.9;">
                          {", ".join(members[1:5])}{"…" if len(members) > 5 else ""}
                        </div>
                        <div style="font-size:.7em;margin-top:6px;opacity:.75;">
                          {len(members)} words
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Build & render map ──────────────────────────────────────────
            html_map = build_html(G, best_p, word_freq)

            # Download button in sidebar
            st.sidebar.markdown("---")
            st.sidebar.download_button(
                "💾 Download HTML map",
                data=html_map,
                file_name="semantic_map.html",
                mime="text/html",
                use_container_width=True,
            )

            st.components.v1.html(html_map, height=750, scrolling=False)
