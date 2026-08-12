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
import json
import io
import tempfile
import community as community_louvain  # python-louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
    # Lemmatize FIRST, then filter — the exclusion list should match the word
    # as it actually ends up counted/displayed (the lemma), not the raw
    # inflected form it happened to take in the source text. Filtering on the
    # raw token let plurals/verb forms of an excluded word slip through and
    # then get lemmatized right back into the word you were trying to remove.
    lemmas = (lemmatizer.lemmatize(t) for t in tokens)
    return [
        lemma
        for lemma in lemmas
        if lemma not in STOP_WORDS and lemma not in custom_stops and len(lemma) > 2
    ]

# ─── Network builder ─────────────────────────────────────────────────────────
def build_html(G, partition, word_freq, filename="semantic_map"):
    cluster_ids = sorted(set(partition.values()))
    color_map  = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]  for i, c in enumerate(cluster_ids)}
    border_map = {c: BORDER_COLORS[i % len(BORDER_COLORS)]    for i, c in enumerate(cluster_ids)}

    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333333")

    # Ground-truth per-node styling, computed once in Python. This — and NOT
    # anything read back out of the live vis.js DataSet at click-time — is
    # what the JS below uses to restore colors. Reading "originals" out of the
    # rendered network after it may already have been mutated by a previous
    # fade/highlight is what caused clusters after the first to render wrong
    # and "All" to come back grey.
    node_meta = {}

    for node in G.nodes():
        cluster = partition[node]
        freq    = G.nodes[node].get("size", 10)
        x       = G.nodes[node].get("x", 0)
        y       = G.nodes[node].get("y", 0)
        color = {
            "background": color_map[cluster],
            "border":     border_map[cluster],
            "highlight":  {"background": "#FF8000", "border": "#CC5500"},
        }
        font = {"size": 13, "color": "#ffffff", "face": "Arial",
                "strokeWidth": 2, "strokeColor": "rgba(0,0,0,0.3)"}
        node_meta[node] = {"color": color, "font": font, "group": str(cluster)}

        net.add_node(
            node,
            label=node,
            title=f"<b>{node}</b><br>Occurrences: {freq}<br>Cluster: {cluster + 1}",
            color=color,
            size=max(10, min(40, 10 + freq * 1.2)),
            shape="box",
            group=str(cluster),
            x=x, y=y,
            physics=False,
            font=font,
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

    # ── JS: NODE_META is authored once in Python and never mutated in JS.
    #        showAll()/filterCluster() always reset from THIS, never from
    #        whatever the live DataSet currently happens to show — so
    #        switching clusters repeatedly, or hitting "All" after several
    #        switches, always reproduces the correct original colors.  ───────
    node_meta_json = json.dumps(node_meta)

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
  <div onclick="exportPNG()"
    style="background:#2b2b2b;color:#fff;padding:6px 14px;border-radius:20px;
    cursor:pointer;font-size:12px;font-weight:bold;
    white-space:nowrap;user-select:none;margin-left:6px;">📷 PNG</div>
</div>

<script>
// ── Immutable ground truth, authored in Python — never derived from the
//    live/rendered network, so it can never pick up a faded/highlighted
//    state by accident. ───────────────────────────────────────────────────
var NODE_META = {node_meta_json};   // {{ nodeId: {{ color, font, group }} }}

var FADE_NODE = {{ background:"rgba(220,220,220,0.25)", border:"rgba(200,200,200,0.2)",
                   highlight:{{ background:"rgba(220,220,220,0.25)", border:"rgba(200,200,200,0.2)" }} }};
var FADE_FONT = {{ color:"rgba(180,180,180,0.25)", strokeWidth:0 }};
var DIM_EDGE  = "rgba(200,200,200,0.12)";
var FULL_EDGE = "#c8d8e8";

function showAll() {{
  network.body.data.nodes.update(
    Object.keys(NODE_META).map(function(id) {{
      var m = NODE_META[id];
      return {{ id:id, color:m.color, font:m.font }};
    }})
  );
  network.body.data.edges.update(
    network.body.data.edges.get().map(function(e) {{
      return {{ id:e.id, color:{{ color:FULL_EDGE, highlight:"#FF8000" }} }};
    }})
  );
}}

function filterCluster(cid) {{
  var cs = String(cid);

  // Which node ids belong to the target cluster — from NODE_META, always.
  var inCluster = {{}};
  Object.keys(NODE_META).forEach(function(id) {{
    if (NODE_META[id].group === cs) inCluster[id] = true;
  }});

  // Every node gets an explicit, fully-specified color/font on every call —
  // selected nodes from NODE_META (true originals), everything else faded.
  network.body.data.nodes.update(
    Object.keys(NODE_META).map(function(id) {{
      var m = NODE_META[id];
      if (inCluster[id]) {{
        return {{ id:id, color:m.color, font:m.font }};
      }} else {{
        return {{ id:id, color:FADE_NODE, font:FADE_FONT }};
      }}
    }})
  );

  network.body.data.edges.update(
    network.body.data.edges.get().map(function(e) {{
      var keep = inCluster[e.from] && inCluster[e.to];
      return {{ id:e.id, color:{{ color: keep ? "#7ab4c8" : DIM_EDGE, highlight:"#FF8000" }} }};
    }})
  );
}}

function exportPNG() {{
  try {{
    var canvas = network.canvas.frame.canvas;
    var link = document.createElement("a");
    link.download = "{filename}.png";
    link.href = canvas.toDataURL("image/png");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }} catch (e) {{
    alert("PNG export failed: " + e.message);
  }}
}}
</script>
"""
    # Inject just before </body>
    return html.replace("</body>", inject + "\n</body>")


# ─── Cluster bubbles (circle-packing) builder ────────────────────────────────
def build_bubbles_html(word_freq, full_partition, cluster_ids, scope, filename="cluster_bubbles"):
    """Force-directed, draggable bubble chart: each word is its own bubble,
    sized by frequency and colored by cluster, gently pulled toward its
    cluster's 'gravity well' but free to be dragged around. Replaces the old
    static circle-packing layout, which couldn't be interacted with and
    under-sized most word labels to invisibility."""
    color_map = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(cluster_ids)}

    if scope == "Entire sample":
        scopes = list(enumerate(cluster_ids))
    else:
        idx = int(scope.split(" ")[1]) - 1
        scopes = [(idx, cluster_ids[idx])]

    nodes = []
    for i, cid in scopes:
        members = [w for w, c in full_partition.items() if c == cid]
        for w in members:
            nodes.append({
                "id": w,
                "name": w,
                "value": max(1, int(word_freq.get(w, 1))),
                "cluster": cid,
                "clusterLabel": f"Cluster {i+1}",
            })

    if not nodes:
        return None

    nodes_json  = json.dumps(nodes)
    colors_json = json.dumps(color_map)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; background:#ffffff; font-family:Arial, sans-serif; overflow:hidden; }}
  #wrap {{ position:relative; width:100%; height:100%; }}
  #toolbar {{
    position:absolute; top:12px; left:50%; transform:translateX(-50%); z-index:9999;
    background:rgba(255,255,255,0.96); padding:8px 16px; border-radius:40px;
    box-shadow:0 2px 14px rgba(0,0,0,0.13); border:1px solid #e8e8e8;
    display:flex; align-items:center; gap:8px; font-size:12px;
  }}
  #toolbar div.btn {{
    background:#2b2b2b;color:#fff;padding:6px 14px;border-radius:20px;
    cursor:pointer;font-weight:bold;white-space:nowrap;user-select:none;
  }}
  #toolbar div.btn.reset {{ background:#f0f0f0;color:#555;border:1px solid #ddd; }}
  .bubble-label {{ pointer-events:none; font-weight:600; fill:#fff; text-shadow:0 1px 2px rgba(0,0,0,0.4); }}
  .group-label {{ pointer-events:none; font-weight:700; fill:#555; letter-spacing:.04em; }}
  circle.word {{ cursor:grab; }}
  circle.word:active {{ cursor:grabbing; }}
  #tooltip {{
    position:absolute; z-index:10000; pointer-events:none; opacity:0;
    background:rgba(30,30,30,0.94); color:#fff; padding:8px 12px; border-radius:8px;
    font-size:12px; line-height:1.5; box-shadow:0 4px 16px rgba(0,0,0,0.25);
    transition:opacity 0.12s ease; max-width:220px;
  }}
  #tooltip b {{ font-size:13px; }}
  #tooltip .swatch {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="toolbar">
    <div class="btn reset" onclick="resetView()">↺ Reset</div>
    <div class="btn" onclick="exportPNG()">📷 PNG</div>
  </div>
  <div id="tooltip"></div>
  <svg id="viz"></svg>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
var NODES  = {nodes_json};
var COLORS = {colors_json};
var W = window.innerWidth, H = Math.max(window.innerHeight, 560);

var svg = d3.select("#viz").attr("width", W).attr("height", H)
            .attr("viewBox", [0, 0, W, H]);
var g = svg.append("g");
var tooltip = d3.select("#tooltip");

var zoomBeh = d3.zoom().scaleExtent([0.3, 8]).on("zoom", function(ev) {{
  g.attr("transform", ev.transform);
}});
svg.call(zoomBeh);

// ── Radius scale — floor kept high enough that short words almost always fit
var maxFreq = d3.max(NODES, function(d) {{ return d.value; }}) || 1;
var rScale = d3.scaleSqrt().domain([1, maxFreq]).range([16, 58]);
NODES.forEach(function(d) {{ d.r = rScale(d.value); }});

// ── Cluster "gravity well" centers, spread evenly around the canvas
var clusterIds = Array.from(new Set(NODES.map(function(d) {{ return d.cluster; }})));
var centers = {{}};
if (clusterIds.length === 1) {{
  centers[clusterIds[0]] = {{ x: W / 2, y: H / 2 }};
}} else {{
  var cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.33;
  clusterIds.forEach(function(cid, i) {{
    var angle = (i / clusterIds.length) * 2 * Math.PI - Math.PI / 2;
    centers[cid] = {{ x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) }};
  }});
}}

// ── Approximate "zone" circle behind each cluster's bubbles (static, purely
//    visual — real positions can drift slightly once dragged) ─────────────
var zoneR = {{}};
clusterIds.forEach(function(cid) {{
  var members = NODES.filter(function(d) {{ return d.cluster === cid; }});
  var area = members.reduce(function(s, d) {{ return s + d.r * d.r; }}, 0);
  zoneR[cid] = Math.sqrt(area) * 1.5 + 24;
}});

if (clusterIds.length > 1) {{
  g.selectAll("circle.zone")
    .data(clusterIds)
    .join("circle")
    .attr("class", "zone")
    .attr("cx", function(d) {{ return centers[d].x; }})
    .attr("cy", function(d) {{ return centers[d].y; }})
    .attr("r", function(d) {{ return zoneR[d]; }})
    .attr("fill", function(d) {{ return (COLORS[d] || "#999999") + "14"; }})
    .attr("stroke", function(d) {{ return (COLORS[d] || "#999999") + "55"; }})
    .attr("stroke-width", 1.5);

  g.selectAll("text.group-label")
    .data(clusterIds)
    .join("text")
    .attr("class", "group-label")
    .attr("text-anchor", "middle")
    .attr("x", function(d) {{ return centers[d].x; }})
    .attr("y", function(d) {{ return centers[d].y - zoneR[d] - 10; }})
    .style("font-size", "13px")
    .text(function(d, i) {{ return NODES.find(function(n) {{ return n.cluster === d; }}).clusterLabel; }});
}}

// ── Word bubbles ────────────────────────────────────────────────────────
var node = g.selectAll("g.node")
  .data(NODES)
  .join("g")
  .attr("class", "node");

node.append("circle")
  .attr("class", "word")
  .attr("r", function(d) {{ return d.r; }})
  .attr("fill", function(d) {{ return COLORS[d.cluster] || "#999999"; }})
  .attr("stroke", "rgba(0,0,0,0.18)")
  .attr("stroke-width", 1)
  .on("mouseenter", function(ev, d) {{
    d3.select(this).attr("stroke", "#333").attr("stroke-width", 2);
    tooltip.style("opacity", 1).html(
      '<span class="swatch" style="background:' + (COLORS[d.cluster] || "#999") + '"></span>' +
      '<b>' + d.name + '</b><br>Frequency: ' + d.value + '<br>' + d.clusterLabel
    );
  }})
  .on("mousemove", function(ev) {{
    var box = document.getElementById("wrap").getBoundingClientRect();
    tooltip.style("left", (ev.clientX - box.left + 16) + "px")
           .style("top",  (ev.clientY - box.top + 12) + "px");
  }})
  .on("mouseleave", function() {{
    d3.select(this).attr("stroke", "rgba(0,0,0,0.18)").attr("stroke-width", 1);
    tooltip.style("opacity", 0);
  }});

// ── Label: font-size fit to the bubble, falling back to truncation only
//    when even the smallest readable size can't fit the whole word ────────
function fitLabel(d) {{
  var usable = d.r * 1.7;
  var estCharW = 0.62;
  var size = Math.min(15, Math.max(8, usable / (d.name.length * estCharW)));
  var maxChars = Math.max(3, Math.floor(usable / (size * estCharW)));
  var text = d.name.length > maxChars ? d.name.slice(0, maxChars - 1) + "…" : d.name;
  return {{ size: size, text: text }};
}}

node.append("text")
  .attr("class", "bubble-label")
  .attr("text-anchor", "middle")
  .attr("dy", "0.32em")
  .style("font-size", function(d) {{ return fitLabel(d).size + "px"; }})
  .text(function(d) {{ return fitLabel(d).text; }});

// ── Force simulation: cluster gravity + collision + light repulsion ───────
var simulation = d3.forceSimulation(NODES)
  .force("x", d3.forceX(function(d) {{ return centers[d.cluster].x; }}).strength(0.08))
  .force("y", d3.forceY(function(d) {{ return centers[d.cluster].y; }}).strength(0.08))
  .force("collide", d3.forceCollide(function(d) {{ return d.r + 2; }}).strength(0.9))
  .force("charge", d3.forceManyBody().strength(-1))
  .on("tick", ticked);

function ticked() {{
  node.attr("transform", function(d) {{ return "translate(" + d.x + "," + d.y + ")"; }});
}}

node.call(
  d3.drag()
    .on("start", function(ev, d) {{
      if (!ev.active) simulation.alphaTarget(0.25).restart();
      d.fx = d.x; d.fy = d.y;
    }})
    .on("drag", function(ev, d) {{
      d.fx = ev.x; d.fy = ev.y;
    }})
    .on("end", function(ev, d) {{
      if (!ev.active) simulation.alphaTarget(0);
      d.fx = null; d.fy = null;
    }})
);

function resetView() {{
  svg.transition().duration(400).call(zoomBeh.transform, d3.zoomIdentity);
  NODES.forEach(function(d) {{ d.fx = null; d.fy = null; }});
  simulation.alpha(0.6).restart();
}}

function exportPNG() {{
  var svgEl = document.getElementById("viz");
  var serializer = new XMLSerializer();
  var source = serializer.serializeToString(svgEl);
  if (!source.match(/^<svg[^>]+xmlns="http:\\/\\/www\\.w3\\.org\\/2000\\/svg"/)) {{
    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }}
  var svgBlob = new Blob([source], {{type: "image/svg+xml;charset=utf-8"}});
  var url = URL.createObjectURL(svgBlob);
  var img = new Image();
  img.onload = function() {{
    var scale = 2;
    var canvas = document.createElement("canvas");
    canvas.width = W * scale;
    canvas.height = H * scale;
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, W, H);
    URL.revokeObjectURL(url);
    var link = document.createElement("a");
    link.download = "{filename}.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  }};
  img.src = url;
}}
</script>
</body>
</html>
"""
    return html



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

            cluster_ids = sorted(set(best_p.values()))

            # ── Everything downstream (map html, word cloud, bubbles) only
            #    needs word_freq / G / best_p / cluster_ids — precompute the
            #    one expensive artifact (the map) now and stash it all in
            #    session_state. Rendering below then runs on EVERY rerun
            #    (including ones triggered by a selectbox change), not just
            #    the run where the button was clicked. ──────────────────────
            st.session_state["results"] = {
                "word_freq": word_freq,
                "G": G,
                "best_p": best_p,
                "cluster_ids": cluster_ids,
                "html_map": build_html(G, best_p, word_freq, filename="semantic_map"),
            }

# ── Render from session_state — survives selectbox/slider reruns ────────────
if "results" in st.session_state:
    res = st.session_state["results"]
    word_freq   = res["word_freq"]
    G           = res["G"]
    best_p      = res["best_p"]
    cluster_ids = res["cluster_ids"]
    html_map    = res["html_map"]

    # ── Cluster summary cards ───────────────────────────────────────────────
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

    # Download button in sidebar — now stable across reruns since html_map
    # comes from session_state rather than only existing mid-button-click.
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        "💾 Download HTML map",
        data=html_map,
        file_name="semantic_map.html",
        mime="text/html",
        use_container_width=True,
        key="download_map_html",
    )

    st.components.v1.html(html_map, height=750, scrolling=False)

    # ── Word cloud ───────────────────────────────────────────────────────────
    st.markdown("### ☁️ Word Cloud")
    cloud_options = ["Entire sample"] + [f"Cluster {i+1}" for i in range(len(cluster_ids))]
    cloud_scope = st.selectbox("Show word cloud for:", cloud_options, key="cloud_scope")

    if cloud_scope == "Entire sample":
        cloud_freqs = dict(word_freq)
    else:
        idx = int(cloud_scope.split(" ")[1]) - 1
        cid = cluster_ids[idx]
        cloud_freqs = {w: word_freq[w] for w, c in best_p.items() if c == cid}

    if cloud_freqs:
        wc = WordCloud(
            width=1100, height=550, background_color="white",
            colormap="viridis", prefer_horizontal=0.9,
        ).generate_from_frequencies(cloud_freqs)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        st.download_button(
            "💾 Download word cloud (PNG)",
            data=buf.getvalue(),
            file_name=f"wordcloud_{cloud_scope.replace(' ', '_').lower()}.png",
            mime="image/png",
            use_container_width=True,
            key="download_wordcloud_png",
        )
    else:
        st.info("No words to display for this selection.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cluster bubbles (circle-packing, replaces the old word tree) ─────────
    st.markdown("### 🔵 Cluster Bubbles")
    st.caption("Word size = frequency · color = cluster · drag a bubble to move it · scroll/drag background to zoom & pan")
    bubble_options = ["Entire sample"] + [f"Cluster {i+1}" for i in range(len(cluster_ids))]
    bubble_scope = st.selectbox("Show bubbles for:", bubble_options, key="bubble_scope")

    bubble_fname = (
        "cluster_bubbles_all" if bubble_scope == "Entire sample"
        else f"cluster_bubbles_{bubble_scope.replace(' ', '_').lower()}"
    )
    bubble_html = build_bubbles_html(word_freq, best_p, cluster_ids, bubble_scope, filename=bubble_fname)
    if bubble_html:
        st.components.v1.html(bubble_html, height=650, scrolling=False)
        st.download_button(
            "💾 Download cluster bubbles (HTML)",
            data=bubble_html,
            file_name=f"{bubble_fname}.html",
            mime="text/html",
            use_container_width=True,
            key="download_bubbles_html",
        )
    else:
        st.info("No words to display for this selection.")
