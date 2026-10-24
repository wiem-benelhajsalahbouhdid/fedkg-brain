"""
FedKG-Brain platform
====================
Give it the input of the project, get the output:

  INPUT   a resting-state fMRI scan as a region time series (.1D file: time points x 200 cc200 regions)
  OUTPUT  probability of autism + the brain networks behind it + a 3-D view of the key connections
          (plus: federated training across sites, and prediction of how the connectome evolves)

Run
  pip install -r requirements.txt
  python app.py --root /path/to/fedkg_brain            # local:  http://127.0.0.1:7860
  python app.py --root /content/drive/MyDrive/FedKG-Brain/fedkg_brain --share   # Colab: public link

Expected layout under --root (the project folder; data are NOT part of this repository):
  data/raw/ABIDE_PCP/ABIDE_pcp/cpac/filt_noglobal/*_rois_cc200.1D   ABIDE I (CPAC, cc200) time series
  data/raw/ABIDE_PCP/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv   phenotype file
  data/raw/atlases/cc200_roi_atlas.nii.gz      optional: 3-D brain view
  data/processed/roi_to_network.json           optional: functional-network labels
  results_real/                                cache + trained model (created automatically)

Research prototype. Not a medical device.
"""
import argparse, copy, glob, json, os, re, time, warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from sklearn.metrics import roc_auc_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("FEDKG_ROOT", "./fedkg_brain"))
ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--fast", action="store_true", help="fewer epochs (quick test)")
ARGS, _ = ap.parse_known_args()

ROOT = ARGS.root
RAW_DIR = f"{ROOT}/data/raw/ABIDE_PCP/ABIDE_pcp/cpac/filt_noglobal"
PHENO_CSV = f"{ROOT}/data/raw/ABIDE_PCP/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
ATLAS = f"{ROOT}/data/raw/atlases/cc200_roi_atlas.nii.gz"
ROI_JSON = f"{ROOT}/data/processed/roi_to_network.json"
OUT_DIR = f"{ROOT}/results_real"
WINDOW_FILES = [f"{ROOT}/results_final_v5/windows_pearson_W2_g0_f0.667.npz",
                f"{ROOT}/results_prime/windows_pearson_W2_g0_f0.667.npz", f"{OUT_DIR}/cache_windows.npz"]
os.makedirs(OUT_DIR, exist_ok=True)

N, TOPK, HIDDEN, LR, BATCH = 200, 20, 64, 1e-3, 16
EPOCHS = 8 if ARGS.fast else 50
MIN_SITE = 20
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IU = np.triu_indices(N, 1)
DIAG = np.arange(N)

# palette (dark, neon violet / teal)
INK, MUTED, VIOLET, TEAL, PINK = "#E9E6F5", "#9A93B8", "#A78BFA", "#2DD4BF", "#F472B6"
DIVERGING = [[0.0, "#2DD4BF"], [0.5, "#100E1C"], [1.0, "#A78BFA"]]          # teal ← 0 → violet
SEQ = [[0.0, "#0B0A14"], [0.5, "#5B2D90"], [1.0, "#F0ABFC"]]
NET_PALETTE = ["#A78BFA", "#2DD4BF", "#F472B6", "#60A5FA", "#FBBF24", "#34D399", "#F87171", "#C4B5FD"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def parse_sid(fname):
    m = re.search(r"_(\d{7})_rois", fname) or re.search(r"(\d{5,7})", fname)
    return int(m.group(1)) if m else None


def correlation(ts):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(np.corrcoef(ts, rowvar=False), nan=0.0).astype(np.float32)


def load_dataset():
    cache = f"{OUT_DIR}/cache_corr_cc200.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["R"], d["y"], d["sites"], d["sids"]
    ph = pd.read_csv(PHENO_CSV)
    lab = dict(zip(ph.SUB_ID.astype(int), (ph.DX_GROUP == 1).astype(int)))
    sit = dict(zip(ph.SUB_ID.astype(int), ph.SITE_ID.astype(str)))
    R, y, sites, sids = [], [], [], []
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.1D")))
    log(f"first start: building the connectivity cache from {len(files)} scans ...")
    for f in files:
        sid = parse_sid(os.path.basename(f))
        if sid is None or sid not in lab:
            continue
        ts = np.loadtxt(f)
        if ts.ndim == 2 and ts.shape[1] == N:
            R.append(correlation(ts)); y.append(lab[sid]); sites.append(sit[sid]); sids.append(sid)
    R, y, sites, sids = np.stack(R), np.array(y), np.array(sites), np.array(sids)
    np.savez_compressed(cache, R=R, y=y, sites=sites, sids=sids)
    return R, y, sites, sids


R, y, sites, sids = load_dataset()
n = len(y)
FILE_OF = {parse_sid(os.path.basename(f)): f for f in glob.glob(os.path.join(RAW_DIR, "*.1D"))}
NETS = None
if os.path.exists(ROI_JSON):
    _rj = json.load(open(ROI_JSON))
    NETS = np.array([_rj.get(str(i), "Unassigned") for i in range(N)])
NET_NAMES = sorted(set(NETS)) if NETS is not None else []
NET_ORDER = np.argsort(NETS, kind="stable") if NETS is not None else np.arange(N)
NET_COLOR = {nm: ("#6B6485" if nm == "Unassigned" else NET_PALETTE[i % len(NET_PALETTE)])
             for i, nm in enumerate([x for x in NET_NAMES if x != "Unassigned"] + ["Unassigned"])}
COORDS = None
try:
    if os.path.exists(ATLAS):
        from nilearn import plotting as _nplot
        COORDS = np.asarray(_nplot.find_parcellation_cut_coords(ATLAS))
        COORDS = COORDS if len(COORDS) == N else None
except Exception as e:
    log("3-D brain view disabled:", e)
log(f"{n} subjects · {len(np.unique(sites))} sites · device {DEV} · networks {len(NET_NAMES)} · atlas {COORDS is not None}")


# ─────────────────────────────────────────────────────────────────────────────
# Model: top-20 brain graph + 2-layer GCN (the project's recipe)
# ─────────────────────────────────────────────────────────────────────────────
def topk_adjacency(M, k=TOPK, chunk=64):
    out = np.zeros(M.shape, np.float32)
    for a in range(0, len(M), chunk):
        A = np.abs(M[a:a + chunk]).astype(np.float32)
        A[:, DIAG, DIAG] = 0
        part = np.argpartition(-A, k - 1, axis=2)[:, :, :k]
        np.put_along_axis(out[a:a + chunk], part, np.take_along_axis(A, part, axis=2), axis=2)
    return out


def gcn_operator(A):
    M = A + torch.eye(A.shape[-1], device=A.device)
    dinv = M.sum(1).clamp(min=1e-12).pow(-0.5)
    return (dinv.unsqueeze(2) * M * dinv.unsqueeze(1)).transpose(1, 2).contiguous()


class DenseGCN(nn.Module):
    def __init__(self, in_dim=N, hid=HIDDEN):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hid, bias=False); self.b1 = nn.Parameter(torch.zeros(hid))
        self.W2 = nn.Linear(hid, hid, bias=False); self.b2 = nn.Parameter(torch.zeros(hid))
        self.head = nn.Linear(hid, 2)
        nn.init.xavier_uniform_(self.W1.weight); nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, S, X):
        h = self.W1(X)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        h = F.relu(S @ h + self.b1)
        h = F.relu(S @ self.W2(h) + self.b2)
        return self.head(h.mean(1))


S_ALL = gcn_operator(torch.tensor(topk_adjacency(R), device=DEV))
EYE = torch.eye(N, device=DEV)
Y_T = torch.tensor(y, device=DEV)
N_PARAMS = sum(p.numel() for p in DenseGCN().parameters())


@torch.no_grad()
def predict(model, ids, bs=128):
    model.eval(); out = []
    for i in range(0, len(ids), bs):
        b = torch.as_tensor(ids[i:i + bs], device=DEV)
        out.append(F.softmax(model(S_ALL[b], EYE), 1)[:, 1])
    return torch.cat(out).cpu().numpy()


def train_gcn(tr, epochs=EPOCHS, seed=42, model=None, lr=LR, balanced=False):
    torch.manual_seed(seed)
    model = DenseGCN().to(DEV) if model is None else model
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = None
    if balanced:
        cnt = np.bincount(y[tr], minlength=2).astype(float)
        w = torch.tensor(len(tr) / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        model.train(); perm = rng.permutation(tr)
        for i in range(0, len(perm), BATCH):
            b = torch.as_tensor(perm[i:i + BATCH], device=DEV)
            loss = F.cross_entropy(model(S_ALL[b], EYE), Y_T[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def site_split(seed, test_frac=0.2):
    rs = np.random.RandomState(seed); tr, te = {}, {}
    for s in np.unique(sites):
        ids = np.where(sites == s)[0]
        if len(ids) < MIN_SITE:
            continue
        a, c = rs.permutation(ids[y[ids] == 1]), rs.permutation(ids[y[ids] == 0])
        na, nc = max(1, int(round(test_frac * len(a)))), max(1, int(round(test_frac * len(c))))
        te[s], tr[s] = np.r_[a[:na], c[:nc]], np.r_[a[na:], c[nc:]]
    return tr, te


def safe_auc(t, p):
    return float(roc_auc_score(t, p)) if len(set(t)) > 1 else float("nan")


def bacc(t, p, thr=0.5):
    tn, fp, fn, tp = confusion_matrix(t, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))


TR_D, TE_D = site_split(123)
TR_ALL, TE_ALL = np.concatenate(list(TR_D.values())), np.concatenate(list(TE_D.values()))
MODEL_PT = f"{OUT_DIR}/demo_model.pt"
MODEL = DenseGCN().to(DEV)
if os.path.exists(MODEL_PT) and not ARGS.fast:
    MODEL.load_state_dict(torch.load(MODEL_PT, map_location=DEV)); log("model loaded:", MODEL_PT)
else:
    log("training the model once (80% of subjects) ..."); MODEL = train_gcn(TR_ALL, model=MODEL)
    torch.save(MODEL.state_dict(), MODEL_PT)
P_TE = predict(MODEL, TE_ALL)
MODEL_AUC = safe_auc(y[TE_ALL], P_TE)
log(f"model ready · AUC {MODEL_AUC:.3f} on {len(TE_ALL)} held-out subjects")


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers (plotly, dark)
# ─────────────────────────────────────────────────────────────────────────────
def dark(fig, h=420, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      font=dict(family="Inter, Arial, sans-serif", color=INK, size=12),
                      margin=dict(l=10, r=10, t=56 if title else 16, b=10),
                      title=dict(text=title, x=0.02, font=dict(size=16, family="Space Grotesk, Inter, sans-serif")) if title else None)
    return fig


def _net_axes(fig, row=None, col=None):
    if NETS is None:
        fig.update_xaxes(showticklabels=False, row=row, col=col); fig.update_yaxes(showticklabels=False, row=row, col=col)
        return
    s = NETS[NET_ORDER]; b = [0] + [j for j in range(1, N) if s[j] != s[j - 1]] + [N]
    centers = [(b[j] + b[j + 1]) / 2 for j in range(len(b) - 1)]
    fig.update_yaxes(tickvals=centers, ticktext=[s[b[j]][:12] for j in range(len(b) - 1)], autorange="reversed",
                     showgrid=False, tickfont=dict(size=10), row=row, col=col)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
    for x in b[1:-1]:
        for kw in (dict(x0=x - .5, x1=x - .5, y0=-.5, y1=N - .5), dict(y0=x - .5, y1=x - .5, x0=-.5, x1=N - .5)):
            fig.add_shape(type="line", line=dict(color="rgba(255,255,255,0.18)", width=1), row=row, col=col, **kw)


def connectome_fig(M, title, h=430):
    o = NET_ORDER
    fig = go.Figure(go.Heatmap(z=M[np.ix_(o, o)], zmin=-1, zmax=1, colorscale=DIVERGING,
                               colorbar=dict(thickness=10, len=0.85, title=dict(text="r", side="right")),
                               hovertemplate="r = %{z:.2f}<extra></extra>"))
    _net_axes(fig)
    return dark(fig, h, title)


def network_attr_fig(attr):
    names = [x for x in NET_NAMES if x != "Unassigned"]
    Mn = np.array([[attr[np.ix_(NETS == a, NETS == b)].sum() for b in names] for a in names])
    v = float(np.abs(Mn).max()) + 1e-12
    fig = go.Figure(go.Heatmap(z=Mn, x=names, y=names, zmin=-v, zmax=v, colorscale=DIVERGING,
                               colorbar=dict(thickness=10, len=0.85),
                               hovertemplate="%{y} ↔ %{x}<br>contribution %{z:+.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(tickangle=-40)
    return dark(fig, 430, "Which networks drove it  ·  violet → autism · teal → control")


def brain3d_fig(attr, k=30, title="The 30 most influential connections"):
    fig = go.Figure()
    if COORDS is None:
        return None
    cols = [NET_COLOR.get(x, "#6B6485") for x in NETS] if NETS is not None else ["#8B85A8"] * N
    fig.add_trace(go.Mesh3d(x=COORDS[:, 0] * 1.08, y=COORDS[:, 1] * 1.08, z=COORDS[:, 2] * 1.08, alphahull=0,
                            color="#7C3AED", opacity=0.07, hoverinfo="skip", showscale=False))   # glass-brain hull
    fig.add_trace(go.Scatter3d(x=COORDS[:, 0], y=COORDS[:, 1], z=COORDS[:, 2], mode="markers",
                               marker=dict(size=3.2, color=cols, opacity=0.55), hoverinfo="text",
                               text=[f"ROI {i} · {NETS[i] if NETS is not None else ''}" for i in range(N)], showlegend=False))
    au = attr[IU]; top = np.argsort(-np.abs(au))[:k]
    for sign, color, name in ((1, VIOLET, "toward autism"), (-1, TEAL, "toward control")):
        xs, ys, zs = [], [], []
        for t in top:
            if np.sign(au[t]) != sign:
                continue
            a, b = IU[0][t], IU[1][t]
            xs += [COORDS[a, 0], COORDS[b, 0], None]; ys += [COORDS[a, 1], COORDS[b, 1], None]; zs += [COORDS[a, 2], COORDS[b, 2], None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=6), name=name,
                                       hoverinfo="skip"))
    hub = np.unique(np.r_[IU[0][top], IU[1][top]])
    fig.add_trace(go.Scatter3d(x=COORDS[hub, 0], y=COORDS[hub, 1], z=COORDS[hub, 2], mode="markers",
                               marker=dict(size=7, color=[cols[i] for i in hub], line=dict(color="#FFFFFF", width=1)),
                               hoverinfo="text", text=[f"ROI {i} · {NETS[i] if NETS is not None else ''}" for i in hub],
                               showlegend=False))
    ax = dict(visible=False, showbackground=False)
    fig.update_layout(scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", bgcolor="rgba(0,0,0,0)",
                                 camera=dict(eye=dict(x=1.45, y=0.55, z=0.55))),
                      legend=dict(orientation="h", y=0.02, x=0.02, bgcolor="rgba(0,0,0,0)"))
    return dark(fig, 520, title + "  ·  drag to rotate")


def sparkline_fig(ts):
    k = min(6, ts.shape[1]); idx = np.linspace(0, ts.shape[1] - 1, k).astype(int)
    fig = go.Figure()
    for j, r in enumerate(idx):
        s = ts[:, r]; s = (s - s.mean()) / (s.std() + 1e-8)
        fig.add_trace(go.Scatter(y=s + 4 * j, mode="lines", line=dict(width=1.6, color=NET_PALETTE[j % len(NET_PALETTE)]),
                                 name=f"region {r}", hoverinfo="skip"))
    fig.update_xaxes(title="time points", showgrid=False); fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False)
    return dark(fig, 230, f"Raw signal · 6 of 200 regions · {ts.shape[0]} time points")


# ─────────────────────────────────────────────────────────────────────────────
# UI pieces (HTML)
# ─────────────────────────────────────────────────────────────────────────────
def caption(tag, text, sub="", on=True, kind="in", final=False):
    """Caption bar pinned to the bottom of the screen. The final caption of a run fades out by itself."""
    mark = '<span class="run-done" hidden></span>' if final else ""
    if not on:
        return mark
    return (f'<div class="cap-fixed fade-in{" linger" if final else ""}"><span class="tag {kind}">{tag}</span>{text}'
            f'{f"<div class=sub>{sub}</div>" if sub else ""}</div>{mark}')


def show(x):
    """Reveal an output only once it has content (no empty placeholders)."""
    return gr.update(value=x, visible=x is not None)


def read_ts(path):
    try:
        ts = np.loadtxt(path)
    except ValueError:
        ts = np.loadtxt(path, delimiter=",")
    return np.atleast_2d(ts)


def file_head(path, rows=3, cols=5):
    try:
        out = []
        with open(path) as f:
            for line in f:
                if line.strip() and not line.lstrip().startswith("#"):
                    v = line.replace(",", " ").split()
                    out.append("  ".join(f"{float(x):+.3f}" for x in v[:cols]) + f"  … ({len(v)} values)")
                if len(out) == rows:
                    break
        return "<br>".join(out)
    except Exception:
        return ""


ANALYZE_STEPS = [("Read the scan", "time series of 200 brain regions"),
                 ("Functional connectivity", "Pearson correlation between every pair of regions"),
                 ("Brain graph", "keep each region’s 20 strongest connections"),
                 ("Graph neural network", "2-layer GCN → probability of autism"),
                 ("Explanation", "gradient × input on every connection")]


def steps_html(steps, state, detail):
    rows = []
    for i, (title, sub) in enumerate(steps):
        st = state[i]
        dot = "✓" if st == "done" else ("" if st == "run" else str(i + 1))
        d = detail.get(i, sub)
        rows.append(f'<div class="step {st}"><div class="dot">{dot}</div><div><div class="st-title">{title}</div>'
                    f'<div class="st-sub">{d}</div></div></div>')
    complete = " complete" if all(s == "done" for s in state) else ""
    return f'<div class="steps glass{complete}">{"".join(rows)}</div>'


def gauge_wait(text="Awaiting a scan"):
    return (f'<div class="glass gauge empty"><svg viewBox="0 0 180 180" width="180" height="180">'
            f'<circle cx="90" cy="90" r="70" stroke="rgba(255,255,255,0.06)" stroke-width="14" fill="none"/>'
            f'<circle class="orbit" cx="90" cy="90" r="70" stroke="url(#gw)" stroke-width="14" fill="none" stroke-linecap="round" stroke-dasharray="90 350"/>'
            f'<defs><linearGradient id="gw" x1="0" x2="1"><stop offset="0" stop-color="#2DD4BF"/><stop offset="1" stop-color="#A78BFA"/></linearGradient></defs>'
            f'<text x="90" y="96" text-anchor="middle" class="g-lbl">P(autism)</text></svg><div class="note">{text}</div></div>')


def scan_card(name, size, state, info="", head=""):
    bar = '<div class="bar"><i></i></div>' if state == "load" else ""
    icon = "⬆" if state == "load" else "✓"
    sub = f"{size / 1e3:.0f} KB · receiving and parsing…" if state == "load" else info
    pre = f'<div class="head">{head}</div>' if head else ""
    return (f'<div class="glass scan fade-in {state}"><div class="row"><b>{icon} {name}</b><span>{sub}</span></div>'
            f'{bar}{pre}</div>')


def gauge_html(prob, truth=None, note=""):
    C = 2 * np.pi * 70
    off = C * (1 - prob)
    color = VIOLET if prob >= .5 else TEAL
    verdict = "Autism-like connectivity pattern" if prob >= .5 else "Typical connectivity pattern"
    t = ""
    if truth is not None:
        ok = (prob >= .5) == (truth == "autism")
        t = (f'<div class="truth">True diagnosis: <b>{truth}</b> · '
             f'<span class="{"ok" if ok else "ko"}">{"model correct" if ok else "model wrong"}</span></div>')
    return f"""<div class="glass gauge fade-in">
<svg viewBox="0 0 180 180" width="180" height="180">
 <defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="#2DD4BF"/><stop offset="1" stop-color="#A78BFA"/></linearGradient></defs>
 <circle cx="90" cy="90" r="70" stroke="rgba(255,255,255,0.08)" stroke-width="14" fill="none"/>
 <circle class="ring-anim" cx="90" cy="90" r="70" stroke="url(#gg)" stroke-width="14" fill="none" stroke-linecap="round"
   stroke-dasharray="{C:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 90 90)" style="--c:{C:.1f}"/>
 <text x="90" y="88" text-anchor="middle" class="g-num">{prob * 100:.0f}%</text>
 <text x="90" y="112" text-anchor="middle" class="g-lbl">P(autism)</text>
</svg>
<div class="verdict" style="color:{color}">{verdict}</div>{t}
<div class="note">{note}</div></div>"""


def stat_cards(items):
    return '<div class="cards fade-in">' + "".join(
        f'<div class="glass card"><div class="big" style="color:{c}">{v}</div><div class="lbl">{l}</div></div>' for v, l, c in items) + "</div>"


EMPTY_GAUGE = gauge_wait("Awaiting a scan")


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Analyze a scan  (INPUT → OUTPUT)
# ─────────────────────────────────────────────────────────────────────────────
def subject_choices():
    order = np.argsort(sites[TE_ALL], kind="stable")
    return [(f"Subject {sids[i]} · {sites[i]}", int(i)) for i in TE_ALL[order]]


def explain(Rm):
    A = torch.tensor(topk_adjacency(Rm[None]), device=DEV, requires_grad=True)
    MODEL.eval()
    out = MODEL(gcn_operator(A), EYE)
    prob = F.softmax(out, 1)[0, 1].item()
    out[0, 1].sub(out[0, 0]).backward()
    attr = (A.grad * A).detach()[0].cpu().numpy()
    return prob, attr + attr.T, int((A.detach()[0] > 0).sum().item())


def analyze(choice, upload, captions):
    state = ["todo"] * 5; det = {}

    def outs(cap, scan="", spark=None, gauge=EMPTY_GAUGE, conn=None, attrf=None, brain=None, table=None):
        return (cap, steps_html(ANALYZE_STEPS, state, det), scan, show(spark), gauge, show(conn), show(attrf),
                show(brain), show(table))

    # ---- INPUT
    truth = None
    if upload is not None:
        path = upload if isinstance(upload, str) else upload.name
        name, size = os.path.basename(path), os.path.getsize(path)
        label = f"{name} · uploaded by the user"
    elif choice is not None:
        i = int(choice); sid = int(sids[i]); path = FILE_OF.get(sid)
        truth = "autism" if y[i] else "control"; label = f"Subject {sid} · {sites[i]} · never seen in training"
        name, size = (os.path.basename(path), os.path.getsize(path)) if path else (f"subject_{sid}", 0)
    else:
        raise gr.Error("Upload a .1D scan or pick a subject.")
    state[0] = "run"
    yield outs(caption("INPUT", "Receiving the patient’s scan", label, captions),
               scan_card(name, size, "load", head=file_head(path) if path else ""), gauge=gauge_wait("Reading the scan…"))
    t0 = time.time()
    ts = None
    if path:
        try:
            ts = read_ts(path)
        except Exception as e:
            raise gr.Error(f"Could not read the file: {e}")
        if ts.shape[1] != N and ts.shape[0] == N:
            ts = ts.T
        if ts.shape[1] != N or ts.shape[0] < 30:
            raise gr.Error(f"Expected a time series of shape (time points ≥ 30, 200 regions); got {ts.shape}.")
    dt = time.time() - t0          # real compute time (the short pauses below only pace the animation)
    time.sleep(1.0)
    if ts is not None:
        flat = int((ts.std(0) < 1e-8).sum())
        info = (f"{ts.shape[0]} time points × {ts.shape[1]} regions · {size / 1e3:.0f} KB · "
                f"{'all regions have signal' if flat == 0 else f'{flat} flat regions'}")
        det[0] = f"{ts.shape[0]} time points × {ts.shape[1]} regions · read in {dt * 1000:.0f} ms"
        scan, spark = scan_card(name, size, "ok", info, file_head(path)), sparkline_fig(ts)
    else:
        det[0] = "precomputed connectivity (raw file not on this machine)"
        scan, spark = scan_card(name, size, "ok", "precomputed connectivity"), None
    state[0], state[1] = "done", "run"
    yield outs(caption("STEP 2", "Computing functional connectivity", "Pearson correlation between every pair of the 200 regions", captions),
               scan, spark, gauge_wait("Computing connectivity…"))
    t1 = time.time()
    Rm = correlation(ts) if ts is not None else R[int(choice)]
    dt = time.time() - t1
    time.sleep(0.6)
    det[1] = f"19,900 connections · {dt * 1000:.0f} ms"
    conn = connectome_fig(Rm, "Functional connectome · 200 regions sorted by network")
    state[1], state[2] = "done", "run"
    yield outs(caption("STEP 3", "Building the brain graph", "each region keeps its 20 strongest connections", captions),
               scan, spark, gauge_wait("Building the graph…"), conn)
    t2 = time.time()
    prob, attr, n_edges = explain(Rm)
    dt = time.time() - t2
    time.sleep(0.6)
    det[2] = f"200 nodes · {n_edges:,} directed edges · graph + GCN + gradients in {dt * 1000:.0f} ms"
    state[2], state[3] = "done", "run"
    yield outs(caption("STEP 4", "Running the graph neural network", "2-layer GCN trained on 80% of ABIDE", captions),
               scan, spark, gauge_wait("Graph neural network…"), conn)
    time.sleep(0.9)
    det[3] = f"P(autism) = {prob:.2f}"
    gauge = gauge_html(prob, truth, f"Model AUC on {len(TE_ALL)} held-out subjects: {MODEL_AUC:.3f} · research prototype, not a diagnosis")
    state[3], state[4] = "done", "run"
    yield outs(caption("OUTPUT", f"Probability of autism: {prob * 100:.0f}%", "now explaining which connections drove it", captions, "out"),
               scan, spark, gauge, conn)
    time.sleep(0.8)
    au = attr[IU]; top = np.argsort(-np.abs(au))[:10]
    table = pd.DataFrame([[f"ROI {IU[0][t]} ↔ ROI {IU[1][t]}",
                           f"{NETS[IU[0][t]]} ↔ {NETS[IU[1][t]]}" if NETS is not None else "",
                           round(float(Rm[IU[0][t], IU[1][t]]), 2), "autism" if au[t] > 0 else "control", f"{au[t]:+.4f}"]
                          for t in top], columns=["Connection", "Networks", "r", "Pushes toward", "Contribution"])
    attrf = network_attr_fig(attr) if NETS is not None else None
    brain = brain3d_fig(attr)
    det[4] = "top connections and networks identified"
    state[4] = "done"
    yield outs(caption("OUTPUT", "The brain networks behind the decision", "violet pushes toward autism · teal toward control", captions,
                       "out", final=True), scan, spark, gauge, conn, attrf, brain, table)


def pick_random():
    return gr.update(value=int(np.random.choice(TE_ALL))), None


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Federated training across hospitals
# ─────────────────────────────────────────────────────────────────────────────
FED_TR, FED_TE = site_split(0)
FED_TE_ALL, FED_TR_ALL = np.concatenate(list(FED_TE.values())), np.concatenate(list(FED_TR.values()))
FED_SITES = sorted(FED_TR, key=lambda s: -(len(FED_TR[s]) + len(FED_TE[s])))


def hospital_net(round_i=None, total=None, active=True, auc=None, final=False):
    W, H, cx, cy = 680, 400, 340, 195
    k = len(FED_SITES); parts = []
    for j, s in enumerate(FED_SITES):
        ang = 2 * np.pi * j / k - np.pi / 2
        x, yy = cx + 272 * np.cos(ang), cy + 150 * np.sin(ang)
        r = 6 + 10 * np.sqrt((len(FED_TR[s]) + len(FED_TE[s])) / 172)
        cls = "flow" if active else "flow idle"
        parts.append(f'<line x1="{x:.0f}" y1="{yy:.0f}" x2="{cx}" y2="{cy}" class="{cls}" style="animation-delay:{(j % 6) * 0.15:.2f}s"/>')
        if active:   # weight packets: site → server (violet), then global model back (teal)
            d = (j * 0.09) % 1.2
            parts.append(f'<circle r="3.4" class="pkt up"><animateMotion dur="1.5s" begin="{d:.2f}s" repeatCount="indefinite" '
                         f'path="M{x:.0f},{yy:.0f} L{cx},{cy}"/></circle>')
            parts.append(f'<circle r="2.6" class="pkt down"><animateMotion dur="1.5s" begin="{d + 0.75:.2f}s" repeatCount="indefinite" '
                         f'path="M{cx},{cy} L{x:.0f},{yy:.0f}"/></circle>')
        parts.append(f'<circle cx="{x:.0f}" cy="{yy:.0f}" r="{r:.1f}" class="site{" busy" if active else ""}" '
                     f'style="animation-delay:{(j % 5) * 0.2:.1f}s"/>')
        parts.append(f'<text x="{x:.0f}" y="{yy + r + 13:.0f}" class="site-lbl" text-anchor="middle">{s[:9]}</text>')
    status = (f"Round {round_i}/{total}" if round_i else "Ready") + (f" · global AUC {auc:.3f}" if auc is not None else "")
    return f"""<div class="glass fed-net{' complete' if final else ''}"><svg viewBox="0 0 {W} {H}" width="100%">
{''.join(parts)}
<circle cx="{cx}" cy="{cy}" r="34" class="server {'pulse' if active else ''}"/>
<text x="{cx}" y="{cy - 2}" text-anchor="middle" class="srv-lbl">SERVER</text>
<text x="{cx}" y="{cy + 14}" text-anchor="middle" class="srv-sub">global model</text>
</svg><div class="fed-status">{status} · each site sends {N_PARAMS * 4 / 1024:.0f} KB of weights · scans never leave the site</div></div>"""


def fed_curves(hist, ref=None):
    r = list(range(1, len(hist["auc"]) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Global model on held-out subjects", "Subjects called autistic"))
    fig.add_trace(go.Scatter(x=r, y=hist["auc"], mode="lines+markers", name="AUC", line=dict(color=VIOLET, width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=r, y=hist["bacc"], mode="lines+markers", name="balanced accuracy", line=dict(color=TEAL, width=2)), 1, 1)
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=PINK, dash="dot"), annotation_text=f"centralized {ref:.3f}", row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.3)", dash="dash"), row=1, col=1)
    fig.add_trace(go.Bar(x=r, y=hist["pos"], name="predicted autistic", marker_color="#7C3AED"), 1, 2)
    fig.add_hline(y=int(y[FED_TE_ALL].sum()), line=dict(color=TEAL, dash="dash"), annotation_text="true number", row=1, col=2)
    fig.update_xaxes(title="round", dtick=max(1, len(r) // 10))
    fig.update_yaxes(range=[0.35, 1.0], row=1, col=1)
    return dark(fig, 360)


def run_federated(rounds, local_epochs, balanced, compare, captions):
    rounds, local_epochs = int(rounds), int(local_epochs)
    tr, te = FED_TR, FED_TE
    torch.manual_seed(0); gm = DenseGCN().to(DEV)
    hist = {"auc": [], "bacc": [], "pos": []}
    yield (caption("PRIVACY", f"Federated training across {len(tr)} hospitals",
                   "each site trains on its own scans; only model weights travel", captions),
           hospital_net(0, rounds, True), "Starting round 1…", show(None), show(None))
    t0 = time.time()
    for r in range(rounds):
        states, ws = [], []
        for k, (s, ids) in enumerate(tr.items()):
            local = train_gcn(ids, epochs=local_epochs, seed=1000 * r + k, model=copy.deepcopy(gm), balanced=balanced)
            states.append(local.state_dict()); ws.append(len(ids))
        gm.load_state_dict({key: sum(w * st[key] for w, st in zip(ws, states)) / sum(ws) for key in states[0]})
        p = predict(gm, FED_TE_ALL)
        hist["auc"].append(safe_auc(y[FED_TE_ALL], p)); hist["bacc"].append(bacc(y[FED_TE_ALL], p)); hist["pos"].append(int((p >= .5).sum()))
        yield (caption("LIVE", f"Round {r + 1}/{rounds} · {len(tr)} weight updates averaged",
                       f"global AUC {hist['auc'][-1]:.3f} · {hist['pos'][-1]} of {len(FED_TE_ALL)} held-out subjects called autistic", captions),
               hospital_net(r + 1, rounds, True, hist["auc"][-1]),
               f"**Round {r + 1}/{rounds}** · {time.time() - t0:.0f}s", show(fed_curves(hist)), show(None))
    results = {"FedAvg (federated)": {s: predict(gm, te[s]) for s in te}}
    if "Local only" in compare:
        yield (caption("COMPARE", "Each hospital alone", "same split, no collaboration", captions),
               hospital_net(rounds, rounds, False, hist["auc"][-1]), "Training one model per site…", show(fed_curves(hist)), show(None))
        results["Each site alone"] = {s: predict(train_gcn(tr[s], balanced=balanced), te[s]) for s in tr}
    ref = None
    if "Centralized" in compare:
        yield (caption("COMPARE", "All data pooled (not possible in real hospitals)", "the upper bound", captions),
               hospital_net(rounds, rounds, False, hist["auc"][-1]), "Training on pooled data…", show(fed_curves(hist)), show(None))
        m = train_gcn(FED_TR_ALL, balanced=balanced)
        results["Centralized (pooled)"] = {s: predict(m, te[s]) for s in te}
        ref = safe_auc(y[FED_TE_ALL], predict(m, FED_TE_ALL))
    names, aucs = [], []
    for name, d in results.items():
        pp = np.zeros(n)
        for s in te:
            pp[te[s]] = d[s]
        names.append(name); aucs.append(safe_auc(y[FED_TE_ALL], pp[FED_TE_ALL]))
    cmp = go.Figure(go.Bar(x=names, y=aucs, marker_color=[VIOLET, "#6B6485", TEAL][:len(names)],
                           text=[f"{a:.3f}" for a in aucs], textposition="outside"))
    cmp.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
    lo = min([0.4] + [a - 0.08 for a in aucs if a == a])
    cmp.update_yaxes(range=[lo, 1.0], title="AUC on held-out subjects")
    yield (caption("OUTPUT", "One shared model, trained without moving any scan",
                   " · ".join(f"{a}: {b:.3f}" for a, b in zip(names, aucs)), captions, "out", final=True),
           hospital_net(rounds, rounds, False, hist["auc"][-1], final=True),
           f"**Done in {time.time() - t0:.0f}s.** " + " · ".join(f"{a}: AUC {b:.3f}" for a, b in zip(names, aucs)),
           show(fed_curves(hist, ref)), show(dark(cmp, 360, "Collaboration strategies, same split · AUC on held-out subjects")))


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Connectivity evolution (within-scan proxy)
# ─────────────────────────────────────────────────────────────────────────────
TRK = {"V": None}


def windows_of(ts, frac=0.667):
    k = int(round(frac * len(ts))); out = []
    for w in (ts[:k], ts[k:]):
        w = (w - w.mean(0)) / (w.std(0) + 1e-8)
        out.append(np.arctanh(np.clip(np.nan_to_num(np.corrcoef(w.T)), -.999, .999))[IU].astype(np.float32))
    return np.stack(out)


def load_windows(progress=None):
    if TRK["V"] is not None:
        return True
    src = next((p for p in WINDOW_FILES if os.path.exists(p)), None)
    if src is None:
        files = sorted(glob.glob(os.path.join(RAW_DIR, "*.1D")))
        if not files:
            return False
        V, ss, ids = [], [], []
        it = progress.tqdm(files, desc="first use: cutting every scan into windows") if progress else files
        for f in it:
            ts = np.loadtxt(f)
            if ts.ndim != 2 or ts.shape[1] != N or np.any(ts.std(0) < 1e-8) or len(ts) < 90:
                continue
            V.append(windows_of(ts))
            m = re.match(r"^(.*?)_(\d{5,})", os.path.basename(f)); ss.append(m.group(1) if m else "UNK"); ids.append(m.group(2) if m else f)
        src = f"{OUT_DIR}/cache_windows.npz"
        np.savez_compressed(src, V=np.stack(V), sites=np.array(ss), ids=np.array(ids))
    d = np.load(src, allow_pickle=True)
    V, s = d["V"].astype(np.float32), np.array(d["sites"]).astype(str)
    test_sites = set(np.unique(s)[::5])
    te, tr = np.where(np.isin(s, list(test_sites)))[0], np.where(~np.isin(s, list(test_sites)))[0]
    M0, M1 = V[tr, 0].mean(0), V[tr, 1].mean(0)
    a = float(((V[tr, 0] - M0) * (V[tr, 1] - M1)).sum() / ((V[tr, 0] - M0) ** 2).sum())
    TRK.update(V=V, sites=s, ids=np.array(d["ids"]).astype(str), tr=tr, te=te, M0=M0, M1=M1, a=a)
    return True


def evo_choices():
    if not load_windows():
        return gr.update(choices=[])
    return gr.update(choices=[(f"Subject {TRK['ids'][i]} · {TRK['sites'][i]}", int(i)) for i in TRK["te"][:150]])


def evo_random():
    if not load_windows():
        raise gr.Error("No window data and no raw scans found under --root.")
    return gr.update(choices=[(f"Subject {TRK['ids'][i]} · {TRK['sites'][i]}", int(i)) for i in TRK["te"][:150]],
                     value=int(np.random.choice(TRK["te"][:150]))), None


EVO_STEPS = [("Read the scan", "time series of 200 regions"), ("Split in time", "first 2/3 = input · last 1/3 = hidden target"),
             ("Connectivity of the input", "Pearson → Fisher z, 19,900 connections"),
             ("Predict the future connectome", "shrinkage toward the population trajectory"),
             ("Compare with reality", "error vs simple baselines")]


def vec2mat(v):
    M = np.zeros((N, N), np.float32); M[IU] = v; return M + M.T


def evo_fig(panels):
    k = len(panels)
    fig = make_subplots(rows=1, cols=k, horizontal_spacing=0.035, subplot_titles=[p[0] for p in panels])
    for j, (_, M, cs, zmin, zmax) in enumerate(panels, 1):
        oo = NET_ORDER
        fig.add_trace(go.Heatmap(z=M[np.ix_(oo, oo)], colorscale=cs, zmin=zmin, zmax=zmax, showscale=(cs is SEQ),
                                 colorbar=dict(thickness=8, len=0.8), hoverinfo="skip"), 1, j)
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=j)
        fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed", row=1, col=j)
    fig.update_annotations(font=dict(size=13, color=INK))
    fig = dark(fig, 400 if k > 1 else 430)
    fig.update_layout(margin=dict(l=10, r=10, t=44, b=10))
    return fig


def evolution(choice, upload, captions, progress=gr.Progress()):
    state = ["todo"] * 5; det = {}
    o = lambda cap, scan="", fig=None, cards="": (cap, steps_html(EVO_STEPS, state, det), scan, show(fig), cards)
    if not load_windows(progress):
        raise gr.Error("No window data and no raw scans found under --root.")
    state[0] = "run"
    if upload is not None:
        path = upload if isinstance(upload, str) else upload.name
        name, size = os.path.basename(path), os.path.getsize(path)
        yield o(caption("INPUT", "Receiving the scan", "we will hide its last third and try to predict it", captions),
                scan_card(name, size, "load", head=file_head(path)))
        ts = read_ts(path)
        if ts.shape[1] != N and ts.shape[0] == N:
            ts = ts.T
        if ts.shape[1] != N or len(ts) < 90:
            raise gr.Error(f"Expected (time points ≥ 90, 200 regions); got {ts.shape}.")
        W = windows_of(ts); who = name
        det[0] = f"{name} · {len(ts)} time points"
        scan = scan_card(name, size, "ok", f"{len(ts)} time points × 200 regions · {size / 1e3:.0f} KB", file_head(path))
    elif choice is not None:
        i = int(choice); W = TRK["V"][i]; who = f"Subject {TRK['ids'][i]} · {TRK['sites'][i]} (site never seen in training)"
        yield o(caption("INPUT", "Reading the scan", "we will hide its last third and try to predict it", captions))
        det[0] = who
        scan = scan_card(f"Subject {TRK['ids'][i]}", 0, "ok", f"{TRK['sites'][i]} · site never seen in training")
    else:
        raise gr.Error("Upload a .1D scan or pick a subject.")
    time.sleep(0.8); state[0], state[1] = "done", "run"
    yield o(caption("STEP 2", "Splitting the scan in time", "first two thirds → input · last third → hidden target", captions), scan)
    time.sleep(0.7); det[1] = "input = first 2/3 · target = last 1/3 (hidden)"; state[1], state[2] = "done", "run"
    yield o(caption("STEP 3", "Connectivity of the input window", "19,900 connections", captions), scan)
    x0, x1 = W[0], W[1]
    time.sleep(0.6); state[2], state[3] = "done", "run"
    f_in = evo_fig([("INPUT · first 2/3 of the scan", vec2mat(np.tanh(x0)), DIVERGING, -1, 1)])
    yield o(caption("STEP 4", "Predicting the connectome of the last third", "model fitted on subjects from other sites", captions),
            scan, f_in)
    pred = TRK["M1"] + TRK["a"] * (x0 - TRK["M0"])
    time.sleep(1.0); state[3], state[4] = "done", "run"
    e_pred, e_pop, e_copy = (float(np.abs(p - x1).mean()) for p in (pred, TRK["M1"], x0))
    gain = 100 * (1 - e_pred / e_pop)
    P, T = vec2mat(np.tanh(pred)), vec2mat(np.tanh(x1))
    fig = evo_fig([("INPUT · first 2/3", vec2mat(np.tanh(x0)), DIVERGING, -1, 1), ("PREDICTED · last 1/3", P, DIVERGING, -1, 1),
                   ("TRUE · last 1/3", T, DIVERGING, -1, 1), ("|predicted − true|", np.abs(P - T), SEQ, 0, 0.6)])
    det[4] = f"error {e_pred:.3f} vs {e_pop:.3f} (population mean) and {e_copy:.3f} (copy input)"
    state[4] = "done"
    cards = stat_cards([(f"{gain:+.1f}%", "error reduction vs the population-mean graph", VIOLET),
                        (f"{e_pred:.3f}", "prediction error (MAE, Fisher z)", TEAL),
                        (f"{e_copy:.3f}", "error if we just copy the input", MUTED)])
    yield o(caption("OUTPUT", "Predicted vs true connectome of the last third", f"error {gain:+.1f}% vs the population mean · {who}",
                    captions, "out", final=True), scan, fig, cards)


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Model card
# ─────────────────────────────────────────────────────────────────────────────
LOGGED = pd.DataFrame([
    ["Logistic regression, raw edges", "0.680 ± 0.032", "0.736 ± 0.026", "5-fold CV"],
    ["GCN, raw graphs (this platform's model)", "0.650 ± 0.019", "0.694 ± 0.020", "5-fold CV"],
    ["GCN + knowledge graph (6 relations)", "0.643 ± 0.023", "0.701 ± 0.029", "5-fold CV"],
    ["GCN, harmonized graphs", "0.552 ± 0.039", "0.565 ± 0.039", "5-fold × 5 seeds"],
    ["FedAvg KG-GCN, 18 sites", "0.547", "0.595", "hold-out, 150 subjects"],
    ["Majority class", "0.537", "0.500", "—"],
], columns=["Model", "Accuracy", "AUC", "Protocol"])

PIPE_HTML = f"""<div class="pipe fade-in">
<div class="glass pstep"><span class="pk">INPUT</span><b>rs-fMRI scan</b><small>ABIDE I · CPAC · cc200 (200 regions)</small></div><div class="parrow">→</div>
<div class="glass pstep"><span class="pk">1</span><b>Connectome</b><small>Pearson correlation · 19,900 edges</small></div><div class="parrow">→</div>
<div class="glass pstep"><span class="pk">2</span><b>Brain graph</b><small>20 strongest edges per region</small></div><div class="parrow">→</div>
<div class="glass pstep"><span class="pk">3</span><b>2-layer GCN</b><small>64 units · mean pooling</small></div><div class="parrow">→</div>
<div class="glass pstep out"><span class="pk">OUTPUT</span><b>P(autism) + why</b><small>network attribution · 3-D view</small></div>
</div>"""

HERO = f"""<div id="hero"><canvas id="neuro-canvas"></canvas><div class="hero-inner">
<div class="eyebrow">Brain connectivity · graph neural networks · federated learning</div>
<div class="hero-title">FedKG-Brain</div>
<div class="hero-sub">Upload a resting-state fMRI scan. Get the probability of autism, the brain networks behind it,
and a 3-D view of the connections that mattered.</div>
<div class="chips"><span>{n} subjects</span><span>{len(np.unique(sites))} sites</span><span>200 brain regions</span>
<span>model AUC {MODEL_AUC:.3f} (held-out)</span></div></div></div>"""

CSS = """
@keyframes shine{to{background-position:300% 0}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes ring{from{stroke-dashoffset:var(--c)}}
@keyframes flow{to{stroke-dashoffset:-36}}
@keyframes pulse{0%{filter:drop-shadow(0 0 4px #A78BFA)}50%{filter:drop-shadow(0 0 18px #A78BFA)}100%{filter:drop-shadow(0 0 4px #A78BFA)}}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(167,139,250,.0)}50%{box-shadow:0 0 22px 2px rgba(167,139,250,.45)}}
body,.gradio-container{background:radial-gradient(1100px 600px at 0% -10%,rgba(124,58,237,.28),transparent 60%),
  radial-gradient(900px 520px at 110% 0%,rgba(20,184,166,.20),transparent 60%),#07060F !important}
.gradio-container{max-width:1320px !important;margin:auto !important}
#hero{position:relative;border-radius:26px;overflow:hidden;border:1px solid rgba(167,139,250,.28);
  background:linear-gradient(135deg,rgba(91,45,144,.45),rgba(7,6,15,.6) 55%,rgba(15,139,141,.25));min-height:250px}
#neuro-canvas{position:absolute;inset:0;width:100%;height:100%}
.hero-inner{position:relative;z-index:1;padding:34px 40px}
.eyebrow{color:#5EEAD4;letter-spacing:4px;text-transform:uppercase;font-size:12px;font-weight:600}
.hero-title{font-family:'Space Grotesk',Inter,sans-serif;font-size:64px;font-weight:700;line-height:1.05;margin:6px 0;
  background:linear-gradient(90deg,#E9D5FF,#A78BFA,#2DD4BF,#E9D5FF);background-size:300% 100%;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 9s linear infinite}
.hero-sub{color:#CFC8E8;font-size:18px;max-width:760px;line-height:1.5}
.chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.chips span{border:1px solid rgba(167,139,250,.35);background:rgba(167,139,250,.10);color:#E9E6F5;border-radius:999px;padding:5px 14px;font-size:13px}
.glass{background:rgba(255,255,255,.035);border:1px solid rgba(167,139,250,.18);border-radius:18px;padding:16px;backdrop-filter:blur(10px)}
.sec{font-family:'Space Grotesk',Inter,sans-serif;font-size:13px;letter-spacing:3px;text-transform:uppercase;color:#5EEAD4;margin:6px 2px}
.steps{display:flex;flex-direction:column;gap:8px}
.step{display:flex;gap:14px;align-items:center;padding:10px 12px;border-radius:14px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);transition:all .3s}
.step .dot{width:30px;height:30px;min-width:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px}
.step.todo{opacity:.55}.step.todo .dot{border:2px solid #3B3555;color:#8B85A8}
.step.run{border-color:#A78BFA;animation:glow 1.4s ease-in-out infinite}
.step.run .dot{border:3px solid #A78BFA;border-top-color:transparent;animation:spin .8s linear infinite}
.step.done{border-color:rgba(45,212,191,.45)}.step.done .dot{background:linear-gradient(135deg,#2DD4BF,#14B8A6);color:#07060F}
.st-title{font-weight:600;color:#F3F0FF}.st-sub{font-size:13px;color:#9A93B8}
.scan{display:flex;justify-content:space-between;align-items:center;gap:12px;border-color:rgba(45,212,191,.35)}
.scan b{color:#E9E6F5}.scan span{color:#9A93B8;font-size:13px}
.gauge{display:flex;flex-direction:column;align-items:center;gap:6px;min-height:330px;justify-content:center}
.gauge.empty{color:#6E6790}
.ring-anim{animation:ring 1.6s cubic-bezier(.2,.8,.2,1) both}
.g-num{fill:#F3F0FF;font:700 36px 'Space Grotesk',Inter,sans-serif}.g-lbl{fill:#9A93B8;font:500 12px Inter,sans-serif}
.verdict{font:700 20px 'Space Grotesk',Inter,sans-serif;text-align:center}
.truth{color:#CFC8E8;font-size:14px}.truth .ok{color:#2DD4BF;font-weight:700}.truth .ko{color:#F87171;font-weight:700}
.note{color:#8B85A8;font-size:12px;text-align:center;max-width:320px}
.cap-fixed{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:9999;max-width:82vw;text-align:center;
  background:rgba(14,10,30,.92);border:1px solid rgba(167,139,250,.35);border-left:6px solid #2DD4BF;border-radius:16px;
  padding:12px 26px;color:#FFFFFF;font:600 21px Inter,sans-serif;box-shadow:0 14px 40px rgba(0,0,0,.45)}
.cap-fixed .sub{font-weight:400;font-size:15px;color:#CFC8E8;margin-top:4px}
.cap-fixed .tag{display:inline-block;margin-right:12px;padding:2px 10px;border-radius:8px;font-size:14px;letter-spacing:1px;background:#0D9488}
.cap-fixed .tag.out{background:#7C3AED}
.fade-in{animation:fade .6s ease both}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.card .big{font:700 34px 'Space Grotesk',Inter,sans-serif}.card .lbl{color:#9A93B8;font-size:13px;margin-top:4px}
.fed-net svg{display:block}.fed-status{color:#CFC8E8;font-size:13px;text-align:center;margin-top:6px}
.flow{stroke:#A78BFA;stroke-width:1.6;stroke-dasharray:4 8;opacity:.75;animation:flow 1.1s linear infinite}
.flow.idle{animation:none;stroke:#2DD4BF;opacity:.35;stroke-dasharray:none}
.site{fill:#14B8A6;stroke:#5EEAD4;stroke-width:1.5}.site-lbl{fill:#9A93B8;font:500 10px Inter,sans-serif}
.server{fill:#5B21B6;stroke:#C4B5FD;stroke-width:2}.server.pulse{animation:pulse 1.4s ease-in-out infinite}
.srv-lbl{fill:#FFFFFF;font:700 11px 'Space Grotesk',Inter,sans-serif}.srv-sub{fill:#DDD6FE;font:500 9px Inter,sans-serif}
.pipe{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap}.pstep{flex:1;min-width:150px;display:flex;flex-direction:column;gap:4px}
.pstep b{color:#F3F0FF;font-family:'Space Grotesk',Inter,sans-serif}.pstep small{color:#9A93B8}.pstep.out{border-color:rgba(45,212,191,.5)}
.pk{font-size:11px;letter-spacing:2px;color:#5EEAD4}.parrow{align-self:center;color:#A78BFA;font-size:22px}
button[role=tab]{font-family:'Space Grotesk',Inter,sans-serif !important;font-size:15px !important}
button[role=tab][aria-selected=true]{color:#C4B5FD !important}
.foot{color:#6E6790;font-size:12px;text-align:center;margin-top:10px}
@keyframes away{to{opacity:0;visibility:hidden}}
@keyframes load{from{width:0}to{width:100%}}
@keyframes blink{50%{opacity:.35}}
@keyframes beat{50%{r:inherit;filter:drop-shadow(0 0 8px #2DD4BF)}}
.cap-fixed.linger{animation:fade .6s ease both,away .8s ease 7s forwards}
.scan{flex-direction:column;align-items:stretch}.scan .row{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.scan.load{border-color:rgba(167,139,250,.5)}.scan.load span{animation:blink 1s ease-in-out infinite}
.scan .bar{height:4px;border-radius:4px;background:rgba(255,255,255,.07);overflow:hidden;margin-top:10px}
.scan .bar i{display:block;height:100%;background:linear-gradient(90deg,#2DD4BF,#A78BFA);animation:load .95s ease-out forwards}
.scan .head{margin-top:10px;font:12px/1.6 'JetBrains Mono',ui-monospace,Menlo,monospace;color:#7DD3C8;background:rgba(0,0,0,.35);
  border-radius:10px;padding:8px 12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.orbit{transform-origin:90px 90px;animation:spin 1.4s linear infinite}
.pkt.up{fill:#F0ABFC;filter:drop-shadow(0 0 4px #F0ABFC)}.pkt.down{fill:#5EEAD4;filter:drop-shadow(0 0 4px #5EEAD4)}
.site.busy{animation:beat 1.2s ease-in-out infinite}
#capt{max-width:220px;margin:0 0 0 auto;background:transparent !important;border:none !important;opacity:.75}
#capt label{font-size:13px !important;color:#9A93B8 !important}
#capbox{min-height:0 !important;padding:0 !important;border:none !important;background:transparent !important}
.gradio-container{width:100% !important}
"""

JS = """
() => {
  document.body.classList.add('dark');
  const go = () => {
    const c = document.getElementById('neuro-canvas');
    if (!c) { setTimeout(go, 300); return; }
    const ctx = c.getContext('2d'); const dpr = window.devicePixelRatio || 1; let W = 0, H = 0;
    const size = () => { W = c.width = c.offsetWidth * dpr; H = c.height = c.offsetHeight * dpr; };
    size(); window.addEventListener('resize', size);
    const P = []; for (let i = 0; i < 80; i++) P.push({x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - .5) * .35 * dpr, vy: (Math.random() - .5) * .35 * dpr, t: Math.random() < .5});
    const R = 130 * dpr;
    const step = () => {
      ctx.clearRect(0, 0, W, H);
      for (const p of P) { p.x += p.vx; p.y += p.vy; if (p.x < 0 || p.x > W) p.vx *= -1; if (p.y < 0 || p.y > H) p.vy *= -1; }
      for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
        const a = P[i], b = P[j], d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < R) { ctx.strokeStyle = `rgba(${a.t ? '167,139,250' : '45,212,191'},${(1 - d / R) * .32})`; ctx.lineWidth = dpr;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); } }
      for (const p of P) { ctx.fillStyle = p.t ? '#A78BFA' : '#2DD4BF'; ctx.beginPath(); ctx.arc(p.x, p.y, 2.2 * dpr, 0, 6.3); ctx.fill(); }
      requestAnimationFrame(step);
    };
    step();
  };
  go();
}
"""
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&display=swap">')

_base = dict(body_background_fill="#07060F", body_text_color="#E9E6F5", body_text_color_subdued="#9A93B8",
             background_fill_primary="#0D0B18", background_fill_secondary="#12101F",
             block_background_fill="rgba(255,255,255,0.035)", block_border_color="rgba(167,139,250,0.18)",
             block_label_background_fill="rgba(124,58,237,0.20)", block_label_text_color="#DDD6FE",
             block_title_text_color="#DDD6FE", input_background_fill="rgba(255,255,255,0.05)",
             input_border_color="rgba(167,139,250,0.28)", border_color_primary="rgba(167,139,250,0.22)",
             button_primary_background_fill="linear-gradient(90deg,#7C3AED,#0D9488)",
             button_primary_background_fill_hover="linear-gradient(90deg,#8B5CF6,#14B8A6)",
             button_primary_text_color="#FFFFFF", button_secondary_background_fill="rgba(255,255,255,0.06)",
             button_secondary_text_color="#E9E6F5", color_accent="#A78BFA", color_accent_soft="rgba(167,139,250,0.16)",
             panel_background_fill="rgba(255,255,255,0.03)", table_even_background_fill="rgba(255,255,255,0.03)",
             table_odd_background_fill="rgba(255,255,255,0.015)", table_border_color="rgba(167,139,250,0.16)",
             checkbox_background_color="#12101F", slider_color="#A78BFA", loader_color="#A78BFA",
             block_shadow="0 10px 30px rgba(0,0,0,0.35)")
_keys = set(__import__("inspect").signature(gr.themes.Base().set).parameters)
THEME = gr.themes.Base(primary_hue=gr.themes.colors.violet, secondary_hue=gr.themes.colors.teal,
                       neutral_hue=gr.themes.colors.slate,
                       font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]).set(
    **{**_base, **{k + "_dark": v for k, v in _base.items() if k + "_dark" in _keys}})


# ─────────────────────────────────────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="FedKG-Brain", fill_width=True) as demo:
    gr.HTML(HERO)
    cap_on = gr.Checkbox(value=True, label="Live captions", container=False, elem_id="capt")
    cap_box = gr.HTML(elem_id="capbox")
    HIDE = dict(show_progress="hidden")

    with gr.Tab("⚡ Analyze a scan") as t_an:
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.HTML('<div class="sec">01 · Input</div>')
                up = gr.File(label="Drop a resting-state scan (.1D · time points × 200 regions)", file_types=[".1D", ".txt", ".csv"],
                             type="filepath", height=150, elem_id="an-file")
                pick = gr.Dropdown(choices=subject_choices(), label="…or a subject the model has never seen", value=None,
                                   elem_id="an-pick")
                with gr.Row():
                    rnd = gr.Button("🎲 Random subject", elem_id="an-random")
                    go_btn = gr.Button("Analyze", variant="primary", elem_id="an-go")
                scan_box = gr.HTML()
                spark = gr.Plot(show_label=False, visible=False)
            with gr.Column(scale=4):
                gr.HTML('<div class="sec">02 · Processing</div>')
                steps_box = gr.HTML(steps_html(ANALYZE_STEPS, ["todo"] * 5, {}), elem_id="an-steps")
                gr.HTML('<div class="sec">03 · Output</div>')
                gauge = gr.HTML(EMPTY_GAUGE, elem_id="an-gauge")
        with gr.Row():
            conn = gr.Plot(show_label=False, visible=False, elem_id="an-conn")
            attrp = gr.Plot(show_label=False, visible=False, elem_id="an-attr")
        brain = gr.Plot(show_label=False, visible=False, elem_id="an-brain")
        table = gr.Dataframe(label="Most influential connections", interactive=False, visible=False, elem_id="an-table")
        outs = [cap_box, steps_box, scan_box, spark, gauge, conn, attrp, brain, table]
        go_btn.click(analyze, [pick, up, cap_on], outs, **HIDE)
        up.upload(analyze, [pick, up, cap_on], outs, **HIDE)
        pick.input(lambda: None, None, up, **HIDE).then(analyze, [pick, up, cap_on], outs, **HIDE)
        rnd.click(pick_random, None, [pick, up], **HIDE).then(analyze, [pick, up, cap_on], outs, **HIDE)

    with gr.Tab("🌐 Federated training") as t_fed:
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                gr.HTML('<div class="sec">Input · training settings</div>')
                f_rounds = gr.Slider(1, 40, value=3 if ARGS.fast else 15, step=1, label="Communication rounds", elem_id="fed-rounds")
                f_local = gr.Slider(1, 5, value=2, step=1, label="Local epochs per round")
                f_bal = gr.Checkbox(value=False, label="Class-balanced loss")
                f_cmp = gr.CheckboxGroup(["Local only", "Centralized"], value=["Local only", "Centralized"], label="Also compare with")
                f_go = gr.Button("Start federated training", variant="primary", elem_id="fed-go")
                f_log = gr.Markdown(elem_id="fed-log")
            with gr.Column(scale=7):
                gr.HTML('<div class="sec">Hospitals ↔ server</div>')
                f_net = gr.HTML(hospital_net(active=False), elem_id="fed-net")
        f_curve = gr.Plot(show_label=False, visible=False, elem_id="fed-curve")
        f_cmpf = gr.Plot(show_label=False, visible=False, elem_id="fed-cmp")
        f_go.click(run_federated, [f_rounds, f_local, f_bal, f_cmp, cap_on], [cap_box, f_net, f_log, f_curve, f_cmpf], **HIDE)

    with gr.Tab("⏳ Connectivity evolution") as t_evo:
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.HTML('<div class="sec">Input · a scan</div>')
                e_up = gr.File(label="Drop a scan (.1D, ≥ 90 time points)", file_types=[".1D", ".txt", ".csv"], type="filepath",
                               height=130, elem_id="evo-file")
                e_pick = gr.Dropdown(choices=[], label="…or a subject from a site never seen in training", elem_id="evo-pick")
                with gr.Row():
                    e_rnd = gr.Button("🎲 Random subject", elem_id="evo-random")
                    e_go = gr.Button("Predict the future connectome", variant="primary", elem_id="evo-go")
                e_scan = gr.HTML()
            with gr.Column(scale=5):
                gr.HTML('<div class="sec">Processing</div>')
                e_steps = gr.HTML(steps_html(EVO_STEPS, ["todo"] * 5, {}), elem_id="evo-steps")
        e_fig = gr.Plot(show_label=False, visible=False, elem_id="evo-fig")
        e_cards = gr.HTML()
        e_outs = [cap_box, e_steps, e_scan, e_fig, e_cards]
        t_evo.select(evo_choices, None, e_pick)
        e_go.click(evolution, [e_pick, e_up, cap_on], e_outs, **HIDE)
        e_up.upload(evolution, [e_pick, e_up, cap_on], e_outs, **HIDE)
        e_pick.input(lambda: None, None, e_up, **HIDE).then(evolution, [e_pick, e_up, cap_on], e_outs, **HIDE)
        e_rnd.click(evo_random, None, [e_pick, e_up], **HIDE).then(evolution, [e_pick, e_up, cap_on], e_outs, **HIDE)

    with gr.Tab("🧠 Model") as t_model:
        gr.HTML('<div class="sec">How it works</div>' + PIPE_HTML)
        gr.HTML(stat_cards([("0.694", "AUC · GCN on raw graphs (5-fold CV)", VIOLET), ("0.736", "AUC · logistic-regression probe", TEAL),
                            ("0.701", "AUC · + knowledge graph (no real gain)", "#C4B5FD"), ("0.595", "AUC · FedAvg across 18 sites", PINK)]))
        gr.Dataframe(LOGGED, label="Logged results (ABIDE I, 871 subjects)", interactive=False)
        gr.Markdown("**Knowledge graphs** (functional networks, anatomy, gene co-expression, PrimeKG) were added as node features; "
                    "they did not change the results, because one-hot region identity already lets the GCN learn a vector per region.")
    for t in (t_an, t_fed, t_evo, t_model):        # a caption never follows you to another tab
        t.select(lambda: "", None, cap_box, **HIDE)
    gr.HTML('<div class="foot">FedKG-Brain · research prototype, not a medical device · ABIDE I (CPAC, cc200)</div>')

if __name__ == "__main__":
    demo.queue().launch(share=ARGS.share, server_port=ARGS.port, server_name="0.0.0.0", theme=THEME, css=CSS, js=JS,
                        head=HEAD, show_error=True)
