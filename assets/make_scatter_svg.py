#!/usr/bin/env python3
"""Generate assets/scale_scatter.svg — the intuitive "contact sheet" of the scale study, with each
2D map anchored to the real transformer architecture that produced it.

For each Qwen2.5-Coder model (0.5B -> 14B) we draw its actual transformer stack as a column of
blocks (one block per layer; stack height = layer count, stack width = hidden size), highlight the
single layer we read, and funnel it down into a 2D map of that layer's activations. Each dot is one
decision-point context; the three agent behaviors land in three separate clumps, in every model.

Data: assets/scale_scatter.json (PCA-2D coords per model at its sharpest layer) +
mech_interp/results/scale_localize.json (d_model per model). Pure-stdlib renderer.
Re-run:  python3 assets/make_scatter_svg.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "scale_scatter.json").read_text())
ARCH = {r["size"]: r for r in json.loads((ROOT.parent / "mech_interp" / "results" / "scale_localize.json").read_text())}
OUT = ROOT / "scale_scatter.svg"

PAGE = "#ffffff"; SURF2 = "#f6f6f7"
BORDER = "#e4e4e6"; BORDER_M = "#d8d8dc"
INK = "#0a0a0a"; GRAY = "#5c5c60"; MUTE = "#8a8a90"; FAINT = "#c2c2c8"
BLK_A = "#edeef0"; BLK_B = "#f6f6f8"
ACCENT = "#00e4b4"; ACCENT_D = "#0a8f76"; ACCENT_DK = "#0a6b58"
SANS = "'Geist','Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,system-ui,sans-serif"
MONO = "'Geist Mono','SF Mono',ui-monospace,'JetBrains Mono','Roboto Mono',Menlo,Consolas,monospace"

CONDS = DATA["conds"]                                          # ["loop","vfail","prog"]
COL = {"loop": "#d97706", "vfail": "#6366f1", "prog": "#0e9f8b"}
HEAD = {"loop": "Going in circles", "vfail": "Busy but stuck", "prog": "Making progress"}
SUB = {"loop": "re-runs the same failing command",
       "vfail": "different commands, all failing",
       "prog": "edits that move the tests forward"}
LEG_ORDER = ["loop", "vfail", "prog"]
ORDER = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
MODELS = sorted(DATA["models"], key=lambda m: ORDER.index(m["size"]) if m["size"] in ORDER else 99)

DMS = [ARCH[m["size"]]["d_model"] for m in MODELS]
NLS = [m["n_layers"] for m in MODELS]
DMIN, DMAX = min(DMS), max(DMS)

W = 1200; PAD = 52
els = []
def add(s): els.append(s)
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def T(x, y, s, size: float = 14, w=400, fill=INK, fam=SANS, anchor="start", ls=None, upper=False, opacity=None):
    if upper:
        s = s.upper(); ls = 0.11 * size if ls is None else ls
    e = f' letter-spacing="{ls:.2f}"' if ls is not None else ""
    e += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
        f'font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{e}>{esc(s)}</text>')

def R(x, y, w, h, r: float = 0, fill="none", stroke=None, sw=1.0, opacity=None):
    a = f' rx="{r}"' if r else ""
    a += f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    a += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"{a} fill="{fill}"/>')

def line(x1, y1, x2, y2, stroke=BORDER, sw=1.0, dash=None, opacity=None):
    a = f' stroke-dasharray="{dash}"' if dash else ""
    a += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{a}/>')

def dot(cx, cy, r, fill, stroke=None, sw=0.0, opacity=None):
    a = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    a += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{a}/>')

# ============================================================ header
add(f'<rect x="{PAD}" y="30" width="7" height="7" fill="{ACCENT}"/>')
T(PAD + 15, 37, "Mechanistic interpretability", size=10.5, w=700, fill=ACCENT_D, upper=True, ls=1.1)
T(PAD, 69, "Every model size from 0.5B to 14B knows it's stuck", size=26, w=300, fill=INK)
T(PAD, 93, "Each map below is one transformer layer of one model, reduced to 2D. A dot is one moment in an agent's run; we read the",
  size=13.5, w=400, fill=GRAY)
T(PAD, 111, "layer that separates the behaviors best. They land in three separate clumps in every model, so being stuck is directly readable.",
  size=13.5, w=400, fill=GRAY)

line(PAD, 126, W - PAD, 126, stroke="url(#fade)", sw=1)

# ============================================================ per-model column: architecture -> map
N = len(MODELS)
GAP = 16
LEG_W = 220
CONTENT_R = W - PAD - LEG_W                                   # leave a right gutter for the legend
CW = (CONTENT_R - PAD - (N - 1) * GAP) / N
X0 = PAD
SIZE_Y = 150
STACK_TOP = 182
CELL = 3.0                                                    # px per transformer block
SY = 356                                                      # scatter top (below tallest stack)
PS = CW

def pct(a, q):
    a = sorted(a)
    return a[min(len(a) - 1, max(0, int(q * (len(a) - 1))))]

def arch_width(dm):                                           # wider stack = larger hidden size
    return 44 + (dm - DMIN) / (DMAX - DMIN) * 52

for k, m in enumerate(MODELS):
    size = m["size"]; nL = m["n_layers"]; L = m["layer"]; dm = ARCH[size]["d_model"]
    colx = X0 + k * (CW + GAP); cx = colx + CW / 2
    Hs = nL * CELL; Wd = arch_width(dm)
    sx = cx - Wd / 2; sb = STACK_TOP + Hs                     # stack bottom (input side)
    gap = min(0.5, CELL * 0.16)

    # title above stack
    T(cx, SIZE_Y, size, size=15, w=700, fill=INK, anchor="middle")
    T(cx, SIZE_Y + 14, f"{nL} layers · {dm}-d", size=9, w=500, fill=MUTE, anchor="middle")

    # transformer blocks (one rect per layer; input at bottom, output at top)
    for i in range(nL):
        by = sb - (i + 1) * CELL
        if i == L:
            continue
        R(sx, by, Wd, CELL - gap, r=0.6, fill=(BLK_A if i % 2 == 0 else BLK_B), stroke=BORDER, sw=0.35)
    R(sx, STACK_TOP, Wd, Hs, r=3, fill="none", stroke=BORDER_M, sw=1)
    # probed layer (highlighted, slight overhang) + index
    by = sb - (L + 1) * CELL
    R(sx - 3.5, by - 1, Wd + 7, (CELL - gap) + 2, r=1.4, fill=ACCENT)
    T(sx - 9, by + CELL / 2 + 2.5, str(L), size=8.5, w=700, fill=ACCENT_DK, anchor="end")

    # funnel: read THIS layer -> THIS map
    yL = by + (CELL - gap) / 2
    line(cx, yL + 3, cx, SY - 9, stroke=ACCENT_D, sw=1.1, dash="2.5,2.5", opacity=0.8)
    add(f'<polygon points="{cx-4:.1f},{SY-9:.1f} {cx+4:.1f},{SY-9:.1f} {cx:.1f},{SY-2:.1f}" fill="{ACCENT_D}"/>')

    # the 2D map
    R(colx, SY, PS, PS, r=10, fill="#ffffff", stroke=BORDER, sw=1)
    T(colx + PS / 2, SY + PS + 18, f"Read at Layer {L}", size=10.5, w=600, fill=GRAY, anchor="middle")
    pts = m["points"]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    minx, maxx = pct(xs, .02), pct(xs, .98); miny, maxy = pct(ys, .02), pct(ys, .98)
    spanx = (maxx - minx) or 1; spany = (maxy - miny) or 1
    ipad = 18
    sc = min((PS - 2 * ipad) / spanx, (PS - 2 * ipad) / spany)
    mx, my = (minx + maxx) / 2, (miny + maxy) / 2
    pcx, pcy = colx + PS / 2, SY + PS / 2
    def place(x, y): return (pcx + (x - mx) * sc, pcy - (y - my) * sc)
    for c in CONDS:
        ci = CONDS.index(c); g = [(x, y) for (x, y, kk) in pts if kk == ci]
        if not g:
            continue
        gcx = sum(a for a, _ in g) / len(g); gcy = sum(b for _, b in g) / len(g)
        sxx = (sum((a - gcx) ** 2 for a, _ in g) / len(g)) ** .5
        syy = (sum((b - gcy) ** 2 for _, b in g) / len(g)) ** .5
        ex, ey = place(gcx, gcy)
        add(f'<ellipse cx="{ex:.1f}" cy="{ey:.1f}" rx="{max(sxx*sc*2.3,12):.1f}" ry="{max(syy*sc*2.3,12):.1f}" '
            f'fill="{COL[c]}" opacity="0.10"/>')
    for c in CONDS:
        ci = CONDS.index(c)
        for (x, y, kk) in pts:
            if kk == ci:
                xx, yy = place(x, y); dot(xx, yy, 2.1, COL[c], stroke="#ffffff", sw=0.5, opacity=0.9)

# ============================================================ behavior legend (stacked, right gutter)
lgx = CONTENT_R + 20
lg0 = SY + PS / 2 - 60
for i, c in enumerate(LEG_ORDER):
    yy = lg0 + i * 60
    dot(lgx, yy - 4, 6, COL[c], stroke="#ffffff", sw=1)
    T(lgx + 17, yy, HEAD[c], size=12.5, w=700, fill=COL[c])
    T(lgx + 17, yy + 15, SUB[c], size=10, w=400, fill=GRAY)

# ============================================================ footer
fy = SY + PS + 46
line(PAD, fy - 18, W - PAD, fy - 18, stroke="url(#fade)", sw=1)
T(PAD, fy, "Why it matters: whatever size of coding model you run, it already “knows” inside whether it is stuck or making "
  "real progress.",
  size=11.5, w=600, fill=GRAY)
T(PAD, fy + 17, "So catching failures doesn't take a huge model. A small, cheap monitor can read that signal and flag a stuck run "
  "early, before it wastes your time and compute.",
  size=11.5, w=400, fill=MUTE)
H = fy + 36

# ============================================================ assemble
defs = (
    '<defs><linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
    f'<stop offset="0" stop-color="{BORDER}" stop-opacity="0"/>'
    f'<stop offset="0.5" stop-color="{BORDER}" stop-opacity="1"/>'
    f'<stop offset="1" stop-color="{BORDER}" stop-opacity="0"/></linearGradient></defs>'
)
sizes = ", ".join(m["size"] for m in MODELS)
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {int(H)}" width="{W}" height="{int(H)}" '
    f'font-family="{SANS}" role="img" '
    f'aria-label="A contact sheet across Qwen2.5-Coder ({sizes}). Above each 2D map sits the model\'s real transformer '
    f'stack drawn as a column of per-layer blocks, with height proportional to its layer count and width to its hidden '
    f'size, the probed layer highlighted and funneling into the map below. In every map the three agent behaviors, going '
    f'in circles (amber), busy but stuck (indigo), and making progress (teal), form three separate clusters, so being '
    f'stuck is directly readable from the internal state at any size.">'
    + defs + f'<rect x="0" y="0" width="{W}" height="{int(H)}" fill="{PAGE}"/>' + "".join(els) + "</svg>\n"
)
OUT.write_text(svg)
print(f"wrote {OUT}  ({W}x{int(H)}, {N} models [{sizes}], {len(els)} elements, {len(svg)//1024} KB)")
