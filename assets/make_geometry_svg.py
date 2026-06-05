#!/usr/bin/env python3
"""Generate assets/neural_geometry.svg — a Hold-Your-Fire-styled, implication-first view of the
mechanistic-interpretability study.

Instead of an abstract layer-by-layer sweep, this shows ONE map: a 1.5B coding model's internal
state at a decision point, reduced to 2D (PCA) at the layer where the structure is sharpest.
Three things an agent can be doing land in three separate regions:
  - loop  : re-runs the SAME failing command            -> "going in circles"
  - vfail : DIFFERENT commands, still failing            -> "busy but stuck"
  - prog  : distinct productive commands                 -> "making progress"
(Definitions are verbatim from mech_interp/run_localize.py, which built acts.npz.)

The right column states what that buys you, honestly: the separation is why a tiny monitor can
read "stuck" from the trajectory and why, on this small model, a command-targeted penalty breaks
loops, plus the caveat that a capable 30B has no such steerable rut.

House style matches make_overview_svg.py. Geometry is precomputed in assets/geometry_data.json
so this renderer is pure-stdlib. Re-run:  python3 assets/make_geometry_svg.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "geometry_data.json").read_text())
OUT = ROOT / "neural_geometry.svg"
HERO = max(DATA["panels"], key=lambda p: p["sep"])          # sharpest-separation layer

# ---- light theme (from overview) ----
PAGE = "#ffffff"; SURF1 = "#fbfbfb"; SURF2 = "#f6f6f7"; SURF3 = "#f1f1f2"
BORDER = "#e4e4e6"; BORDER_M = "#d8d8dc"
INK = "#0a0a0a"; INK2 = "#2c2c2e"; GRAY = "#5c5c60"; MUTE = "#8a8a90"; FAINT = "#c2c2c8"
ACCENT = "#00e4b4"; ACCENT_D = "#0a8f76"; ACCENT_LN = "#9be8d5"; ACCENT_TINT = "#ecfbf6"
AMBER = "#d97706"; AMBER_TINT = "#fdf1e3"
SANS = "'Geist','Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,system-ui,sans-serif"
MONO = "'Geist Mono','SF Mono',ui-monospace,'JetBrains Mono','Roboto Mono',Menlo,Consolas,monospace"

# behaviour: amber = the literal loop, indigo = the no-progress churn, teal = healthy progress
CCOL = {0: "#d97706", 1: "#0e9f8b", 2: "#6366f1"}
# (key, headline, plain mono example) per condition index
MEAN = {
    0: ("GOING IN CIRCLES", "$ test.py -> FAIL  (x N)"),
    1: ("MAKING PROGRESS",  "$ edit -> test -> pass"),
    2: ("BUSY BUT STUCK",   "$ cat / grep / ls -> all FAIL"),
}

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

def line(x1, y1, x2, y2, stroke=BORDER, sw=1.0, dash=None, opacity=None, cap="butt"):
    a = f' stroke-dasharray="{dash}"' if dash else ""
    a += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"{a}/>')

def dot(cx, cy, r, fill, stroke=None, sw=0.0, opacity=None):
    a = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    a += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{a}/>')

# ============================================================ header
add(f'<rect x="{PAD}" y="30" width="7" height="7" fill="{ACCENT}"/>')
T(PAD + 15, 37, "Mechanistic interpretability", size=10.5, w=700, fill=ACCENT_D, upper=True, ls=1.1)
T(PAD, 69, "The model knows before it fails", size=26, w=300, fill=INK)
T(PAD, 93, "We read a coding model's internal state at a decision point and map it to 2D. The three things an",
  size=13.5, w=400, fill=GRAY)
T(PAD, 111, "agent can be doing land in three separate regions, so “stuck” is a real, readable state. The same split holds from 0.5B to 14B (bottom).",
  size=13.5, w=400, fill=GRAY)
line(PAD, 126, W - PAD, 126, stroke="url(#fade)", sw=1)

# ============================================================ left: the model -> zoom into layer 14 -> 2D map
# the 1.5B model drawn as a 28-layer stack, layer 14 highlighted
MX, MTOP, MW, MH = 102, 190, 26, 314
NLAY, L14 = 28, 14
LH = MH / NLAY
R(MX, MTOP, MW, MH, r=6, fill=SURF2, stroke=BORDER_M, sw=1)
for i in range(NLAY):
    yc = MTOP + MH - (i + 0.5) * LH
    if i == L14:
        R(MX - 3, yc - LH / 2 + 1, MW + 6, LH - 2, r=2, fill=ACCENT)
    else:
        line(MX + 4, yc, MX + MW - 4, yc, stroke=BORDER, sw=0.7, opacity=0.7)
L14y = MTOP + MH - (L14 + 0.5) * LH
T(MX + MW / 2, L14y + 3, "14", size=8, w=700, fill="#0a6b58", anchor="middle")
T(MX + MW / 2, MTOP + MH + 16, "28 transformer layers", size=8.5, w=600, fill=MUTE, anchor="middle", upper=True, ls=0.3)
add(f'<text x="{MX-14:.1f}" y="{MTOP+MH/2:.1f}" font-family="{SANS}" font-size="9.5" font-weight="700" '
    f'fill="{GRAY}" text-anchor="middle" letter-spacing="1.3" '
    f'transform="rotate(-90 {MX-14:.1f} {MTOP+MH/2:.1f})">QWEN2.5-CODER</text>')

# the 2D map is "layer 14, zoomed in": a viewport the model feeds into
PL, PT, PWD, PHT = 232, 190, 428, 314
PR, PB = PL + PWD, PT + PHT
line(MX + MW, L14y - LH / 2, PL, PT, stroke=ACCENT, sw=1, opacity=0.55, dash="3,3")
line(MX + MW, L14y + LH / 2, PL, PB, stroke=ACCENT, sw=1, opacity=0.55, dash="3,3")
R(PL, PT, PWD, PHT, r=8, fill="#ffffff", stroke=BORDER, sw=1)
T(PL, PT - 9, "Transformer layer 14 of 28, in 2D (PCA)", size=10.5, w=700, fill=ACCENT_D, upper=True, ls=0.6)

pts = HERO["points"]
xs = sorted(p[0] for p in pts); ys = sorted(p[1] for p in pts)
def pct(a, q): return a[min(len(a) - 1, max(0, int(q * (len(a) - 1))))]
minx, maxx, miny, maxy = pct(xs, .01), pct(xs, .99), pct(ys, .01), pct(ys, .99)
spanx, spany = (maxx - minx) or 1, (maxy - miny) or 1
pad = 40
sc = min((PWD - 2 * pad) / spanx, (PHT - 2 * pad) / spany)
cx0, cy0 = PL + PWD / 2, PT + PHT / 2
mx, my = (minx + maxx) / 2, (miny + maxy) / 2
def place(x, y): return (cx0 + (x - mx) * sc, cy0 - (y - my) * sc)

# per-cluster: soft region ellipse + crisp points + centroid
cen = {}
for k in (1, 2, 0):
    g = [(x, y) for (x, y, kk) in pts if kk == k]
    gcx, gcy = sum(a for a, _ in g) / len(g), sum(b for _, b in g) / len(g)
    sx = (sum((a - gcx) ** 2 for a, _ in g) / len(g)) ** .5
    sy = (sum((b - gcy) ** 2 for _, b in g) / len(g)) ** .5
    ecx, ecy = place(gcx, gcy)
    erx, ery = max(sx * sc * 2.6, 18), max(sy * sc * 2.6, 18)
    cen[k] = (ecx, ecy, erx, ery)
    add(f'<ellipse cx="{ecx:.1f}" cy="{ecy:.1f}" rx="{erx:.1f}" ry="{ery:.1f}" fill="{CCOL[k]}" opacity="0.10"/>')
for k in (1, 2, 0):
    for (x, y, kk) in pts:
        if kk == k:
            sx, sy = place(x, y); dot(sx, sy, 2.7, CCOL[k], stroke="#ffffff", sw=0.6, opacity=0.92)

# label each region directly above/below its own cluster (colour-matched, no crossing leaders)
def tag(k, side):
    head, mono = MEAN[k]
    ecx, ecy, _, ery = cen[k]
    if side == "above":
        hy, my = ecy - ery - 28, ecy - ery - 12
        line(ecx, my + 5, ecx, ecy - ery - 1, stroke=CCOL[k], sw=1.2, opacity=0.5)
    else:
        hy, my = ecy + ery + 22, ecy + ery + 38
        line(ecx, ecy + ery + 1, ecx, hy - 13, stroke=CCOL[k], sw=1.2, opacity=0.5)
    T(ecx, hy, head, size=12.5, w=700, fill=CCOL[k], anchor="middle", ls=0.3)
    T(ecx, my, mono, size=10, w=400, fill=GRAY, fam=MONO, anchor="middle")

tag(0, "above")      # GOING IN CIRCLES (loop, centre)
tag(1, "above")      # MAKING PROGRESS  (prog, lower-left)
tag(2, "below")      # BUSY BUT STUCK   (vfail, upper)

# ============================================================ "why it matters" (right column)
RX, RW = 688, W - PAD - 688
CARDS = [
    (1, ACCENT_D, ACCENT_TINT, ACCENT_LN, "YOU CAN SEE IT COMING",
     ["“Stuck” has its own region. A 1.14 MB monitor flags a",
      "failing run from the trajectory, ~13 steps before it ends."]),
    (2, ACCENT_D, ACCENT_TINT, ACCENT_LN, "YOU CAN BREAK THE LOOP",
     ["A penalty on the exact repeated command breaks 100% of",
      "audited loops at near-zero disruption. Steering alone: 0%."]),
    (3, AMBER, AMBER_TINT, "#f0d9b8", "STEERING STAYS SMALL-MODEL",
     ["Bigger models still read “stuck” (the split holds below),",
      "but no vector steers one out: the failure is competence."]),
]
ch, gap, PADX = 108, 18, 26
for i, (num, kc, tint, lnc, kick, body) in enumerate(CARDS):
    cy = PT + i * (ch + gap)
    R(RX, cy, RW, ch, r=14, fill=tint, stroke=lnc, sw=1)            # soft tinted wash
    bcx, bcy = RX + PADX + 13, cy + 30                              # numbered badge
    add(f'<circle cx="{bcx:.1f}" cy="{bcy:.1f}" r="13" fill="{kc}"/>')
    T(bcx, bcy + 4.5, str(num), size=13, w=700, fill="#ffffff", anchor="middle")
    T(bcx + 27, bcy + 4, kick, size=11.5, w=700, fill=kc, upper=True, ls=0.8)
    for j, ln in enumerate(body):
        T(RX + PADX, cy + 67 + j * 19, ln, size=12.5, w=400, fill=INK2)

# ============================================================ scale strip: the split holds across the family
sd = DATA["scale"]
sy0 = max(PB, PT + 3 * ch + 2 * gap) + 36
line(PAD, sy0 - 20, W - PAD, sy0 - 20, stroke="url(#fade)", sw=1)
add(f'<rect x="{PAD}" y="{sy0-9}" width="7" height="7" fill="{ACCENT}"/>')
T(PAD + 15, sy0 - 2, "The same split holds across the family", size=11, w=700, fill=ACCENT_D, upper=True, ls=0.9)
# mini chart (left): loop-vs-churn AUC vs model size (log), length-controlled
SCX, SCW, SCT, SCH = 84, 412, sy0 + 18, 52
pb, au, lo = sd["params_b"], sd["clean_auc"], sd["len_only"]
xmin, xmax = math.log10(min(pb)), math.log10(max(pb))
def gx(p): return SCX + (math.log10(p) - xmin) / (xmax - xmin) * SCW
def gy(v): return SCT + (1.03 - v) / (1.03 - 0.85) * SCH
line(SCX, gy(lo), SCX + SCW, gy(lo), stroke=AMBER, sw=1, dash="3,3", opacity=0.85)          # length-only cue
T(SCX + SCW, gy(lo) - 4, f"length cue {lo:.2f}", size=8.5, w=600, fill=AMBER, anchor="end")
add('<polyline points="' + " ".join(f"{gx(p):.1f},{gy(v):.1f}" for p, v in zip(pb, au)) +
    f'" fill="none" stroke="{ACCENT_D}" stroke-width="2"/>')                                # genuine signal
for p, v, s in zip(pb, au, sd["sizes"]):
    dot(gx(p), gy(v), 3.2, ACCENT_D, stroke="#ffffff", sw=1)
    T(gx(p), SCT + SCH + 13, s, size=9, w=600, fill=MUTE, fam=MONO, anchor="middle")
T(SCX - 6, gy(1.0) + 3, "1.0", size=8.5, w=500, fill=MUTE, anchor="end")
T(SCX - 6, gy(0.9) + 3, "0.9", size=8.5, w=500, fill=MUTE, anchor="end")
T(SCX + SCW, gy(1.0) - 6, "loop vs churn (clean)", size=8.5, w=600, fill=ACCENT_D, anchor="end")
# takeaway (right)
tx = SCX + SCW + 48
T(tx, sy0 + 22, "Re-run the same battery on Qwen2.5-Coder 0.5B to 14B and the loop-vs-churn", size=12, w=400, fill=INK2)
T(tx, sy0 + 41, "signal stays at ceiling (AUC ~1.0, length-controlled) across a 28x size range.", size=12, w=400, fill=INK2)
T(tx, sy0 + 63, "So the “stuck” signal a monitor reads is family-wide; only the looping behaviour", size=12, w=400, fill=INK2)
T(tx, sy0 + 82, "and the one-vector steering lever stay small-model.", size=12, w=400, fill=INK2)

# ============================================================ footer note
fy = max(SCT + SCH + 16, sy0 + 86) + 26
line(PAD, fy - 16, W - PAD, fy - 16, stroke="url(#fade)", sw=1)
T(PAD, fy, "Each dot in the map is one matched decision-point context, 72 per behaviour; a loop repeats one command, "
  "“busy but stuck” tries different commands that all keep failing. Map shown at the 1.5B's sharpest layer.",
  size=11.5, w=400, fill=MUTE)
H = fy + 20

# ============================================================ assemble
defs = (
    '<defs><linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
    f'<stop offset="0" stop-color="{BORDER}" stop-opacity="0"/>'
    f'<stop offset="0.5" stop-color="{BORDER}" stop-opacity="1"/>'
    f'<stop offset="1" stop-color="{BORDER}" stop-opacity="0"/></linearGradient></defs>'
)
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {int(H)}" width="{W}" height="{int(H)}" '
    f'font-family="{SANS}" role="img" '
    f'aria-label="The model knows before it fails. A 2D map of a coding model\'s internal state at a '
    f'decision point shows three separate regions: going in circles (re-running the same failing command), busy but '
    f'stuck (different commands that all keep failing), and making progress. Because the failing states are separate '
    f'and readable, a 1.14 MB monitor can flag a failing run early, and on this small model a penalty on the '
    f'specific repeated command breaks 100 percent of audited loops; a capable 30B model has no such steerable rut.">'
    + defs + f'<rect x="0" y="0" width="{W}" height="{int(H)}" fill="{PAGE}"/>' + "".join(els) + "</svg>\n"
)
OUT.write_text(svg)
print(f"wrote {OUT}  ({W}x{int(H)}, hero L{HERO['layer']}, {len(els)} elements, {len(svg)//1024} KB)")
