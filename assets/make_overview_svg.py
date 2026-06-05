#!/usr/bin/env python3
"""Generate assets/overview.svg — a light-themed, one-glance explainer of the Hold Your Fire paper.

Design: adapts design.json (a dark-mode-first system) to a LIGHT value scale while keeping its
soul — a single teal accent (#00e4b4) used sparingly, weight-inverted typography (large = light
300, small labels = bold uppercase + wide tracking), minimal borders, sharp cards with one
featured rounded card, and blueprint-style geometric decorations.

All numbers are taken verbatim from paper/paper_draft.md. Re-run to regenerate:
    python3 assets/make_overview_svg.py
"""
from pathlib import Path

# ----------------------------------------------------------------------------- design tokens
W = 1200
PAD = 56
CW = W - 2 * PAD                      # content width = 1088

# light-theme value scale (inverted from the dark system, same structure)
PAGE     = "#ffffff"
SURF1    = "#fbfbfb"
SURF2    = "#f6f6f7"
SURF3    = "#f1f1f2"
BORDER_S = "#eeeeee"
BORDER   = "#e4e4e6"
BORDER_M = "#d8d8dc"
INK      = "#0a0a0a"                  # primary text (near-black)
INK2     = "#2c2c2e"
GRAY     = "#5c5c60"                  # secondary / body
MUTE     = "#8a8a90"                  # labels / metadata
FAINT    = "#b6b6bc"                  # faint
ACCENT   = "#00e4b4"                  # accent FILL (bright teal) — shapes/bars only
ACCENT_D = "#0a8f76"                  # accent TEXT (darkened for AA contrast on white)
ACCENT_TINT = "#ecfbf6"              # accent wash background
ACCENT_LN = "#9be8d5"                # accent hairline
SUCCESS  = "#16a34a"
WARNING  = "#d97706"
ERROR    = "#dc2626"
ERROR_T  = "#fdecec"
INFO     = "#3D56F0"

SANS = "'Geist','Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,system-ui,sans-serif"
MONO = "'Geist Mono','SF Mono',ui-monospace,'JetBrains Mono','Roboto Mono',Menlo,Consolas,monospace"

els = []
def add(s): els.append(s)

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def T(x, y, s, size: float = 14, w=400, fill=INK, fam=SANS, anchor="start", ls=None, upper=False, opacity=None):
    if upper:
        s = s.upper()
        if ls is None:
            ls = 0.09 * size
    extra = f' letter-spacing="{ls:.2f}"' if ls is not None else ""
    extra += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
        f'font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{extra}>{esc(s)}</text>')

def R(x, y, w, h, r: float = 0, fill="none", stroke=None, sw: float = 1, dash=None, opacity=None):
    a = f' rx="{r}"' if r else ""
    a += f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    a += f' stroke-dasharray="{dash}"' if dash else ""
    a += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"{a} fill="{fill}"/>')

def line(x1, y1, x2, y2, stroke=BORDER, sw: float = 1, dash=None, cap="butt", opacity=None):
    a = f' stroke-dasharray="{dash}"' if dash else ""
    a += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{sw}" stroke-linecap="{cap}"{a}/>')

def circle(cx, cy, r, fill="none", stroke=None, sw: float = 1, opacity=None):
    a = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    a += f' opacity="{opacity}"' if opacity is not None else ""
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{a}/>')

def gradline(x1, x2, y):
    """blueprint-style gradient fade separator (design.json decorativeElements.separators)."""
    add(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="url(#fade)" stroke-width="1"/>')

def arrow(x1, x2, y, color=FAINT):
    """horizontal connector with a small chevron head, pointing right."""
    line(x1, y, x2 - 6, y, stroke=color, sw=1.5)
    add(f'<path d="M {x2-7:.1f} {y-4:.1f} L {x2:.1f} {y:.1f} L {x2-7:.1f} {y+4:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>')

def square_icon(x, y, sz, glyph):
    """square icon container (design.json sectionHeader.iconContainer) + a geometric glyph."""
    R(x, y, sz, sz, r=4, fill=SURF2, stroke=BORDER)
    cx, cy = x + sz / 2, y + sz / 2
    if glyph == "shield":
        add(f'<path d="M {cx:.1f} {cy-9:.1f} l 8 3 v 5 q 0 6 -8 9 q -8 -3 -8 -9 v -5 z" '
            f'fill="none" stroke="{ACCENT_D}" stroke-width="1.6" stroke-linejoin="round"/>')
        circle(cx, cy, 1.7, fill=ACCENT)
    elif glyph == "alert":
        add(f'<path d="M {cx:.1f} {cy-9:.1f} L {cx+9:.1f} {cy+7:.1f} L {cx-9:.1f} {cy+7:.1f} Z" '
            f'fill="none" stroke="{WARNING}" stroke-width="1.6" stroke-linejoin="round"/>')
        line(cx, cy - 4, cx, cy + 1.5, stroke=WARNING, sw=1.6, cap="round")
        circle(cx, cy + 4, 0.9, fill=WARNING)
    elif glyph == "flow":
        for i, dx in enumerate((-8, 8)):
            R(cx + dx - 4, cy - 8, 8, 7, r=1.5, fill="none", stroke=ACCENT_D, sw=1.5)
            R(cx + dx - 4, cy + 1, 8, 7, r=1.5, fill="none", stroke=ACCENT_D, sw=1.5)
    elif glyph == "chart":
        for i, h in enumerate((6, 11, 16)):
            R(cx - 9 + i * 7, cy + 8 - h, 4.5, h, r=1, fill=ACCENT if i == 2 else "none",
              stroke=ACCENT_D, sw=1.4)
    elif glyph == "star":
        add(f'<path d="M {cx:.1f} {cy-9:.1f} l 2.4 6.0 6.3 .3 -5 4 1.8 6.1 -5.5 -3.5 -5.5 3.5 1.8 -6.1 -5 -4 6.3 -.3 z" '
            f'fill="{ACCENT}" stroke="{ACCENT_D}" stroke-width="1" stroke-linejoin="round"/>')

def corner(x, y, s, dx, dy):
    """architectural corner accent (design.json decorativeElements.cornerAccents)."""
    add(f'<path d="M {x+dx*s:.1f} {y:.1f} L {x:.1f} {y:.1f} L {x:.1f} {y+dy*s:.1f}" '
        f'fill="none" stroke="{BORDER_M}" stroke-width="1.2" opacity="0.9"/>')

def section_header(y, label, title, desc=None, glyph="shield"):
    """square icon + uppercase kicker + light title + optional description. returns new y."""
    sz = 46
    square_icon(PAD, y, sz, glyph)
    tx = PAD + sz + 20
    T(tx, y + 16, label, size=11, w=600, fill=ACCENT_D, upper=True, ls=1.4)
    T(tx, y + 40, title, size=24, w=300, fill=INK, ls=-0.4)
    if desc:
        T(tx, y + 62, desc, size=13.5, w=400, fill=GRAY)
        return y + sz + 34
    return y + sz + 14

# ============================================================================= build
y = 56

# ---- title ----------------------------------------------------------------------
T(PAD, y + 22, "Disruption-Aware Failure Monitoring for Coding Agents", size=33, w=300, fill=INK, ls=-0.8)
y += 52
# two-tone thesis line (tspans auto-flow — no manual x)
add(f'<text x="{PAD:.1f}" y="{y+12:.1f}" font-family="{SANS}" font-size="18" letter-spacing="-0.2">'
    f'<tspan font-weight="400" fill="{GRAY}">Accurate prediction isn’t the goal — </tspan>'
    f'<tspan font-weight="500" fill="{ACCENT_D}">calibrated abstention is.</tspan></text>')
y += 34
gradline(PAD, W - PAD, y)
y += 40

# ---- 1. THE PROBLEM: intervention paradox --------------------------------------
y = section_header(y, "The problem", "The Intervention Paradox",
                   "A monitor that interrupts a run which would have succeeded causes the very failure it meant to prevent.",
                   glyph="alert")
py = y
ph = 132
# two side-by-side scenario cards
gap = 22
cwd = (CW - gap) / 2
def scenario(x, w, headline, hcol, traj_fail, verdict_good):
    R(x, py, w, ph, r=8, fill=PAGE, stroke=BORDER)
    R(x, py, 4, ph, r=0, fill=hcol)            # left status spine
    T(x + 22, py + 30, headline, size=13, w=600, fill=hcol, upper=True, ls=0.8)
    # mini trajectory: dots along a line, monitor fires at the marked step
    tx0, tx1 = x + 24, x + w - 130
    ty = py + 82
    n = 7
    am = tx0 + (tx1 - tx0) * 3 / (n - 1)            # alarm at step 4
    # alarm label + downward caret, on their own line above the track (clears the headline)
    T(am, py + 56, "monitor fires", size=9, w=600, fill=WARNING, anchor="middle", upper=True, ls=0.6)
    add(f'<path d="M {am-5:.1f} {ty-16:.1f} L {am+5:.1f} {ty-16:.1f} L {am:.1f} {ty-8:.1f} Z" fill="{WARNING}"/>')
    line(tx0, ty, tx1, ty, stroke=BORDER_M, sw=1.5)
    for i in range(n):
        cx = tx0 + (tx1 - tx0) * i / (n - 1)
        if i == n - 1:
            circle(cx, ty, 5.2, fill=(ERROR if traj_fail else SUCCESS))
        else:
            circle(cx, ty, 3.4, fill=PAGE, stroke=BORDER_M, sw=1.5)
    T(tx0, ty + 26, "trajectory prefix", size=10.5, w=500, fill=MUTE, upper=True, ls=0.5)
    # verdict tag on the right
    vx = x + w - 112
    vcol = SUCCESS if verdict_good else ERROR
    vtint = "#edf9f1" if verdict_good else ERROR_T
    R(vx, py + 50, 92, 46, r=6, fill=vtint, stroke=(vcol + "55"))
    mark = ("good", "catch") if verdict_good else ("back-", "fires")
    # check or x glyph
    gx, gy = vx + 16, py + 73
    if verdict_good:
        add(f'<path d="M {gx-5:.1f} {gy:.1f} l 3.5 4 l 7 -9" fill="none" stroke="{vcol}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        add(f'<path d="M {gx-5:.1f} {gy-4.5:.1f} l 9 9 m 0 -9 l -9 9" fill="none" stroke="{vcol}" stroke-width="2" stroke-linecap="round"/>')
    T(vx + 30, py + 70, mark[0], size=11.5, w=600, fill=vcol)
    T(vx + 30, py + 85, mark[1], size=11.5, w=600, fill=vcol)

scenario(PAD, cwd, "Fire on a run that was failing", ERROR,
         traj_fail=True, verdict_good=True)
scenario(PAD + cwd + gap, cwd, "Fire on a run that would succeed", SUCCESS,
         traj_fail=False, verdict_good=False)
# caption under the pair
T(PAD, py + ph + 24, "So the cost that matters isn’t classification error—it’s the false-alarm rate on healthy runs.",
  size=12.5, w=400, fill=GRAY)
y = py + ph + 24 + 26
gradline(PAD, W - PAD, y)
y += 38

# ---- 2. THE APPROACH: pipeline -------------------------------------------------
y = section_header(y, "The approach", "A local monitor that knows when to stay silent",
                   "Cheap CPU-only features → a calibrated 1.14 MB model → a selective gate that abstains until the evidence is there.",
                   glyph="flow")
py = y
# top row: 3 process cards
mgap = 40
mcw = (CW - 2 * mgap) / 3
mch = 104
steps = [
    ("01", "Agent trajectory", "partial run, step k", "raw", False),
    ("02", "Structured features", "repetition · search/edit ratio · test signals", "cheap", False),
    ("03", "Calibrated monitor", "HistGradientBoosting · 1.14 MB · isotonic", "featured", True),
]
mx = []
for i, (idx, ttl, sub, kind, feat) in enumerate(steps):
    x = PAD + i * (mcw + mgap)
    mx.append(x)
    if feat:
        R(x, py, mcw, mch, r=8, fill=ACCENT_TINT, stroke=ACCENT_LN, sw=1.4)
        R(x, py, mcw, 3, fill=ACCENT)
    else:
        R(x, py, mcw, mch, r=8, fill=PAGE, stroke=BORDER)
    T(x + 20, py + 30, idx, size=11, w=700, fill=(ACCENT_D if feat else FAINT), fam=MONO, ls=0.5)
    T(x + 20, py + 56, ttl, size=17, w=500, fill=INK, ls=-0.3)
    T(x + 20, py + 80, sub, size=11.5, w=400, fill=GRAY)
    if i < 2:
        arrow(x + mcw + 8, x + mcw + mgap - 8, py + mch / 2, color=FAINT)

# down-connector from card 03 to the gate panel
gx_c = mx[2] + mcw / 2
line(gx_c, py + mch, gx_c, py + mch + 22, stroke=FAINT, sw=1.5)
add(f'<path d="M {gx_c-4:.1f} {py+mch+15:.1f} L {gx_c:.1f} {py+mch+22:.1f} L {gx_c+4:.1f} {py+mch+15:.1f} '
    f'" fill="none" stroke="{FAINT}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>')

# gate panel (featured, full width)
gy = py + mch + 22
gh = 96
R(PAD, gy, CW, gh, r=8, fill=SURF1, stroke=BORDER)
# left: the gate label + rule
square_icon(PAD + 20, gy + 26, 44, "shield")
T(PAD + 84, gy + 30, "04 · ABSTENTION GATE", size=11, w=700, fill=ACCENT_D, upper=True, ls=1.0)
T(PAD + 84, gy + 54, "Commit only when", size=15, w=400, fill=INK)
add(f'<text x="{PAD+228:.1f}" y="{gy+54:.1f}" font-family="{MONO}" font-size="13.5" font-weight="500" '
    f'fill="{INK}">step ≥ S</text>')
T(PAD + 312, gy + 54, "and", size=13.5, w=400, fill=GRAY)
add(f'<text x="{PAD+345:.1f}" y="{gy+54:.1f}" font-family="{MONO}" font-size="13.5" font-weight="500" '
    f'fill="{INK}">|risk − 0.5| ≥ C</text>')
T(PAD + 84, gy + 76, "otherwise it returns “insufficient evidence” and stays silent.", size=12, w=400, fill=GRAY)
# right: two outcome chips
chip_w, chip_gap = 166, 14
ox = PAD + CW - 26 - (2 * chip_w + chip_gap)   # 26px right padding inside the panel
def outcome(x, w, h, color, tint, glyph_kind, top, bot):
    R(x, gy + (gh - h) / 2, w, h, r=6, fill=tint, stroke=(color if color != MUTE else BORDER_M))
    cx, cyc = x + 22, gy + gh / 2
    if glyph_kind == "fire":
        add(f'<path d="M {cx:.1f} {cyc-10:.1f} L {cx+9:.1f} {cyc+6:.1f} L {cx-9:.1f} {cyc+6:.1f} Z" '
            f'fill="{color}"/>')
    else:
        circle(cx, cyc, 8.5, fill="none", stroke=MUTE, sw=1.8)
        line(cx - 5.5, cyc + 5.5, cx + 5.5, cyc - 5.5, stroke=MUTE, sw=1.8, cap="round")
    T(x + 40, cyc - 2, top, size=12.5, w=600, fill=(color if color != MUTE else GRAY))
    T(x + 40, cyc + 15, bot, size=9.5, w=500, fill=MUTE, upper=True, ls=0.4)
outcome(ox, chip_w, 64, ACCENT_D, ACCENT_TINT, "fire", "Fire warning", "confident failure")
outcome(ox + chip_w + chip_gap, chip_w, 64, MUTE, SURF2, "mute", "Stay silent", "not yet judgeable")
y = gy + gh + 30
gradline(PAD, W - PAD, y)
y += 38

# ---- 3. THE RESULTS: stat grid + evidence --------------------------------------
y = section_header(y, "The results", "What the monitor buys you",
                   glyph="chart")
y += 4
# 6 stat cards, 3 x 2
gx = [PAD, PAD + (CW + 22) / 3 * 0 + (CW - 44) / 3 + 22, 0]
col_w = (CW - 2 * 22) / 3
colx = [PAD + i * (col_w + 22) for i in range(3)]
sch = 118
stats = [
    ("0.722", "ROC AUC", "base monitor · ~13-step warning lead", ACCENT_D),
    ("0.80", "AUC @ 50% COVERAGE", "as a selective predictor that abstains", ACCENT_D),
    ("79% → 7%", "LIVE DISRUPTIONS", "gated vs ungated, replayed on a live agent", ACCENT_D),
    ("−35%", "FALSE ALARMS", "fewer interruptions of healthy runs", INK),
    ("210 MB · 9 ms", "FOOTPRINT / PREFIX", "CPU-only · no GPU · serves from RAM", INK),
    ("0.011", "ECE (CALIBRATED)", "risk scores you can threshold honestly", INK),
]
def stat_card(x, yy, big, label, cap, bigcol):
    R(x, yy, col_w, sch, r=8, fill=PAGE, stroke=BORDER)
    corner(x + 12, yy + 12, 7, 1, 1)
    corner(x + col_w - 12, yy + 12, 7, -1, 1)
    T(x + 22, yy + 56, big, size=30, w=300, fill=bigcol, ls=-1.0)
    T(x + 22, yy + 80, label, size=10.5, w=600, fill=MUTE, upper=True, ls=0.8)
    T(x + 22, yy + 100, cap, size=11.5, w=400, fill=GRAY)
for i, (big, label, cap, bigcol) in enumerate(stats):
    r, c = divmod(i, 3)
    stat_card(colx[c], y + r * (sch + 22), big, label, cap, bigcol)
y += 2 * sch + 22 + 30

# evidence card: judges bar chart (left) + cross-scaffold recovery (right)
ev_h = 226
R(PAD, y, CW, ev_h, r=8, fill=SURF1, stroke=BORDER)
mid = PAD + CW * 0.56
line(mid, y + 22, mid, y + ev_h - 22, stroke=BORDER, sw=1)

# -- left: out-predicts LLM judges
lx = PAD + 26
T(lx, y + 32, "OUT-PREDICTS FRONTIER LLM JUDGES", size=11, w=700, fill=INK, upper=True, ls=0.8)
T(lx, y + 50, "same 200 prefixes · ROC AUC · ~10⁴× lower cost", size=11.5, w=400, fill=GRAY)
judges = [
    ("Structured classifier", 0.77, True),
    ("GPT-5.5", 0.63, False),
    ("Claude Opus 4.8", 0.62, False),
    ("qwen2.5-coder 7B", 0.56, False),
]
bx = lx + 152
bx_max = mid - 70
lo, hi = 0.5, 0.82
by0 = y + 70
bh, bgap = 20, 12
for i, (name, v, hot) in enumerate(judges):
    yy = by0 + i * (bh + bgap)
    T(lx, yy + bh - 6, name, size=11.5, w=(600 if hot else 400), fill=(INK if hot else GRAY))
    R(bx, yy, bx_max - bx, bh, r=3, fill=SURF3)                      # track
    wv = (v - lo) / (hi - lo) * (bx_max - bx)
    R(bx, yy, max(wv, 2), bh, r=3, fill=(ACCENT if hot else "#c9c9cf"))
    T(bx_max + 8, yy + bh - 6, f"{v:.2f}", size=11.5, w=600,
      fill=(ACCENT_D if hot else MUTE), fam=MONO)
# chance marker
cxp = bx + (0.5 - lo) / (hi - lo) * (bx_max - bx)  # = bx (0.5 is lo)
T(lx, by0 + 4 * (bh + bgap) + 8, "local 7B + two frontier judges — all beaten, robustly.",
  size=11, w=400, fill=MUTE)

# -- right: cross-scaffold free recovery
rx = mid + 30
T(rx, y + 32, "CROSS-SCAFFOLD: FREE RECOVERY", size=11, w=700, fill=INK, upper=True, ls=0.8)
T(rx, y + 50, "new agent (CodeAct) · no labels, no retraining", size=11.5, w=400, fill=GRAY)
rec = [("naive", 0.53), ("+align", 0.59), ("+ensemble", 0.60), ("+abstain", 0.66)]
ceiling = 0.72
chart_x = rx + 6
chart_w = (W - PAD - 26) - chart_x
base_y = y + ev_h - 54
top_y = y + 78
vlo, vhi = 0.5, 0.74
def vy(v): return base_y - (v - vlo) / (vhi - vlo) * (base_y - top_y)
# ceiling dashed line
line(chart_x, vy(ceiling), chart_x + chart_w, vy(ceiling), stroke=FAINT, sw=1, dash="4 4")
T(chart_x + chart_w, vy(ceiling) - 6, "in-domain ceiling 0.72", size=9.5, w=500, fill=MUTE, anchor="end", upper=True, ls=0.4)
# baseline (chance)
line(chart_x, base_y, chart_x + chart_w, base_y, stroke=BORDER_M, sw=1)
T(chart_x, base_y + 14, "0.50 chance", size=9, w=500, fill=FAINT, upper=True, ls=0.4)
nb = len(rec)
bw = 40
slot = chart_w / nb
for i, (lab, v) in enumerate(rec):
    cxb = chart_x + slot * i + (slot - bw) / 2
    topv = vy(v)
    hot = i == nb - 1
    R(cxb, topv, bw, base_y - topv, r=3, fill=(ACCENT if hot else "#cfeee5"))
    T(cxb + bw / 2, topv - 7, f"{v:.2f}", size=11, w=600, fill=(ACCENT_D if hot else GRAY), anchor="middle", fam=MONO)
    T(cxb + bw / 2, base_y + 28, lab, size=9.5, w=600, fill=(INK if hot else MUTE), anchor="middle", upper=True, ls=0.3)
# connecting trend dots
pts = []
for i, (lab, v) in enumerate(rec):
    cxb = chart_x + slot * i + slot / 2
    pts.append((cxb, vy(v)))
add('<polyline points="' + " ".join(f"{px:.1f},{py2:.1f}" for px, py2 in pts) +
    f'" fill="none" stroke="{ACCENT_D}" stroke-width="1.4" stroke-dasharray="2 3" opacity="0.6"/>')

y += ev_h + 30
gradline(PAD, W - PAD, y)
y += 38

# ---- 4. CONTRIBUTIONS ----------------------------------------------------------
y = section_header(y, "Contributions", "What’s new here", glyph="star")
y += 6
contribs = [
    ("Disruption-aware selective prediction",
     "Reframe agent monitoring as calibrated abstention, not a higher-AUC classifier."),
    ("A protocol that keeps you honest",
     "A leakage-controlled, paired instance-grouped bootstrap that caught validation-overfitting five times."),
    ("Cheap local features beat frontier judges",
     "A 1.14 MB CPU model out-predicts GPT-5.5 and Claude Opus 4.8 at ~10⁴× lower cost."),
    ("Cross-scaffold collapse is a fixable artifact",
     "The near-chance transfer is mostly a feature-scale mismatch—recoverable for free."),
    ("Monitorability is scaffold-driven",
     "Not capability-driven: within a scaffold, monitorability rises with capability (0.56→0.76)."),
]
rowh = 50
for i, (ttl, desc) in enumerate(contribs):
    ry = y + i * rowh
    T(PAD + 2, ry + 24, f"{i+1:02d}", size=18, w=300, fill=ACCENT_D, fam=MONO)
    line(PAD + 44, ry + 8, PAD + 44, ry + 38, stroke=BORDER, sw=1)
    T(PAD + 64, ry + 18, ttl, size=15, w=600, fill=INK, ls=-0.2)
    T(PAD + 64, ry + 37, desc, size=12.5, w=400, fill=GRAY)
    if i < len(contribs) - 1:
        gradline(PAD + 64, W - PAD, ry + rowh - 2)
y += len(contribs) * rowh + 12

H = int(y + 30)

# ============================================================================= assemble
frame = []
# page background + soft rounded frame so it reads as a card even in GitHub dark mode
frame.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAGE}"/>')
frame.append(f'<rect x="6" y="6" width="{W-12}" height="{H-12}" rx="14" fill="none" stroke="{BORDER}" stroke-width="1"/>')
# blueprint corner accents on the outer frame
for (cx, cy, dx, dy) in [(22, 22, 1, 1), (W - 22, 22, -1, 1), (22, H - 22, 1, -1), (W - 22, H - 22, -1, -1)]:
    frame.append(f'<path d="M {cx+dx*14:.1f} {cy:.1f} L {cx:.1f} {cy:.1f} L {cx:.1f} {cy+dy*14:.1f}" '
                 f'fill="none" stroke="{BORDER_M}" stroke-width="1.4"/>')

defs = (
    '<defs>'
    f'<linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
    f'<stop offset="0" stop-color="{BORDER}" stop-opacity="0"/>'
    f'<stop offset="0.5" stop-color="{BORDER_M}" stop-opacity="1"/>'
    f'<stop offset="1" stop-color="{BORDER}" stop-opacity="0"/>'
    f'</linearGradient>'
    '</defs>'
)

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'font-family="{SANS}" role="img" '
    f'aria-label="Hold Your Fire: disruption-aware failure monitoring for coding agents — techniques, results, and contributions">'
    + defs + "".join(frame) + "".join(els) + "</svg>\n"
)

out = Path(__file__).parent / "overview.svg"
out.write_text(svg)
print(f"wrote {out}  ({W}x{H}, {len(els)} elements, {len(svg)//1024} KB)")
