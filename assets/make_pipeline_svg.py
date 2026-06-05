#!/usr/bin/env python3
"""Generate assets/pipeline.svg — a light-themed, concrete walk-through of the Hold Your Fire
data pipeline: what a trajectory is, how it becomes prefixes + labels, exactly which features
are fed to the model, and what the model outputs.

Grounded in the real code: trajectory = list of (thought, action, observation) StepEvents with
a terminal label (schemas.py); features are the actual families/keys from features.py; the
worked example computes real feature values for a 6-step failing/looping prefix.

Same light design language as assets/overview.svg (single teal accent, weight-inverted type,
mono for data, sharp cards + one featured, blueprint corners). Re-run:
    python3 assets/make_pipeline_svg.py
"""
from pathlib import Path

# ----------------------------------------------------------------------------- design tokens
W = 1200
PAD = 56
CW = W - 2 * PAD

PAGE     = "#ffffff"
SURF1    = "#fbfbfb"
SURF2    = "#f6f6f7"
SURF3    = "#f1f1f2"
BORDER_S = "#eeeeee"
BORDER   = "#e4e4e6"
BORDER_M = "#d8d8dc"
INK      = "#0a0a0a"
INK2     = "#2c2c2e"
GRAY     = "#5c5c60"
MUTE     = "#8a8a90"
FAINT    = "#b6b6bc"
ACCENT   = "#00e4b4"
ACCENT_D = "#0a8f76"
ACCENT_TINT = "#ecfbf6"
ACCENT_LN = "#9be8d5"
SUCCESS  = "#16a34a"
WARNING  = "#d97706"
WARN_TINT = "#fdf3e7"
ERROR    = "#dc2626"
ERROR_T  = "#fdecec"
INFO     = "#3D56F0"
INFO_TINT = "#eef0fe"

SANS = "'Geist','Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,system-ui,sans-serif"
MONO = "'Geist Mono','SF Mono',ui-monospace,'JetBrains Mono','Roboto Mono',Menlo,Consolas,monospace"

els = []
def add(s): els.append(s)

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

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
    add(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="url(#fade)" stroke-width="1"/>')

def varrow(x, y1, y2, color=FAINT):
    """vertical connector, arrowhead pointing down."""
    line(x, y1, x, y2 - 6, stroke=color, sw=1.5)
    add(f'<path d="M {x-4:.1f} {y2-7:.1f} L {x:.1f} {y2:.1f} L {x+4:.1f} {y2-7:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>')

def harrow(x1, x2, y, color=FAINT):
    line(x1, y, x2 - 6, y, stroke=color, sw=1.5)
    add(f'<path d="M {x2-7:.1f} {y-4:.1f} L {x2:.1f} {y:.1f} L {x2-7:.1f} {y+4:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>')

def chip(x, y, w, h, text, tcol, tint, bord, size=10, mono=False):
    R(x, y, w, h, r=4, fill=tint, stroke=bord)
    T(x + w / 2, y + h / 2 + size * 0.36, text, size=size, w=600, fill=tcol,
      fam=(MONO if mono else SANS), anchor="middle", upper=True, ls=0.5)

def square_icon(x, y, sz, glyph):
    R(x, y, sz, sz, r=4, fill=SURF2, stroke=BORDER)
    cx, cy = x + sz / 2, y + sz / 2
    if glyph == "list":      # trajectory
        for i, dy in enumerate((-9, -1, 7)):
            circle(cx - 9, cy + dy + 1, 1.6, fill=ACCENT_D)
            line(cx - 4, cy + dy + 1, cx + 9, cy + dy + 1, stroke=ACCENT_D, sw=1.6, cap="round")
    elif glyph == "scissors":  # prefixes / slicing
        for i, dx in enumerate((-7, 0, 7)):
            line(cx + dx, cy - 9, cx + dx, cy + 9, stroke=(ACCENT_D if i == 1 else BORDER_M),
                 sw=(1.8 if i == 1 else 1.4), dash=(None if i == 1 else "2 2"))
    elif glyph == "grid":      # features
        for r_ in range(2):
            for c_ in range(2):
                R(cx - 9 + c_ * 10, cy - 9 + r_ * 10, 7, 7, r=1.5,
                  fill=(ACCENT if (r_ + c_) == 0 else "none"), stroke=ACCENT_D, sw=1.4)
    elif glyph == "gauge":     # model / risk
        add(f'<path d="M {cx-9:.1f} {cy+5:.1f} A 9 9 0 0 1 {cx+9:.1f} {cy+5:.1f}" '
            f'fill="none" stroke="{ACCENT_D}" stroke-width="1.6"/>')
        line(cx, cy + 5, cx + 6, cy - 4, stroke=ACCENT_D, sw=1.6, cap="round")
        circle(cx, cy + 5, 1.8, fill=ACCENT)
    elif glyph == "shield":    # gate
        add(f'<path d="M {cx:.1f} {cy-9:.1f} l 8 3 v 5 q 0 6 -8 9 q -8 -3 -8 -9 v -5 z" '
            f'fill="none" stroke="{ACCENT_D}" stroke-width="1.6" stroke-linejoin="round"/>')
        circle(cx, cy, 1.7, fill=ACCENT)

def section_header(y, num, label, title, desc=None, glyph="list"):
    sz = 46
    square_icon(PAD, y, sz, glyph)
    tx = PAD + sz + 20
    T(tx, y + 16, f"{num} · {label}", size=11, w=700, fill=ACCENT_D, upper=True, ls=1.3)
    T(tx, y + 40, title, size=22, w=300, fill=INK, ls=-0.4)
    if desc:
        T(tx, y + 61, desc, size=13, w=400, fill=GRAY)
        return y + sz + 30
    return y + sz + 12

# action-type chip styling (real action_type values from action_parser.py)
TYPE_STYLE = {
    "search": (INFO, INFO_TINT, "#cdd4fb"),
    "read":   (GRAY, SURF3, BORDER_M),
    "edit":   (WARNING, WARN_TINT, "#f0d9b3"),
    "test":   (ACCENT_D, ACCENT_TINT, ACCENT_LN),
}

# ============================================================================= build
y = 54

# ---- title ---------------------------------------------------------------------
T(PAD, y + 22, "What the monitor reads, and what it predicts", size=31, w=300, fill=INK, ls=-0.7)
y += 50
add(f'<text x="{PAD:.1f}" y="{y+10:.1f}" font-family="{SANS}" font-size="15" letter-spacing="-0.1">'
    f'<tspan font-weight="400" fill="{GRAY}">One agent run, sliced into prefixes, turned into ~40 cheap numbers, scored as a </tspan>'
    f'<tspan font-weight="500" fill="{ACCENT_D}">calibrated failure risk.</tspan></text>')
y += 30
gradline(PAD, W - PAD, y)
y += 34

# ============================================================ STAGE 1 — the trajectory
y = section_header(y, "1", "The input", "A trajectory = one agent run",
                   "A sequence of steps. Each step is a (thought, action, observation) triple; the run ends with one success/fail label.",
                   glyph="list")
py = y
# left panel: the run as a list of steps
lw = 716
ph = 336
R(PAD, py, lw, ph, r=8, fill=SURF1, stroke=BORDER)
T(PAD + 26, py + 30, "the run  ·  instance: psf__requests-1142  ·  agent: swe-agent-llama", size=10.5, w=600, fill=MUTE, fam=MONO)
steps = [
    (1, "search", 'grep -rn "TimeoutError" src/', 'client.py:88: raise TimeoutError'),
    (2, "read",   'cat src/client.py',            'def connect(self): ... TimeoutError'),
    (3, "edit",   'client.py: timeout 5 -> 30',   'edited src/client.py'),
    (4, "test",   'pytest tests/test_client.py',  '1 failed - AssertionError: timed out'),
    (5, "edit",   'client.py: timeout 30 -> 60',  'edited src/client.py'),
    (6, "test",   'pytest tests/test_client.py',  '1 failed - AssertionError: timed out'),
]
rx = PAD + 26
ry0 = py + 54
rh = 32
ry = ry0
act_x = rx + 100
obs_x = rx + 340
for (idx, typ, action, obs) in steps:
    tcol, tint, bord = TYPE_STYLE[typ]
    cy = ry + rh / 2
    T(rx, cy + 4, f"{idx:02d}", size=11, w=700, fill=FAINT, fam=MONO)
    chip(rx + 24, ry + 5, 64, rh - 10, typ, tcol, tint, bord, size=9)
    T(act_x, cy + 4, action, size=11.5, w=500, fill=INK2, fam=MONO)
    T(obs_x - 8, cy + 4, "->", size=11, w=600, fill=FAINT, fam=MONO)
    fillc = ERROR if "failed" in obs else MUTE
    T(obs_x + 12, cy + 4, obs, size=11, w=400, fill=fillc, fam=MONO)
    ry += rh
# amber bracket grouping the edit/test loop (steps 3-6), end ticks point at the rows
bx = rx + 16
btop, bbot = ry0 + 2 * rh + 4, ry0 + 6 * rh - 4
line(bx, btop, bx, bbot, stroke=WARNING, sw=2)
line(bx, btop, bx + 5, btop, stroke=WARNING, sw=2)
line(bx, bbot, bx + 5, bbot, stroke=WARNING, sw=2)
# loop caption (horizontal, in clear space below the steps)
ry = ry0 + 6 * rh
T(rx, ry + 18, "steps 3-6 repeat the same edit / test loop, and the test fail-count never drops",
  size=10.5, w=600, fill=WARNING)
# terminal label badge
by = ry + 30
R(rx, by, 236, 32, r=6, fill=ERROR_T, stroke="#f3b4b4")
circle(rx + 18, by + 16, 4.5, fill=ERROR)
T(rx + 32, by + 20, "TERMINAL LABEL:  FAILED", size=11, w=700, fill=ERROR, fam=MONO)
T(rx + 250, by + 20, "the only outcome-derived value", size=10, w=400, fill=MUTE)

# right panel: anatomy of one step + action-type legend
ax = PAD + lw + 20
aw = CW - lw - 20
R(ax, py, aw, ph, r=8, fill=PAGE, stroke=BORDER)
T(ax + 24, py + 32, "ANATOMY OF A STEP", size=10.5, w=700, fill=INK, upper=True, ls=1.0)
anat = [
    ("thought", "the agent's reasoning", '"find where the timeout is raised"', GRAY),
    ("action", "the command it runs", 'grep -rn "TimeoutError" src/', INK2),
    ("observation", "what the tool returns", 'src/client.py:88: raise ...', GRAY),
]
ayy = py + 62
for (k, desc, val, vc) in anat:
    chip(ax + 24, ayy, 104, 22, k, ACCENT_D, ACCENT_TINT, ACCENT_LN, size=9)
    T(ax + 138, ayy + 16, desc, size=10.5, w=500, fill=MUTE)
    R(ax + 24, ayy + 30, aw - 48, 24, r=4, fill=SURF2, stroke=BORDER_S)
    T(ax + 33, ayy + 45, val, size=10, w=400, fill=vc, fam=MONO)
    ayy += 66
# action-type legend
line(ax + 24, ayy + 4, ax + aw - 24, ayy + 4, stroke=BORDER_S, sw=1)
T(ax + 24, ayy + 28, "ACTION TYPES", size=10, w=700, fill=MUTE, upper=True, ls=1.0)
lx = ax + 24
for typ in ("search", "read", "edit", "test"):
    tcol, tint, bord = TYPE_STYLE[typ]
    cw = 9 + len(typ) * 7.0 + 16
    chip(lx, ayy + 36, cw, 22, typ, tcol, tint, bord, size=9)
    lx += cw + 8
y = py + ph + 26
varrow(PAD + 200, y, y + 22)
T(PAD + 220, y + 16, "slice the run into growing prefixes", size=12, w=500, fill=GRAY)
y += 22 + 22

# ============================================================ STAGE 2 — prefixes + labels
y = section_header(y, "2", "Prefixes & labels", "Each prefix is one training row",
                   "We score growing prefixes (steps 0..k). Every prefix inherits the run's terminal label, even early ones that still look healthy.",
                   glyph="scissors")
py = y
ph = 178
R(PAD, py, CW, ph, r=8, fill=SURF1, stroke=BORDER)
# step ticks
N = 12
tx0 = PAD + 72
tx1 = PAD + 478
tyy = py + 46
T(PAD + 24, tyy + 4, "steps", size=10, w=600, fill=MUTE, upper=True, ls=0.5)
line(tx0, tyy, tx1, tyy, stroke=BORDER_M, sw=1.5)
for i in range(N):
    cx = tx0 + (tx1 - tx0) * i / (N - 1)
    circle(cx, tyy, 3.2, fill=PAGE, stroke=BORDER_M, sw=1.4)
circle(tx1, tyy, 5, fill=ERROR)  # terminal = failed
T(tx1 + 12, tyy + 4, "FAILED", size=10, w=700, fill=ERROR, fam=MONO)
# three prefix bars at k=3,6,9
cuts = [(3, "looks healthy\n(just searching)", False), (6, "loop emerging", True), (9, "clearly flailing", True)]
byy = py + 78
def kx(k): return tx0 + (tx1 - tx0) * (k - 1) / (N - 1)
for (k, note, strong) in cuts:
    R(tx0 - 6, byy - 11, kx(k) - tx0 + 12, 22, r=4, fill=(ACCENT_TINT if strong else SURF3),
      stroke=(ACCENT_LN if strong else BORDER_M))
    T(tx0 + 10, byy + 4, f"prefix(k={k})", size=10, w=600, fill=(ACCENT_D if strong else GRAY), fam=MONO)
    R(kx(k) + 16, byy - 11, 72, 22, r=4, fill=ERROR_T, stroke="#f3b4b4")
    T(kx(k) + 52, byy + 4, "y = FAIL", size=9.5, w=700, fill=ERROR, fam=MONO, anchor="middle")
    note1 = note.split("\n")[0]
    T(kx(k) + 100, byy + 4, note1, size=10, w=400, fill=MUTE)
    byy += 32
# right note
nx = PAD + 580
line(nx - 16, py + 28, nx - 16, py + ph - 24, stroke=BORDER, sw=1)
T(nx, py + 40, "THE LABEL IS NOISY BY DESIGN", size=10.5, w=700, fill=INK, upper=True, ls=0.8)
for i, ln in enumerate([
    "An early prefix of a failed run is labeled",
    "FAIL even though nothing has gone wrong",
    "yet. That irreducible early noise is why",
    "the monitor learns to abstain, not guess.",
]):
    T(nx, py + 62 + i * 18, ln, size=12, w=400, fill=GRAY)
T(nx, py + 62 + 4 * 18 + 8, "prefix(k) = steps[:k]   ·   n_total_steps is NOT a feature", size=10.5, w=500, fill=MUTE, fam=MONO)
y = py + ph + 26
varrow(PAD + 200, y, y + 22)
T(PAD + 220, y + 16, "extract cheap, structured features from the prefix", size=12, w=500, fill=GRAY)
y += 22 + 22

# ============================================================ STAGE 3 — features
y = section_header(y, "3", "What's fed to the model", "~40 cheap numbers, prefix-visible only",
                   "No raw code, no future info. The model never sees the patch, the outcome, or how long the run will be; just these counts and ratios.",
                   glyph="grid")
py = y
# 5 family columns
fam_cols = [
    ("length / pace", [("prefix_step", "6", 0), ("n_actions_seen", "6", 0), ("avg_obs_chars", "64", 0)]),
    ("action counts", [("n_search", "1", 0), ("n_read", "1", 0), ("n_edit", "2", 0), ("n_test", "2", 0), ("search_to_edit", "0.5", 0)]),
    ("file behavior", [("first_edit_step", "3", 0), ("files_edited", "1", 0), ("same_file_edit_max", "2", 1), ("edited_never_read", "0", 0)]),
    ("testing", [("n_test_runs", "2", 0), ("last_fail_count", "1", 0), ("test_fail_delta", "0", 1), ("tests_improving", "0", 1), ("assertion_errors", "2", 1)]),
    ("loop / repetition", [("max_cmd_repeat", "2", 1), ("edit_test_loops", "2", 1), ("same_test_cmd_rep", "1", 1), ("repeated_cmd_l5", "1", 1)]),
]
ncol = len(fam_cols)
cgap = 16
colw = (CW - (ncol - 1) * cgap) / ncol
maxrows = max(len(c[1]) for c in fam_cols)
colh = 40 + maxrows * 26 + 14
for ci, (fam, feats) in enumerate(fam_cols):
    cx = PAD + ci * (colw + cgap)
    R(cx, py, colw, colh, r=8, fill=PAGE, stroke=BORDER)
    R(cx, py, colw, 3, fill=(ACCENT if fam == "loop / repetition" else BORDER_M))
    T(cx + 14, py + 26, fam, size=10.5, w=700, fill=INK, upper=True, ls=0.4)
    fy = py + 46
    for (k, v, hot) in feats:
        R(cx + 10, fy, colw - 20, 22, r=4, fill=(ACCENT_TINT if hot else SURF2), stroke=(ACCENT_LN if hot else BORDER_S))
        T(cx + 18, fy + 15, k, size=9.5, w=400, fill=(ACCENT_D if hot else GRAY), fam=MONO)
        T(cx + colw - 18, fy + 15, v, size=10, w=700, fill=(ACCENT_D if hot else INK2), fam=MONO, anchor="end")
        fy += 26
# legend for highlight
T(PAD, py + colh + 22, "computed from the 6-step prefix above", size=11.5, w=500, fill=GRAY)
chip(PAD + 282, py + colh + 8, 16, 16, "", ACCENT_D, ACCENT_TINT, ACCENT_LN, size=8)
T(PAD + 306, py + colh + 22, "highlighted = no-progress / repetition signals (what drives the risk up here)", size=11.5, w=400, fill=MUTE)
y = py + colh + 22 + 26
varrow(PAD + 200, y, y + 22)
T(PAD + 220, y + 16, "score the feature vector", size=12, w=500, fill=GRAY)
y += 22 + 22

# ============================================================ STAGE 4 — model -> risk
y = section_header(y, "4", "The output", "A calibrated failure probability",
                   "A 1.14 MB gradient-boosted model maps the features to a risk in [0,1], then isotonic calibration makes it an honest probability.",
                   glyph="gauge")
py = y
ph = 116
R(PAD, py, CW, ph, r=8, fill=SURF1, stroke=BORDER)
# flow: features -> model -> calibrate -> risk
bx = PAD + 24
def fbox(x, w, top, bot, featured=False):
    R(x, py + 26, w, 64, r=8, fill=(ACCENT_TINT if featured else PAGE), stroke=(ACCENT_LN if featured else BORDER))
    T(x + w / 2, py + 54, top, size=13, w=500, fill=INK, anchor="middle")
    T(x + w / 2, py + 74, bot, size=10.5, w=400, fill=GRAY, anchor="middle")
fbox(bx, 196, "feature vector", "~40 numbers")
harrow(bx + 196 + 8, bx + 196 + 40, py + 58)
fbox(bx + 244, 214, "HistGradientBoosting", "1.14 MB · CPU", featured=True)
harrow(bx + 244 + 214 + 8, bx + 244 + 214 + 40, py + 58)
fbox(bx + 506, 168, "isotonic calibration", "ECE 0.011")
# risk gauge on the right
gx = bx + 506 + 168 + 36
gw = CW - (gx - PAD) - 24
harrow(bx + 506 + 168 + 8, gx - 6, py + 58)
risk = 0.81
T(gx, py + 40, "P(this run fails)", size=10.5, w=700, fill=MUTE, upper=True, ls=0.6)
R(gx, py + 50, gw, 16, r=8, fill=SURF3)
R(gx, py + 50, gw * risk, 16, r=8, fill=ACCENT)
line(gx + gw * 0.5, py + 46, gx + gw * 0.5, py + 70, stroke=FAINT, sw=1, dash="3 3")
T(gx, py + 84, "0.0", size=9, w=500, fill=FAINT, fam=MONO)
T(gx + gw, py + 84, "1.0", size=9, w=500, fill=FAINT, fam=MONO, anchor="end")
T(gx + gw * risk, py + 44, f"{risk:.2f}", size=13, w=700, fill=ACCENT_D, fam=MONO, anchor="middle")
y = py + ph + 26
varrow(PAD + 200, y, y + 22)
T(PAD + 220, y + 16, "decide whether there's enough evidence to act", size=12, w=500, fill=GRAY)
y += 22 + 22

# ============================================================ STAGE 5 — gate -> verdict
y = section_header(y, "5", "The decision", "A gate that knows when to stay silent",
                   "The monitor only fires when the prefix is deep enough and the risk is far enough from 0.5. Otherwise it abstains.",
                   glyph="shield")
py = y
ph = 108
R(PAD, py, CW, ph, r=8, fill=SURF1, stroke=BORDER)
T(PAD + 26, py + 38, "ABSTENTION GATE", size=11, w=700, fill=ACCENT_D, upper=True, ls=1.0)
T(PAD + 26, py + 64, "commit only when", size=14, w=400, fill=INK)
add(f'<text x="{PAD+160:.1f}" y="{py+64:.1f}" font-family="{MONO}" font-size="13" font-weight="500" fill="{INK}">step ≥ S</text>')
T(PAD + 244, py + 64, "and", size=13, w=400, fill=GRAY)
add(f'<text x="{PAD+277:.1f}" y="{py+64:.1f}" font-family="{MONO}" font-size="13" font-weight="500" fill="{INK}">|risk − 0.5| ≥ C</text>')
T(PAD + 26, py + 86, "here: step 6 and |0.81 − 0.5| = 0.31  →  committed", size=11.5, w=400, fill=GRAY, fam=MONO)
# two verdict chips
chip_w, chip_gap, ch = 184, 16, 70
ox = PAD + CW - 26 - (2 * chip_w + chip_gap)   # 26px right padding inside the panel
def verdict(x, color, tint, kind, top, bot):
    R(x, py + (ph - ch) / 2, chip_w, ch, r=6, fill=tint, stroke=(color if color != MUTE else BORDER_M))
    cx, cyc = x + 24, py + ph / 2
    if kind == "fire":
        add(f'<path d="M {cx:.1f} {cyc-10:.1f} L {cx+9:.1f} {cyc+6:.1f} L {cx-9:.1f} {cyc+6:.1f} Z" fill="{color}"/>')
    else:
        circle(cx, cyc, 8.5, fill="none", stroke=MUTE, sw=1.8)
        line(cx - 5.5, cyc + 5.5, cx + 5.5, cyc - 5.5, stroke=MUTE, sw=1.8, cap="round")
    T(x + 46, cyc - 6, top, size=12.5, w=600, fill=(color if color != MUTE else GRAY))
    T(x + 46, cyc + 15, bot, size=9.5, w=500, fill=MUTE, upper=True, ls=0.4)
verdict(ox, ACCENT_D, ACCENT_TINT, "fire", "Fire warning", "hint / reset / ping")
verdict(ox + chip_w + chip_gap, MUTE, SURF2, "mute", "Stay silent", "too early to tell")
y = py + ph + 18

H = int(y + 30)

# ============================================================================= assemble
frame = []
frame.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAGE}"/>')
frame.append(f'<rect x="6" y="6" width="{W-12}" height="{H-12}" rx="14" fill="none" stroke="{BORDER}" stroke-width="1"/>')
for (cx, cy, dx, dy) in [(22, 22, 1, 1), (W - 22, 22, -1, 1), (22, H - 22, 1, -1), (W - 22, H - 22, -1, -1)]:
    frame.append(f'<path d="M {cx+dx*14:.1f} {cy:.1f} L {cx:.1f} {cy:.1f} L {cx:.1f} {cy+dy*14:.1f}" '
                 f'fill="none" stroke="{BORDER_M}" stroke-width="1.4"/>')
defs = ('<defs><linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{BORDER}" stop-opacity="0"/>'
        f'<stop offset="0.5" stop-color="{BORDER_M}" stop-opacity="1"/>'
        f'<stop offset="1" stop-color="{BORDER}" stop-opacity="0"/>'
        '</linearGradient></defs>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
       f'font-family="{SANS}" role="img" '
       f'aria-label="Hold Your Fire data pipeline: a trajectory of thought-action-observation steps, sliced into prefixes that inherit the run\'s terminal label, turned into ~40 cheap structured features, scored by a 1.14 MB gradient-boosted model into a calibrated failure probability, then passed through an abstention gate that fires or stays silent">'
       + defs + "".join(frame) + "".join(els) + "</svg>\n")

out = Path(__file__).parent / "pipeline.svg"
out.write_text(svg)
print(f"wrote {out}  ({W}x{H}, {len(els)} elements, {len(svg)//1024} KB)")
