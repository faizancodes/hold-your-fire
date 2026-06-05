#!/usr/bin/env python3
"""Generate assets/benchmark.svg — a Hold-Your-Fire-styled headline card for social: the 1 MB
CPU monitor vs frontier LLM judges at predicting which coding-agent runs will fail.

Numbers are verified from the committed result files (results/offline/small/gpt_judge*.json,
ollama_judge.json) and the paper's Table in §4.2, all on the SAME 200 held-out prefixes (50/50):
  structured classifier (1.14 MB, CPU) ... ROC AUC 0.768   (subset; 0.722 on the full test set)
  Claude Opus 4.8  (best of thinking sweep) 0.639
  GPT-5.5          (best of effort sweep)    0.626
  qwen2.5-coder 7B (local)                    0.559
LLMs judged zero-shot from one shared prompt; the classifier is trained for this single task.
Honest framing is the point: the qualifier ("at spotting failing runs") and the zero-shot vs
trained disclosure are on the card, so it is defensible, not hype.

16:9 (1600x900) for in-feed display on X. Pure-stdlib. Re-run: python3 assets/make_benchmark_svg.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "benchmark.svg"

PAGE = "#ffffff"
BORDER = "#e4e4e6"; BORDER_M = "#d8d8dc"
INK = "#0a0a0a"; INK2 = "#2c2c2e"; GRAY = "#5c5c60"; MUTE = "#8a8a90"; FAINT = "#c2c2c8"
ACCENT = "#00e4b4"; ACCENT_D = "#0a8f76"; ACCENT_DK = "#0a6b58"; ACCENT_LN = "#9be8d5"; ACCENT_TINT = "#ecfbf6"
BAR_GRAY = "#cccdd3"; BAR_GRAY_D = "#9a9aa2"
SANS = "'Geist','Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,system-ui,sans-serif"
MONO = "'Geist Mono','SF Mono',ui-monospace,'JetBrains Mono','Roboto Mono',Menlo,Consolas,monospace"

# verified, source files in the docstring; LLMs shown at their BEST swept config (steelman)
BARS = [
    {"name": "Hold Your Fire",   "tag": "1.14 MB · CPU",   "auc": 0.768, "hero": True},
    {"name": "Claude Opus 4.8",  "tag": "frontier · API",  "auc": 0.639, "hero": False},
    {"name": "GPT-5.5",          "tag": "frontier · API",  "auc": 0.626, "hero": False},
    {"name": "qwen2.5-coder 7B", "tag": "local",           "auc": 0.559, "hero": False},
]

W, H = 1600, 900
PAD = 64
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

# ============================================================ header
add(f'<rect x="{PAD}" y="54" width="8" height="8" fill="{ACCENT}"/>')
T(PAD + 18, 62, "Hold Your Fire · failure-prediction benchmark", size=13, w=700, fill=ACCENT_D, upper=True, ls=1.2)
T(PAD, 116, "A 1 MB model out-predicts GPT-5.5 and Claude Opus 4.8", size=37, w=360, fill=INK)
T(PAD, 160, "at spotting failing AI coding-agent runs", size=37, w=360, fill=INK)
T(PAD, 198, "Predicting which agent runs will eventually fail. Higher bar = better (ROC AUC, on the same 200 held-out runs).",
  size=17, w=400, fill=GRAY)
T(PAD, 223, "The frontier models judge zero-shot from one shared prompt; the 1 MB model is trained for this one job.",
  size=17, w=400, fill=GRAY)
line(PAD, 250, W - PAD, 250, stroke="url(#fade)", sw=1)

# ============================================================ bar chart (left)
BX0, BX1 = 168, 928
CTY, CBY = 332, 652                                            # 0.8 at CTY, 0.5 at CBY
PXU = (CBY - CTY) / 0.30                                       # px per AUC unit
def gy(a): return CBY - (a - 0.5) * PXU

# axis gridlines + labels
T(132, CTY - 14, "ROC AUC", size=11, w=600, fill=MUTE, anchor="end", upper=True, ls=0.5)
for a in (0.5, 0.6, 0.7, 0.8):
    yy = gy(a)
    line(150, yy, BX1, yy, stroke=(GRAY if a == 0.5 else BORDER), sw=(1.2 if a == 0.5 else 1),
         opacity=(1 if a == 0.5 else 0.7))
    T(138, yy + 4, f"{a:.1f}", size=12, w=500, fill=MUTE, fam=MONO, anchor="end")
T(138, CBY + 19, "chance", size=10, w=500, fill=MUTE, anchor="end")

N = len(BARS)
BW = 122
GAP = (BX1 - BX0 - N * BW) / (N - 1)
x = BX0
for b in BARS:
    top = gy(b["auc"]); cx = x + BW / 2
    fill = ACCENT if b["hero"] else BAR_GRAY
    vcol = ACCENT_DK if b["hero"] else BAR_GRAY_D
    R(x, top, BW, CBY - top, r=5, fill=fill)
    if b["hero"]:
        R(x, top, BW, CBY - top, r=5, fill="none", stroke=ACCENT_D, sw=1.4)
    T(cx, top - 14, f"{b['auc']:.3f}", size=27, w=700, fill=vcol, fam=MONO, anchor="middle")
    T(cx, CBY + 32, b["name"], size=17, w=700, fill=(INK if b["hero"] else INK2), anchor="middle")
    T(cx, CBY + 52, b["tag"], size=12.5, w=500, fill=(ACCENT_D if b["hero"] else MUTE), anchor="middle")
    x += BW + GAP

# ============================================================ "the twist" card (right)
KX, KY, KW, KH = 1012, 300, W - PAD - 1012, 352
R(KX, KY, KW, KH, r=18, fill=ACCENT_TINT, stroke=ACCENT_LN, sw=1)
kx = KX + 30
T(kx, KY + 40, "And yet it is tiny and free", size=15, w=700, fill=ACCENT_D, upper=True, ls=0.8)
stats = [
    ("1.14 MB", "on disk — versus gigabytes of weights"),
    ("≈10,000×", "cheaper per prediction, batched"),
    ("$0 · CPU", "no GPU, no API, runs on a laptop"),
]
yy = KY + 92
for big, lab in stats:
    T(kx, yy, big, size=33, w=700, fill=INK)
    T(kx + 200, yy, lab, size=14.5, w=400, fill=GRAY)
    yy += 70
line(kx, KY + 290, KX + KW - 30, KY + 290, stroke=ACCENT_LN, sw=1)
T(kx, KY + 322, "Flags a failing run ~13 steps early, and calibrated (ECE 0.011).", size=14, w=500, fill=INK2)

# ============================================================ honest footer
line(PAD, 792, W - PAD, 792, stroke="url(#fade)", sw=1)
T(PAD, 818, "ROC AUC on 200 held-out agent runs (50/50); identical prompt for all, frontier LLMs at their best swept reasoning "
  "config. The 1 MB model leads by +0.13 to +0.21 AUC (paired bootstrap, all CIs exclude 0).",
  size=13.5, w=400, fill=MUTE)

# ============================================================ assemble
defs = (
    '<defs><linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
    f'<stop offset="0" stop-color="{BORDER}" stop-opacity="0"/>'
    f'<stop offset="0.5" stop-color="{BORDER}" stop-opacity="1"/>'
    f'<stop offset="1" stop-color="{BORDER}" stop-opacity="0"/></linearGradient></defs>'
)
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'font-family="{SANS}" role="img" '
    f'aria-label="Bar chart: at predicting which AI coding-agent runs will fail, a 1.14 MB CPU model scores ROC AUC '
    f'0.768 on 200 held-out runs, beating Claude Opus 4.8 (0.639), GPT-5.5 (0.626), and a local 7B model (0.559). The '
    f'frontier models are judged zero-shot from one shared prompt; the small model is trained for this single task. It is '
    f'about 10,000 times cheaper to run, 1.14 MB on disk, and runs on a laptop CPU for free.">'
    + defs + f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAGE}"/>' + "".join(els) + "</svg>\n"
)
OUT.write_text(svg)
print(f"wrote {OUT}  ({W}x{H}, {len(els)} elements, {len(svg)//1024} KB)")
