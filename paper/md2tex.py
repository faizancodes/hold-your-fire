#!/usr/bin/env python3
"""Faithful markdown -> LaTeX body converter for paper_draft.md (no pandoc available).
Reads the real text (nothing paraphrased); emits paper/body.tex. main.tex \\input's it.
"""
import re, sys, pathlib

SRC = pathlib.Path(__file__).parent / "paper_draft.md"
OUT = pathlib.Path(__file__).parent / "body.tex"

# [Author, Year] -> bibkey   (keys must match refs.bib)
CITE = {
    "Jimenez et al., 2024": "jimenez2024swebench", "Yang et al., 2024": "yang2024sweagent",
    "Wang et al., 2024": "wang2024openhands", "Lightman et al., 2023": "lightman2023verify",
    "Chow, 1970": "chow1970reject", "El-Yaniv & Wiener, 2010": "elyaniv2010foundations",
    "Geifman & El-Yaniv, 2017": "geifman2017selectivenet", "Madras et al., 2018": "madras2018predict",
    "Mozannar & Sontag, 2020": "mozannar2020consistent", "Guo et al., 2017": "guo2017calibration",
    "Zadrozny & Elkan, 2002": "zadrozny2002transforming", "Shimodaira, 2000": "shimodaira2000improving",
    "Sun et al., 2016": "sun2016return", "Zheng et al., 2023": "zheng2023judging",
}
UNI = {  # unicode -> latex
    "→": r"$\to$", "≫": r"$\gg$", "≪": r"$\ll$", "≈": r"$\approx$", "×": r"$\times$",
    "⊕": r"$\oplus$", "≤": r"$\le$", "≥": r"$\ge$", "±": r"$\pm$", "§": r"\S",
    "—": "---", "–": "--", "’": "'", "‘": "'", "“": "``", "”": "''", "…": r"\dots",
    "⁴": r"$^4$", "↑": r"$\uparrow$", "↓": r"$\downarrow$", "•": r"\textbullet ",
}
SPECIAL = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "$": r"\$",
           "{": r"\{", "}": r"\}", "~": r"$\sim$", "^": r"\^{}"}
UNI["−"] = r"$-$"            # U+2212 minus
UNI["Δ"] = r"$\Delta$"       # U+0394
UNI["⇒"] = r"$\Rightarrow$"  # U+21D2
CODE_UNI = {"→": "->", "≥": ">=", "≤": "<=", "−": "-", "×": "x", "≫": ">>",
            "≪": "<<", "≈": "~", "±": "+-", "⊕": "(+)", "§": "S",
            "Δ": "Delta", "…": "...", "⇒": "=>"}

def code_ascii(t):  # unicode -> ASCII inside \texttt code spans
    for k, v in CODE_UNI.items():
        t = t.replace(k, v)
    return t

def esc(t):  # escape LaTeX specials in plain prose
    return "".join(SPECIAL.get(c, c) for c in t)

def uni(t):
    for k, v in UNI.items():
        t = t.replace(k, v)
    return t

def inline(t):
    holes = []
    def stash(m, wrap):
        holes.append(wrap); return f"\0{len(holes)-1}\0"
    # protect inline code  `code`  (break `--` ligature so CLI flags/paths render as literal hyphens,
    # not an en-dash; the empty group is inserted after esc() so the braces stay real TeX grouping)
    t = re.sub(r"`([^`]+)`",
               lambda m: stash(m, r"\texttt{" + re.sub(r"-(?=-)", "-{}", esc(code_ascii(m.group(1)))) + "}"), t)
    # citations [A; B]  (split on ;)
    def cite(m):
        keys = []
        for part in m.group(1).split(";"):
            part = part.strip().rstrip(".")
            part = re.sub(r",?\s*`?\[?verify\]?`?", "", part).strip()
            keys.append(CITE.get(part, "VERIFY:" + part))
        return stash(m, r"\citep{" + ",".join(keys) + "}")
    t = re.sub(r"\[([^\]]*(?:1[0-9]{3}|20[0-9]{2})[^\]]*)\]", cite, t)
    # escape specials, then emphasis, then unicode
    t = esc(t)
    # bold first (non-greedy, allowing inner *italic*); then remaining italics
    t = re.sub(r"\*\*(.+?)\*\*",
               lambda m: r"\textbf{" + re.sub(r"\*([^*]+)\*", r"\\emph{\1}", m.group(1)) + "}", t)
    t = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", t)
    t = re.sub(r'"([^"]*)"', r"``\1''", t)   # straight quotes -> LaTeX paired quotes
    t = uni(t)
    for i, h in enumerate(holes):
        t = t.replace(f"\0{i}\0", h)
    return t

def heading(line):
    m = re.match(r"^(#{2,4})\s+(.*)$", line)
    lvl, txt = len(m.group(1)), m.group(2).strip()
    numbered = bool(re.match(r"^\d+(\.\d+)*\.?\s", txt))  # "1. " / "4.1 " -> auto-number
    txt = re.sub(r"^\d+(\.\d+)*\.?\s+", "", txt)          # strip the manual number
    txt = re.sub(r'^"(.*)"$', r"\1", txt)
    cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[lvl]
    star = "" if numbered else "*"                        # unnumbered: Related work / Figures / Reproducibility
    return f"\\{cmd}{star}{{{inline(txt)}}}"

def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r":?-+:?", x or "-") for x in c)]
    ncol = max(len(r) for r in cells)
    # auto-shrink a too-wide table to the line width, never enlarge a narrow one
    out = ["\\begin{center}\\small",
           "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
           "\\begin{tabular}{" + "l" * ncol + "}", "\\toprule"]
    for i, r in enumerate(cells):
        r = r + [""] * (ncol - len(r))
        out.append(" & ".join(inline(x) for x in r) + r" \\")
        if i == 0:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{center}"]
    return "\n".join(out)

def main():
    lines = SRC.read_text().splitlines()
    # find body start (after the title block + first ---), and the abstract
    i = 0
    out = []
    in_abstract = False
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        # skip title block (handled in main.tex) until first "## "
        if not out and not s.startswith("## "):
            i += 1; continue
        if re.match(r"^#{2,4}\s", s):
            title = re.sub(r"^#{2,4}\s+", "", s)
            if title.lower().startswith("abstract"):
                out.append("\\begin{abstract}"); in_abstract = True; i += 1; continue
            if in_abstract:
                out.append("\\end{abstract}"); in_abstract = False
            if title.lower().startswith("figures"):   # handled specially below
                out.append(heading(s)); i += 1
                while i < len(lines):
                    fs = lines[i].strip()
                    if re.match(r"^#{2,4}\s", fs):      # next top-level section -> stop
                        break
                    if not fs:                          # skip blank lines
                        i += 1; continue
                    fm = re.match(r"^!\[[^\]]*\]\(([^)]+)\)", fs)
                    if fm:
                        path = fm.group(1)              # already ../results/figures/...
                        i += 1
                        capl = []                       # gather multi-line caption until blank/next image
                        while i < len(lines) and lines[i].strip() and not re.match(r"^!\[", lines[i].strip()):
                            capl.append(lines[i].strip()); i += 1
                        cap = " ".join(capl)
                        cap = re.sub(r"^\*\*Figure\s+\d+\.\s*", "**", cap)  # drop redundant label (LaTeX auto-numbers)
                        out += ["\\begin{figure}[t]\\centering",
                                f"\\includegraphics[width=.82\\linewidth]{{{path}}}",
                                f"\\caption{{{inline(cap)}}}", "\\end{figure}"]
                    else:                               # intro italic note
                        out.append(inline(fs)); i += 1
                continue
            out.append(heading(s)); i += 1; continue
        if s == "---" or s == "":
            i += 1; continue
        # table
        if s.startswith("|"):
            tb = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tb.append(lines[i]); i += 1
            out.append(table(tb)); continue
        # numbered list
        if re.match(r"^\d+\.\s", s):
            out.append("\\begin{enumerate}")
            while i < len(lines) and re.match(r"^\s*\d+\.\s", lines[i]):
                item = re.sub(r"^\s*\d+\.\s", "", lines[i]); i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r"^\s*[\d\-]", lines[i]) and not lines[i].strip().startswith("|"):
                    item += " " + lines[i].strip(); i += 1
                out.append("\\item " + inline(item.strip()))
            out.append("\\end{enumerate}"); continue
        # bullet list
        if re.match(r"^[-*]\s", s):
            out.append("\\begin{itemize}")
            while i < len(lines) and re.match(r"^\s*[-*]\s", lines[i]):
                item = re.sub(r"^\s*[-*]\s", "", lines[i]); i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r"^\s*[-*]\s", lines[i]) and not lines[i].strip().startswith("|") and not lines[i].strip().startswith("#"):
                    item += " " + lines[i].strip(); i += 1
                out.append("\\item " + inline(item.strip()))
            out.append("\\end{itemize}"); continue
        # paragraph (gather until blank)
        para = [s]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^[#|>]|^[-*]\s|^\d+\.\s|^---$|^!\[", lines[i].strip()):
            para.append(lines[i].strip()); i += 1
        out.append(inline(" ".join(para)))
        out.append("")
    if in_abstract:
        out.append("\\end{abstract}")
    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT} ({len(out)} blocks)")
    # report unresolved citations
    bad = sorted(set(re.findall(r"VERIFY:[^,}]+", OUT.read_text())))
    if bad: print("UNMAPPED CITATIONS:", bad)

main()
