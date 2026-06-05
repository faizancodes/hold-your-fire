"""Figures for the MVP / localization results. Reads results/mvp_results.json.

    mech_interp/.venv/bin/python -m mech_interp.plots
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"
ACCENT = "#0a8f76"; ACCENT_D = "#0a8f76"; LOOP_C = "#dc2626"; MUTE = "#8a8a90"; GRAY = "#5c5c60"


def main():
    res = json.loads((OUT / "mvp_results.json").read_text())
    probe = {int(k): v for k, v in res["probe"].items()}
    layers = sorted(probe)
    auc = [probe[L]["diffmeans_auc"] for L in layers]
    sd = [probe[L]["diffmeans_sd"] for L in layers]
    peak = res["peak_layer"]

    # --- Fig A: probe AUC by layer ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(0.5, color=MUTE, ls=":", lw=1, label="chance")
    ax.errorbar(layers, auc, yerr=sd, fmt="o-", color=ACCENT, ms=4, lw=1.5, capsize=2,
                label="diff-of-means probe (scenario-grouped CV)")
    ax.axvline(peak, color=LOOP_C, ls="--", lw=1, alpha=0.7)
    ax.annotate(f"peak L={peak}\nAUC={probe[peak]['diffmeans_auc']:.3f}",
                (peak, probe[peak]["diffmeans_auc"]), color=LOOP_C, fontsize=9,
                xytext=(6, -28), textcoords="offset points")
    ax.set_xlabel("decoder layer (residual stream)"); ax.set_ylabel("loop-vs-progress AUC")
    ax.set_title("Is 'I'm in a loop' linearly decodable? (length-matched contrast)", fontsize=11)
    ax.set_ylim(0.4, 1.02); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_probe_auc.png", dpi=130); plt.close()

    # --- Fig B: steering effect on repeat-preference (with controls) ---
    st = res["steering"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = {"steer_v": (ACCENT, "o-", "steering vector v (progress − loop)"),
              "random": (MUTE, "s--", "random direction (control)"),
              "orthogonal": (GRAY, "^:", "orthogonal direction (control)")}
    for dname, (c, fmt, lab) in styles.items():
        d = st[dname]
        ax.plot(d["alpha"], d["delta_pref_mean"], fmt, color=c, lw=1.5, ms=5, label=lab)
        ax.fill_between(d["alpha"], d["ci_lo"], d["ci_hi"], color=c, alpha=0.12)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("steering strength α (residual-norm units)")
    ax.set_ylabel("Δ repeat-preference\n(negative = less likely to repeat)")
    ax.set_title(f"Steering at L={peak} reduces the loop's repeat-preference", fontsize=11)
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_steering.png", dpi=130); plt.close()

    # --- Fig C: dose-response ---
    dr = res["dose_response"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(dr["K"], dr["loop_proj"], "o-", color=LOOP_C, lw=1.5, label="LOOP (same command ×K)")
    ax.plot(dr["K"], dr["prog_proj"], "o-", color=ACCENT, lw=1.5, label="PROGRESS (K distinct commands)")
    ax.set_xlabel("repeat count K"); ax.set_ylabel(f"projection onto loop direction (L={peak})")
    ax.set_title("Does the loop signal grow with repetition? (dose-response)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_dose.png", dpi=130); plt.close()

    print("wrote fig_probe_auc.png, fig_steering.png, fig_dose.png")


def plot_localize():
    p = OUT / "localize_results.json"
    if not p.exists():
        return
    res = json.loads(p.read_text())
    C = res["contrasts"]
    colors = {"loop_vs_prog": MUTE, "loop_vs_vfail": ACCENT, "vfail_vs_prog": "#d97706"}
    labels = {"loop_vs_prog": "loop vs progress (naive)",
              "loop_vs_vfail": "loop vs varied-fail (genuine repetition)",
              "vfail_vs_prog": "varied-fail vs progress (no-progress)"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for name, c in C.items():
        layers = sorted(int(L) for L in c["by_layer"])
        auc = [c["by_layer"][str(L)]["auc"] for L in layers]
        ax.plot(layers, auc, "o-", ms=3, color=colors[name], lw=1.5, label=labels[name])
        ax.axhline(c["length_only_auc"], color=colors[name], ls=":", lw=1, alpha=0.6)
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("decoder layer"); ax.set_ylabel("diff-of-means AUC (scenario-grouped CV)")
    ax.set_title("Repetition decodability by layer\n(dotted = length-only AUC baseline)", fontsize=10)
    ax.set_ylim(0.4, 1.02); ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.25)
    # length-confound for the genuine contrast
    ab = C["loop_vs_vfail"]["by_layer"]
    layers = sorted(int(L) for L in ab)
    ax2.plot(layers, [ab[str(L)]["len_corr"] for L in layers], "o-", color=ACCENT, ms=3, lw=1.5,
             label="|corr(projection, length)|")
    ax2.axhline(0.5, color=LOOP_C, ls="--", lw=1, label="confound threshold")
    ax2.set_xlabel("decoder layer"); ax2.set_ylabel("|length correlation|")
    ax2.set_title("Length-confound check (loop vs varied-fail)", fontsize=10)
    ax2.set_ylim(0, 1.02); ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_localize.png", dpi=130); plt.close()

    dr = res.get("dose_response")
    if dr:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(dr["K"], dr["loop"], "o-", color=LOOP_C, lw=1.5, label="LOOP (same cmd ×K)")
        ax.plot(dr["K"], dr["vfail"], "o-", color="#d97706", lw=1.5, label="VARIED-FAIL (K diff cmds)")
        ax.plot(dr["K"], dr["prog"], "o-", color=ACCENT, lw=1.5, label="PROGRESS")
        ax.set_xlabel("repeat count K"); ax.set_ylabel(f"projection onto loop−vfail dir (L={dr['dir_layer']})")
        ax.set_title("Controlled dose-response: does the repetition signal grow with K?", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(OUT / "fig_localize_dose.png", dpi=130); plt.close()
    print("wrote fig_localize.png, fig_localize_dose.png")


def plot_steer():
    p = OUT / "steer_eval_results.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    st = r["steering"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    styles = {"best": (ACCENT, "o-", f"−STUCK @ L{r['best_layer']} (steering)"),
              "random": (MUTE, "s--", "random direction"),
              "orthogonal": (GRAY, "^:", "orthogonal direction")}
    for k, (c, fmt, lab) in styles.items():
        if k in st:
            ax.plot(st[k]["alpha"], st[k]["delta_pref"], fmt, color=c, lw=1.6, ms=5, label=lab)
            ax.fill_between(st[k]["alpha"], st[k]["ci_lo"], st[k]["ci_hi"], color=c, alpha=0.12)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("steering strength α"); ax.set_ylabel("Δ repeat-preference\n(negative = less repeating)")
    ax.set_title("Causal steering: a layer-8 direction specifically reduces repetition", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_steer.png", dpi=130); plt.close()
    print("wrote fig_steer.png")


def plot_onpolicy():
    p = OUT / "onpolicy_results.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    esc = r["loop_escape_rate"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    keys = list(esc); vals = [esc[k] for k in keys]
    cols = [ACCENT if k == "steer" else MUTE for k in keys]
    ax.bar(keys, vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_ylabel("loop-escape rate"); ax.set_ylim(0, 1.05)
    ax.set_title(f"On-policy loop escape (α={r['alpha']}, −STUCK@L{r['layer']})", fontsize=11)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_onpolicy.png", dpi=130); plt.close()
    print("wrote fig_onpolicy.png")


def plot_interventions():
    p = OUT / "interventions_results.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    bars = [("baseline", r.get("baseline_escape", 0)),
            ("head\nablation", r.get("ablate_heads_escape", 0)),
            ("steer\nα=16", r.get("steer_escape_a16", 0)),
            ("logit-pen\np=4", r.get("logit_penalty_escape", {}).get("4.0", 0)),
            ("logit-pen\np=8", r.get("logit_penalty_escape", {}).get("8.0", 0))]
    labels = [b[0] for b in bars]; vals = [b[1] for b in bars]
    cols = [ACCENT if v >= 0.99 else MUTE for v in vals]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.bar(labels, vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_ylabel("loop-escape rate (real loops)"); ax.set_ylim(0, 1.08)
    ax.set_title("Only a monitor-gated logit penalty breaks the real loops", fontsize=11)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_interventions.png", dpi=130); plt.close()
    print("wrote fig_interventions.png")


def plot_frontier():
    p = OUT / "frontier_results.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())["interventions"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    nice = {"none": "no intervention", "gated_targeted": "targeted penalty, gated (ours)",
            "alwayson_targeted": "targeted penalty, always-on", "alwayson_rep_pen": "always-on repetition penalty",
            "alwayson_norepeat3": "always-on no-repeat-3gram", "steering": "−STUCK steering"}
    off = {"none": (8, -4), "gated_targeted": (10, -16), "alwayson_targeted": (10, 8),
           "alwayson_rep_pen": (-6, 10), "alwayson_norepeat3": (0, 12), "steering": (-6, -16)}
    for name, v in r.items():
        x, y = v["disruption_prodrepeat"], v["efficacy_escape"]
        best = name == "gated_targeted"
        ax.scatter(x, y, s=170 if best else 80, color=ACCENT if "targeted" in name else MUTE,
                   edgecolor="k" if best else "none", zorder=3, marker="*" if best else "o")
        ha = "right" if off[name][0] < 0 else "left"
        ax.annotate(nice.get(name, name), (x, y), fontsize=8.5, ha=ha,
                    xytext=off[name], textcoords="offset points",
                    color=ACCENT_D if "targeted" in name else GRAY,
                    fontweight="bold" if best else "normal")
    ax.set_xlabel("disruption of productive repetition  (lower = better)")
    ax.set_ylabel("loop-escape on real loops  (higher = better)")
    ax.set_title("Only a TARGETED (monitor-identified) penalty breaks loops without disruption", fontsize=10.5)
    ax.set_xlim(-0.07, 1.05); ax.set_ylim(-0.08, 1.12)
    ax.text(0.03, 0.90, "ideal: top-left", fontsize=8.5, color=ACCENT_D, style="italic")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_frontier.png", dpi=130); plt.close()
    print("wrote fig_frontier.png")


def plot_outcome():
    ex = OUT / "outcome_exempt.json"
    if not ex.exists():
        return
    r = json.loads(ex.read_text())["rows"]
    def rate(mode, key):
        rs = [x for x in r if x["mode"] == mode]
        return sum((x[key] > 0 if key != "solved" else x["solved"]) for x in rs) / max(len(rs), 1)
    conds = ["control", "targeted", "aggressive"]
    broke = [0.0, rate("treatment", "n_escape"), 1.0]   # control never intervenes; aggr escaped all
    solved = [rate("control", "solved"), rate("treatment", "solved"), 0.0]
    import numpy as np
    x = np.arange(len(conds)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w / 2, broke, w, color=ACCENT, label="loop broken (escaped)")
    ax.bar(x + w / 2, solved, w, color=LOOP_C, label="task solved")
    for i in range(len(conds)):
        ax.text(i - w / 2, broke[i] + 0.02, f"{broke[i]:.0%}", ha="center", fontsize=9)
        ax.text(i + w / 2, solved[i] + 0.02, f"{solved[i]:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate over 6 solvable bug-fix tasks")
    ax.set_title("Breaking the loop ≠ recovering: un-sticking works, success doesn't (1.5B)", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper center"); ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_outcome.png", dpi=130); plt.close()
    print("wrote fig_outcome.png")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "outcome":
        plot_outcome()
    elif arg == "frontier":
        plot_frontier()
    elif arg == "interventions":
        plot_interventions()
    elif arg == "localize":
        plot_localize()
    elif arg == "steer":
        plot_steer()
    elif arg == "onpolicy":
        plot_onpolicy()
    elif arg == "all":
        plot_localize(); plot_steer(); plot_onpolicy(); plot_interventions(); plot_frontier()
    else:
        main()
