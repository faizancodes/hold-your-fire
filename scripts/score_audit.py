#!/usr/bin/env python3
"""Score the blind human-annotated audit vs the heuristic labeler (#2 strengthening).

Joins results/audits/human_labels.jsonl (author-annotated, blind to the heuristic
label) with human_key.jsonl (kind + heuristic_failure_mode), and reports agreement,
per-mode distributions on flagged vs missed failures, and the looping
precision/recall confusion that tests the "loops dominate" claim.
"""

from __future__ import annotations

from collections import Counter

import _bootstrap  # noqa: F401

from localguard.utils import RESULTS_AUDITS, read_jsonl, write_json


def _dist(rows, key):
    c = Counter(r[key] for r in rows)
    n = max(1, len(rows))
    return {m: f"{cnt} ({cnt/n:.0%})" for m, cnt in c.most_common()}


def main() -> None:
    labels = {r["id"]: r for r in read_jsonl(RESULTS_AUDITS / "human_labels.jsonl")}
    key = {r["id"]: r for r in read_jsonl(RESULTS_AUDITS / "human_key.jsonl")}
    rows = []
    for aid, k in key.items():
        h = labels.get(aid)
        if not h:
            continue
        rows.append({"id": aid, "kind": k["kind"], "heur": k["heuristic_failure_mode"],
                     "human": h["human_mode"], "agree": h["human_mode"] == k["heuristic_failure_mode"]})

    flagged = [r for r in rows if r["kind"] == "flagged_failure"]
    missed = [r for r in rows if r["kind"] == "missed_failure"]

    # "repetition family" = looping OR patch_churn (both are repeat/no-progress modes)
    REP = {"looping", "patch_churn"}
    def rep_rate(rs, key): return sum(1 for r in rs if r[key] in REP) / max(1, len(rs))

    out = {
        "n_total": len(rows), "n_flagged": len(flagged), "n_missed": len(missed),
        "exact_agreement_all": round(sum(r["agree"] for r in rows) / len(rows), 3),
        "exact_agreement_flagged": round(sum(r["agree"] for r in flagged) / max(1, len(flagged)), 3),
        "human_dist_flagged": _dist(flagged, "human"),
        "heuristic_dist_flagged": _dist(flagged, "heur"),
        "human_dist_missed": _dist(missed, "human"),
        "heuristic_dist_missed": _dist(missed, "heur"),
        # the "loops dominate" claim, three ways
        "flagged_looping_heuristic": round(sum(r["heur"] == "looping" for r in flagged) / max(1, len(flagged)), 3),
        "flagged_looping_human": round(sum(r["human"] == "looping" for r in flagged) / max(1, len(flagged)), 3),
        "flagged_repetition_family_human": round(rep_rate(flagged, "human"), 3),
        "flagged_repetition_family_heuristic": round(rep_rate(flagged, "heur"), 3),
    }

    # coarse agreement: collapse fine modes into families (disagreement is mostly
    # looping-vs-churn WITHIN the repetition family, not wild misclassification)
    FAM = {"looping": "repetition", "patch_churn": "repetition",
           "insufficient_context": "context", "wrong_file": "context",
           "test_neglect": "premature", "submission_too_early": "premature",
           "environment_distraction": "environment", "not_observable": "not_observable"}
    out["coarse_agreement_all"] = round(
        sum(FAM.get(r["human"]) == FAM.get(r["heur"]) for r in rows) / len(rows), 3)
    out["coarse_agreement_flagged"] = round(
        sum(FAM.get(r["human"]) == FAM.get(r["heur"]) for r in flagged) / max(1, len(flagged)), 3)

    # looping precision/recall (human as ground truth)
    heur_loop = [r for r in rows if r["heur"] == "looping"]
    human_loop = [r for r in rows if r["human"] == "looping"]
    tp = sum(1 for r in heur_loop if r["human"] == "looping")
    out["looping_precision_heuristic"] = round(tp / max(1, len(heur_loop)), 3)
    out["looping_recall_heuristic"] = round(tp / max(1, len(human_loop)), 3)
    # when the heuristic says "looping" but human disagrees, what is it really?
    out["heuristic_looping_really"] = _dist(heur_loop, "human")
    # observability: what fraction of MISSED failures were genuinely not observable (human)?
    out["missed_not_observable_human"] = round(
        sum(1 for r in missed if r["human"] in {"not_observable", "insufficient_context"}) / max(1, len(missed)), 3)

    write_json(RESULTS_AUDITS / "human_validation.json", out)
    print("=== Human-validated audit (N=%d: %d flagged, %d missed) ===" % (out["n_total"], out["n_flagged"], out["n_missed"]))
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
