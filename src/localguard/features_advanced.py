"""Advanced prefix features: temporal dynamics + semantic signals (v2).

Built strictly on top of the v1 ``extract_features`` (so the v2 set is a
*superset* — a clean controlled comparison) and computed from prefix steps 0..t
ONLY. Adds six families that capture what cumulative counts miss:

  adv_w*    recency windows (last 5/10 steps)
  adv_ts_*  staleness / time-since-last-X
  adv_seq_* sequence structure (entropy, compression, n-gram repeats, similarity)
  adv_err_* semantic error categories (from the agent's own command output)
  adv_flow_* workflow / progress signals
  adv_ix_*  targeted interactions

None of these read ``eval_logs``, the final patch, the target, or any step after
t — verified by the leakage + future-invariance tests.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Sequence

from .features import extract_features
from .schemas import StepEvent

# Semantic error/outcome categories — instance-agnostic structural signals that
# generalize across held-out instances (unlike raw TF-IDF, which overfits names).
_SEMANTIC_PATTERNS: dict[str, re.Pattern] = {
    "cmd_not_found": re.compile(r"command not found|not found:|: not found", re.I),
    "no_such_file": re.compile(r"No such file or directory|cannot find|does not exist", re.I),
    "no_matches": re.compile(r"No matches found|no matches for|0 matches|not found in", re.I),
    "permission": re.compile(r"Permission denied|Operation not permitted", re.I),
    "syntax": re.compile(r"\bSyntaxError\b|\bIndentationError\b|invalid syntax", re.I),
    "import_err": re.compile(r"\bImportError\b|\bModuleNotFoundError\b|cannot import name", re.I),
    "name_err": re.compile(r"\bNameError\b|is not defined", re.I),
    "type_err": re.compile(r"\bTypeError\b"),
    "attr_err": re.compile(r"\bAttributeError\b"),
    "key_index_err": re.compile(r"\bKeyError\b|\bIndexError\b"),
    "value_err": re.compile(r"\bValueError\b"),
    "assertion": re.compile(r"\bAssertionError\b|assert "),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "usage_err": re.compile(r"\busage:|unrecognized arguments|invalid choice|unexpected argument", re.I),
    "timeout": re.compile(r"timed out|timeout|TimeoutError", re.I),
    "test_fail_tok": re.compile(r"\bFAILED\b|\bFAIL\b|\bERROR\b"),
}

ADV_FAMILY_PREFIXES = {
    "adv_windows": "adv_w",
    "adv_timesince": "adv_ts_",
    "adv_sequence": "adv_seq_",
    "adv_errors": "adv_err_",
    "adv_workflow": "adv_flow_",
    "adv_interactions": "adv_ix_",
}


def _norm_cmd(step: StepEvent) -> str:
    base = (step.command or step.action_text or "").strip()
    return re.sub(r"\s+", " ", base)[:160]


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _compression_ratio(text: str) -> float:
    """zlib-compressed size / raw size; low ⇒ highly repetitive (stuck)."""
    if not text:
        return 1.0
    raw = text.encode("utf-8", "ignore")[:20000]
    if not raw:
        return 1.0
    comp = zlib.compress(raw, 6)
    return len(comp) / max(1, len(raw))


def _ngram_repeats(seq: Sequence[str], n: int) -> int:
    if len(seq) < n:
        return 0
    grams = Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    return sum(c - 1 for c in grams.values() if c > 1)


def _longest_run(seq: Sequence[str]) -> int:
    best = run = 0
    prev = None
    for x in seq:
        run = run + 1 if x == prev else 1
        best = max(best, run)
        prev = x
    return best


def _consec_cmd_similarity(cmds: Sequence[str], k: int = 8) -> float:
    pairs = [(cmds[i], cmds[i + 1]) for i in range(max(0, len(cmds) - k - 1), len(cmds) - 1)]
    if not pairs:
        return 0.0
    sims = [SequenceMatcher(None, a, b).ratio() for a, b in pairs if a or b]
    return sum(sims) / len(sims) if sims else 0.0


def _steps_since(flags: list[bool], n: int) -> int:
    """Steps since the last True flag; sentinel ``n`` if never (longer is worse)."""
    for back, val in enumerate(reversed(flags)):
        if val:
            return back
    return n


def extract_features_advanced(steps: Sequence[StepEvent]) -> dict[str, Any]:
    """v1 features ∪ advanced temporal/semantic families. Prefix-only."""
    f: dict[str, Any] = dict(extract_features(steps))
    n = len(steps)
    if n == 0:
        return f

    types = [s.action_type for s in steps]
    cmds = [_norm_cmd(s) for s in steps]
    obs = [(s.observation_text or "") for s in steps]
    obs_lens = [len(o) for o in obs]
    is_err = [bool(s.contains_traceback or s.contains_exception) for s in steps]
    is_test = [t == "test" for t in types]
    is_edit = [t == "edit" for t in types]
    is_search = [t == "search" for t in types]

    # ---- recency windows -------------------------------------------------
    for W in (5, 10):
        wl = min(W, n)
        wt, wc, we, wo = types[-W:], cmds[-W:], is_err[-W:], obs_lens[-W:]
        f[f"adv_w{W}_n_edit"] = sum(1 for t in wt if t == "edit")
        f[f"adv_w{W}_n_test"] = sum(1 for t in wt if t == "test")
        f[f"adv_w{W}_edit_frac"] = f[f"adv_w{W}_n_edit"] / wl
        f[f"adv_w{W}_test_frac"] = f[f"adv_w{W}_n_test"] / wl
        f[f"adv_w{W}_distinct_cmd"] = len({c for c in wc if c})
        f[f"adv_w{W}_repeat_ratio"] = 1.0 - (f[f"adv_w{W}_distinct_cmd"] / wl)
        f[f"adv_w{W}_err_frac"] = sum(we) / wl
        f[f"adv_w{W}_obs_mean"] = sum(wo) / wl

    # ---- staleness / time-since -----------------------------------------
    # progress = read a not-previously-seen file, OR a test with fewer fails
    seen_files: set[str] = set()
    progress_flags: list[bool] = []
    new_file_flags: list[bool] = []
    prev_fail: int | None = None
    for s in steps:
        new_file = False
        if s.action_type == "read":
            for p in s.file_paths:
                if p not in seen_files:
                    new_file = True
            for p in s.file_paths:
                seen_files.add(p)
        improved = False
        if s.is_test_command and s.test_fail_count is not None:
            if prev_fail is not None and s.test_fail_count < prev_fail:
                improved = True
            prev_fail = s.test_fail_count
        new_file_flags.append(new_file)
        progress_flags.append(new_file or improved)

    f["adv_ts_test"] = _steps_since(is_test, n)
    f["adv_ts_edit"] = _steps_since(is_edit, n)
    f["adv_ts_search"] = _steps_since(is_search, n)
    f["adv_ts_error"] = _steps_since(is_err, n)
    f["adv_ts_new_file"] = _steps_since(new_file_flags, n)
    f["adv_ts_progress"] = _steps_since(progress_flags, n)

    # ---- sequence structure ---------------------------------------------
    f["adv_seq_entropy"] = _entropy(Counter(types))
    f["adv_seq_entropy_last10"] = _entropy(Counter(types[-10:]))
    f["adv_seq_compress"] = _compression_ratio("\n".join(cmds))
    f["adv_seq_compress_last10"] = _compression_ratio("\n".join(cmds[-10:]))
    f["adv_seq_bigram_repeat"] = _ngram_repeats(types, 2)
    f["adv_seq_trigram_repeat"] = _ngram_repeats(cmds, 3)
    f["adv_seq_longest_run"] = _longest_run(types)
    f["adv_seq_consec_cmd_sim"] = _consec_cmd_similarity(cmds)
    # behavior narrowing: did action diversity collapse over time?
    half = max(1, n // 2)
    f["adv_seq_entropy_drop"] = _entropy(Counter(types[:half])) - _entropy(Counter(types[-10:]))
    # stuck-on-same-output: repeated identical observations (strong loop signal)
    obs_keys = [o.strip()[:500] for o in obs]
    obs_counts = Counter(k for k in obs_keys if k)
    f["adv_seq_obs_repeat_max"] = max(obs_counts.values()) if obs_counts else 0
    recent_obs_keys = [k for k in obs_keys[-8:] if k]
    f["adv_seq_obs_repeat_recent"] = len(recent_obs_keys) - len(set(recent_obs_keys))

    # ---- semantic error categories --------------------------------------
    # Bounded scan: only the most recent observations (each capped), so cost is
    # O(1) per prefix instead of O(n*total_obs). Recent errors matter most for
    # early warning anyway.
    window_obs = "\n".join(o[:1500] for o in obs[-12:])
    for name, pat in _SEMANTIC_PATTERNS.items():
        f[f"adv_err_{name}"] = len(pat.findall(window_obs))
    f["adv_err_streak"] = _steps_since([not e for e in is_err], n) if is_err and is_err[-1] else 0
    f["adv_err_recent_total"] = sum(is_err[-5:])
    f["adv_err_rate"] = sum(is_err) / n
    # stuck-on-same-error: max repeat of an identical first error line
    err_lines = Counter()
    for o, e in zip(obs, is_err):
        if e:
            line = o.strip().splitlines()[0][:120] if o.strip() else ""
            if line:
                err_lines[line] += 1
    f["adv_err_same_repeated"] = max(err_lines.values()) if err_lines else 0

    # ---- workflow / progress --------------------------------------------
    n_edit = sum(is_edit)
    n_test = sum(is_test)
    f["adv_flow_has_test"] = int(n_test > 0)
    f["adv_flow_has_edit"] = int(n_edit > 0)
    first_test = next((i for i, t in enumerate(is_test) if t), -1)
    f["adv_flow_first_test_step"] = first_test + 1 if first_test >= 0 else -1
    f["adv_flow_edit_then_test"] = sum(
        1 for i in range(n - 1) if is_edit[i] and is_test[i + 1]
    )
    productive = sum(1 for t in types if t in ("read", "search", "edit", "test"))
    f["adv_flow_productive_frac"] = productive / n
    f["adv_flow_env_other_frac"] = sum(1 for t in types if t in ("environment", "other")) / n
    f["adv_flow_blind_edits"] = n_edit if n_test == 0 else 0
    f["adv_flow_new_files_last5"] = sum(new_file_flags[-5:])
    # edits since last progress (churning without getting anywhere)
    last_prog = max((i for i, p in enumerate(progress_flags) if p), default=-1)
    f["adv_flow_edits_since_progress"] = sum(is_edit[last_prog + 1:])
    # consecutive test runs (from the end) that did NOT reduce failures
    test_fails = [s.test_fail_count for s in steps if s.is_test_command and s.test_fail_count is not None]
    streak = 0
    for j in range(len(test_fails) - 1, 0, -1):
        if test_fails[j] >= test_fails[j - 1]:
            streak += 1
        else:
            break
    f["adv_flow_test_no_improve_streak"] = streak
    # over-deliberation: thought length vs action length (recent)
    rt = steps[-8:]
    mean_th = sum(len(s.thought_text or "") for s in rt) / len(rt)
    mean_ac = sum(len(s.action_text or "") for s in rt) / len(rt)
    f["adv_flow_thought_action_ratio"] = mean_th / max(1.0, mean_ac)

    # ---- targeted interactions ------------------------------------------
    edited_before_read = float(f.get("edited_before_any_read", 0) or 0)
    max_rep = float(f.get("max_command_repeat_count", 0) or 0)
    f["adv_ix_premature_edit"] = edited_before_read * n_edit
    f["adv_ix_repeat_x_err"] = max_rep * f["adv_err_rate"]
    f["adv_ix_blind_edit"] = (1.0 if n_test == 0 else 0.0) * n_edit
    f["adv_ix_stall_x_churn"] = f["adv_ts_progress"] * float(f.get("same_file_edit_count_max", 0) or 0)

    return f


def advanced_family_columns(all_cols: list[str], families: list[str], feature_prefix: str = "f__") -> list[str]:
    """Resolve advanced family names to present f__ columns (for ablation)."""
    wanted: list[str] = []
    for fam in families:
        pref = ADV_FAMILY_PREFIXES.get(fam)
        if not pref:
            continue
        for c in all_cols:
            if c.startswith(feature_prefix + pref) and c not in wanted:
                wanted.append(c)
    return wanted
