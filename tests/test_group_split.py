"""Group-split disjointness (Phase 6 verification gate)."""

from conftest import make_synthetic_prefix_df

from localguard.split import make_splits, parse_repo, verify_disjoint


def test_instance_split_disjoint():
    df = make_synthetic_prefix_df(n_traj=200, seed=3)
    splits = make_splits(df, regime="instance", seed=42)
    verify_disjoint(splits)  # raises on overlap
    tr = set(splits.train["instance_id"])
    va = set(splits.val["instance_id"])
    te = set(splits.test["instance_id"])
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_all_rows_accounted_for():
    df = make_synthetic_prefix_df(n_traj=120, seed=4)
    splits = make_splits(df, regime="instance", seed=42)
    total = len(splits.train) + len(splits.val) + len(splits.test)
    assert total == len(df)


def test_parse_repo():
    assert parse_repo("django__django-12345") == "django__django"
    assert parse_repo("scikit-learn__scikit-learn-999") == "scikit-learn__scikit-learn"


def test_repo_split_disjoint():
    df = make_synthetic_prefix_df(n_traj=150, seed=5)
    # give rows repo-style instance ids
    df["instance_id"] = ["org__repo" + str(i % 30) + "-" + str(i) for i in range(len(df))]
    splits = make_splits(df, regime="repo", seed=42)
    verify_disjoint(splits)
