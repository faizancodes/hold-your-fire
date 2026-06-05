"""Action parser coverage (Phase 3 verification gate)."""

from localguard.action_parser import (
    classify_action,
    detect_traceback,
    extract_file_paths,
    is_test_path,
    parse_test_counts,
)


def test_pytest_classified_as_test():
    assert classify_action("python -m pytest tests/test_x.py").action_type == "test"
    assert classify_action("pytest").is_test_command


def test_reproduce_script_is_test():
    assert classify_action("python reproduce.py").action_type == "test"
    assert classify_action("python test_widget.py").action_type == "test"
    # an unrelated python script must NOT be misclassified as a test
    assert classify_action("python tokens.py").action_type != "test"


def test_search_commands():
    for cmd in ["rg 'def foo'", "grep -n x file", "search_dir 'class'", "find_file a.py"]:
        assert classify_action(cmd).action_type == "search", cmd


def test_read_commands():
    for cmd in ["cat src/a.py", "sed -n '1,20p' a.py", "head a.py", "less a.py"]:
        assert classify_action(cmd).action_type == "read", cmd


def test_edit_commands():
    assert classify_action("apply_patch").action_type == "edit"
    assert classify_action("create reproduce.py").action_type == "edit"
    assert classify_action("sed -i 's/a/b/' f.py").action_type == "edit"


def test_git_and_submit():
    assert classify_action("git diff").action_type == "git"
    assert classify_action("git status").is_git_command
    assert classify_action("submit").action_type == "submit"
    assert classify_action("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT").is_submit_command


def test_file_path_extraction():
    paths = extract_file_paths("edit src/app/models.py and tests/test_x.py")
    assert "src/app/models.py" in paths
    assert "tests/test_x.py" in paths
    assert is_test_path("tests/test_x.py")
    assert not is_test_path("src/app/models.py")


def test_test_count_parsing():
    assert parse_test_counts("===== 1 failed, 4 passed in 0.2s =====") == (4, 1)
    assert parse_test_counts("5 passed") == (5, 0)
    assert parse_test_counts("FAILED tests/test_x.py::test_y") == (None, 1)
    assert parse_test_counts("no test output here") == (None, None)


def test_traceback_detection():
    assert detect_traceback("Traceback (most recent call last):\n ...")
    assert not detect_traceback("all good")


def test_priority_python_pytest_over_other():
    # `python -m pytest` must be test, not 'other'
    assert classify_action("python -m pytest -q").action_type == "test"
