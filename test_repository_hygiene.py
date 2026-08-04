from pathlib import Path
import re


def test_no_unresolved_merge_markers():
    root = Path(__file__).resolve().parent
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".txt", ".json", ".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if ("<" * 7 + " HEAD") in text or (">" * 7 + " ") in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"unresolved merge markers: {offenders}"


def test_clean_checkout_has_no_unpublished_runtime_dependencies():
    """Install metadata must not reference private or unpublished package indexes.

    This check intentionally inspects the source metadata as text.  It does not
    need a TOML parser, which keeps the dependency-light Python 3.10 CI lane
    runnable with only pytest installed.
    """
    root = Path(__file__).resolve().parent
    source = "\n".join(
        [
            (root / "pyproject.toml").read_text(encoding="utf-8"),
            (root / "requirements.txt").read_text(encoding="utf-8"),
        ]
    )
    # Strip TOML/requirements comments before checking requirement syntax.
    # URLs and explanatory prose may legitimately name the external projects.
    metadata = "\n".join(line.split("#", 1)[0] for line in source.splitlines())

    for package in ("high-trust-evidence", "qualification-contract"):
        declared_requirement = re.compile(
            rf"(?im)(?:^\s*|[\"']){re.escape(package)}"
            rf"(?=$|[<>=!~;\s\"'])"
        )
        assert not declared_requirement.search(metadata), (
            f"unpublished runtime dependency declared: {package}"
        )
    assert (root / "evidence" / "runtime" / "contract.py").is_file()
    assert (root / "evidence" / "runtime" / "sovereign.py").is_file()
