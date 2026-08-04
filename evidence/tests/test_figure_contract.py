from pathlib import Path

def _repo_root() -> Path:
    """Walk up to the repository root.

    Counting parents breaks whenever a file moves. Locating the root by a
    marker survives relocation, which is what happened when these tests were
    grouped under `evidence/tests/`.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]



import hashlib, json

ROOT = _repo_root()
REQUIRED=("**Question.**","**Observation.**","**Interpretation.**","**Inference status.**")

def test_figure_contract_is_versioned_and_linked():
    standard=(ROOT / "evidence" / "docs" / "FIGURE_STANDARD.md").read_text(encoding="utf-8")
    fmap=(ROOT / "evidence" / "docs" / "FIGURE_MAP.md").read_text(encoding="utf-8")
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    for h in ("Question","Observation","Interpretation","Inference status"):
        assert h in standard
    assert "FIGURE_STANDARD.md" in readme and "FIGURE_MAP.md" in readme
    assert "Single question" in fmap

def test_every_maritime_figure_has_complete_caption_and_sidecar():
    doc=(ROOT/"docs"/"figures"/"FIGURES.md").read_text(encoding="utf-8")
    figures=sorted((ROOT/"figures").glob("*.png"))
    assert len(figures)==2
    for fig in figures:
        assert f"## {fig.name}" in doc
        block=doc.split(f"## {fig.name}",1)[1].split("\n---\n",1)[0]
        for field in REQUIRED: assert field in block
        meta_path=fig.with_suffix(".figure.json")
        assert meta_path.exists()
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["sha256"]==hashlib.sha256(fig.read_bytes()).hexdigest()
        assert meta["research_phase"]=="phase_1_construct_validity_diagnostic"
        assert meta["inference_status"]=="descriptive_geometry_only"
        assert "deployable intent monitor validated" in meta["unsupported_conclusions"]

def test_phase1_inference_footer_blocks_semantic_overclaim():
    doc=(ROOT/"docs"/"figures"/"FIGURES.md").read_text(encoding="utf-8").lower()
    assert "phase 1" in doc
    assert "do not license semantic interpretation" in doc
    assert "deployable" in doc
