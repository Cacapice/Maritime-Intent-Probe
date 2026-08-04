"""CLI compatibility entry point for the dependency-light model-blind witness."""
from science.compute_model_blind_witness import *  # noqa: F401,F403

if __name__ == "__main__":
    from science.compute_model_blind_witness import main
    main()
