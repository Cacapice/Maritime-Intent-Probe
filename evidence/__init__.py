"""Evidence-platform integration for this repository.

Everything here belongs to the shared publication contract rather than
to the science. The repository root holds the research.

  adapter.py     translates this repository's native result into a
                 ScientificResult, and declares the obligations it owes
  sovereign.py   re-export of the platform's sovereign release layer
  examples/      publication and verification examples
  tests/         contract, conformance and governance tests
  docs/          shared platform documentation
"""

from .adapter import *  # noqa: F401,F403
