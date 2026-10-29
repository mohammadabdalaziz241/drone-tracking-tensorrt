import sys
from pathlib import Path

# The scripts are standalone files, not a package; make them importable in tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
