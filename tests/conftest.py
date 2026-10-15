"""Make the gow_rag package importable when running `pytest` from the project folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
