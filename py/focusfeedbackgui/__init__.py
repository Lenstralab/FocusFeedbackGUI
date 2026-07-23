from importlib.metadata import version
from pathlib import Path

try:
    __version__ = version(Path(__file__).parent.name)
except (Exception,):
    __version__ = "unknown"
