import sys
from pathlib import Path

# Add project root to sys.path so 'core', 'agents', 'config' can be imported anywhere
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
