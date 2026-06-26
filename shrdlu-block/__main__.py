"""Run a SHRDLU block-world agent."""

import sys
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.run_agents import main_shrdlu as main


if __name__ == '__main__':
    raise SystemExit(main())
