"""Makes `python -m reviewgraph ...` work: Python runs this file for `-m <package>`."""

import sys

from reviewgraph.cli import main

sys.exit(main())
