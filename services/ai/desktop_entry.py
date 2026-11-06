"""PyInstaller entry script for the desktop engine (see scripts/package_desktop.py).

``app`` is a package with relative imports, so the frozen executable needs a top-level
script that imports it rather than running app/desktop.py as ``__main__``.
"""

import multiprocessing
import sys

from app.desktop import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
