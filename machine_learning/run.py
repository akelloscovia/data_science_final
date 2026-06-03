"""Proxy launcher for the application when executed from the machine_learning folder."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run import app

if __name__ == "__main__":
    app.run(debug=True)
