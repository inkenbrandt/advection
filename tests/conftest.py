"""Pytest configuration for the ``advection`` test suite.

Makes the in-tree package importable without an install by putting the ``src``
directory on ``sys.path``. This lets ``import advection`` succeed when the tests
are run straight from a checkout that has not been ``pip install``-ed; when the
package is already installed (editable or otherwise) it is harmless because the
path is only prepended, not replaced.
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
