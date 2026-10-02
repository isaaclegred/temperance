import sys
import os

# Add the inner temperance/ directory to sys.path so that the solving
# subpackage can be imported as "solving.X" without triggering the
# top-level temperance/__init__.py (which has optional heavy dependencies).
_inner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temperance")
sys.path.insert(0, _inner)
