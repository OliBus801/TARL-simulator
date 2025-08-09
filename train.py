"""Deprecated training entry point.

This file is kept for backwards compatibility and simply forwards to
``main.py``.  It will be removed in a future release."""
import warnings
from main import main

if __name__ == "__main__":
    warnings.warn("train.py is deprecated; use main.py with --algo mpnn+ppo", DeprecationWarning)
    main()
