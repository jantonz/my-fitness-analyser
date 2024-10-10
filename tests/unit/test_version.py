# -*- coding: utf-8 -*-
"""Test for version."""
from myfit import __version__


def test_version() -> None:
    """Test version."""
    assert __version__ == "0.1.0", "Version is not the expected one."
