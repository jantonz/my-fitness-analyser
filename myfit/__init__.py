# -*- coding: utf-8 -*-
"""Top level package for Mi Fitness Analyser."""
import os
from importlib import metadata
from pathlib import Path

__version__ = metadata.version("myfit")

WORKDIR = Path(os.getenv("WORKDIR", Path.cwd()))
BASEPATH = Path(__file__).parent
