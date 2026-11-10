"""Lets the package run as a program: python -m complaints --help"""

import sys

from .cli import main

sys.exit(main())
