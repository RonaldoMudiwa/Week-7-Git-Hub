"""All the errors this package can raise.

Everything inherits from ComplaintError, so one except clause catches the lot.
"""

from __future__ import annotations


class ComplaintError(Exception):
    """Base class for every error in this package. Never raised directly."""


class ValidationError(ComplaintError):
    """A value isn't allowed, e.g. a complaint with no text."""


class DataError(ComplaintError):
    """A data file is missing, or isn't in the shape we expect."""
