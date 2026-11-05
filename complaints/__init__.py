"""Banking Complaint Analyzer.

Reads customer complaints sent to banks and works out what they're about,
how upset and how urgent they are, and what goes wrong most often.

    from complaints import CsvComplaintSource, TextCleaner
"""

from .exceptions import ComplaintError, DataError, ValidationError
from .models import Complaint
from .sources import ComplaintSource, CsvComplaintSource, ListSource, LoadReport, parse_date
from .text import STOP_WORDS, TextCleaner, ngrams

__version__ = "0.1.0"

__all__ = [
    "Complaint",
    "ComplaintSource",
    "CsvComplaintSource",
    "ListSource",
    "LoadReport",
    "parse_date",
    "TextCleaner",
    "ngrams",
    "STOP_WORDS",
    "ComplaintError",
    "DataError",
    "ValidationError",
    "__version__",
]
